import os
import pandas as pd
import random
from time import time
from .logger import get_logger

class Router:
    _TOKEN_ID_MIN = 1
    _TOKEN_ID_MAX = 128000

    def __init__(
            self, 
            num_instances, 
            schedulers, req_num, 
            routing_policy="RR", 
            seed=42
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.instance_status = [0 for _ in range(num_instances)]
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0
        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND":
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "CUSTOM":
            self.routing_fn = self._custom_routing_policy
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, CUSTOM")
        self.logger = get_logger(self.__class__)

    def _rr_routing(self, request_ctr, num_instances):
        return request_ctr % num_instances

    def _rand_routing(self, request_ctr, num_instances):
        return self._rnd.randrange(num_instances)
    
    def _custom_routing_policy(self, request_ctr, num_instances):
        raise NotImplementedError("Implement custom routing policy.")

    def _max_schedulable_input_tokens(self):
        if not self.prefill_schedulers:
            return None
        return min(s.max_num_batched_tokens for s in self.prefill_schedulers)

    def _is_schedulable_row(self, row):
        max_input_tokens = self._max_schedulable_input_tokens()
        if max_input_tokens is None:
            return True
        return int(row['input_length']) <= int(max_input_tokens)

    def _dataset_path(self, path):
        if os.path.isabs(path):
            return path
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(repo_root, path)

    def _is_jsonl_dataset(self, dataset_spec):
        return os.path.isfile(self._dataset_path(dataset_spec))

    def _parse_synthetic_dataset_spec(self, dataset_spec):
        parts = dataset_spec.split(':', 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "Synthetic dataset spec must use the form ARRIVAL:LENGTH"
            )
        return parts[0], parts[1]

    def _infer_synthetic_arrival_rps(self, arrival_name):
        from dataset.dataset import ArrivalTimes

        arrival_times = list(ArrivalTimes.load(arrival_name, load_scale=1.0).arrival_times)
        if len(arrival_times) < 2:
            raise ValueError(
                f"Arrival dataset '{arrival_name}' must contain at least two arrivals to infer --rps"
            )

        # Use the mean inter-arrival rate so the initial startup gap does not skew the target.
        duration = float(arrival_times[-1]) - float(arrival_times[0])
        if duration <= 0:
            raise ValueError(
                f"Arrival dataset '{arrival_name}' must have strictly increasing arrival times to infer --rps"
            )
        return (len(arrival_times) - 1) / duration

    def resolve_load_scale(self, dataset_spec, load_scale=1.0, rps=None):
        if rps is None:
            return load_scale, None
        if dataset_spec is None:
            raise ValueError("--rps requires --dataset ARRIVAL:LENGTH")
        if rps <= 0:
            raise ValueError("--rps must be > 0")
        if self._is_jsonl_dataset(dataset_spec):
            raise ValueError(
                "--rps is only supported for synthetic ARRIVAL:LENGTH datasets"
            )

        arrival_name, _ = self._parse_synthetic_dataset_spec(dataset_spec)
        native_rps = self._infer_synthetic_arrival_rps(arrival_name)
        return rps / native_rps, native_rps

    def _parse_window_spec(self, window):
        if window is None:
            return None
        parts = window.split(':', 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "--window must use START:END for request indices or tSTART:END for arrival-time range"
            )
        start_raw, end_raw = parts
        if start_raw.startswith('t'):
            return {
                'mode': 'time',
                'start': float(start_raw[1:]),
                'end': float(end_raw),
            }
        return {
            'mode': 'index',
            'start': int(start_raw),
            'end': int(end_raw),
        }

    def _apply_time_window_to_arrivals(self, arrivals, window_spec):
        if window_spec is None:
            if self.req_num is None:
                return list(arrivals)
            return list(arrivals)[:self.req_num]
        if window_spec['mode'] != 'time':
            raise ValueError("Internal error: time window required")
        start = window_spec['start']
        end = window_spec['end']
        if start < 0 or end < start:
            raise ValueError(f"Invalid time window 't{start}:{end}'")
        filtered = [
            float(t) - start for t in arrivals
            if start <= float(t) <= end
        ]
        if self.req_num is None:
            return filtered
        return filtered[:self.req_num]

    def _generate_token_ids(self, length, seed_material):
        if length <= 0:
            return []
        rng = random.Random(f"{self.seed}:{seed_material}")
        return [
            rng.randint(self._TOKEN_ID_MIN, self._TOKEN_ID_MAX)
            for _ in range(length)
        ]

    def _build_input_token_ids(self, request, index):
        prompt = getattr(request, 'prompt', None)
        input_length = int(request.input_length)

        if isinstance(prompt, (list, tuple)):
            prompt_ids = [int(tok) for tok in prompt[:input_length]]
            if len(prompt_ids) >= input_length:
                return prompt_ids
            pad_ids = self._generate_token_ids(
                input_length - len(prompt_ids),
                f"prompt-pad:{getattr(request, 'session_id', None)}:{index}",
            )
            return prompt_ids + pad_ids

        cached_length = max(0, min(int(getattr(request, 'cached_length', 0)), input_length))
        session_id = getattr(request, 'session_id', None)
        if cached_length > 0 and session_id is not None:
            shared_prefix = self._generate_token_ids(
                cached_length,
                f"prompt-prefix:{session_id}",
            )
            unique_suffix = self._generate_token_ids(
                input_length - cached_length,
                f"prompt-suffix:{session_id}:{index}",
            )
            return shared_prefix + unique_suffix

        return self._generate_token_ids(
            input_length,
            f"prompt:{session_id}:{index}:{input_length}",
        )

    def _build_output_token_ids(self, request, index):
        output_length = int(request.output_length)
        session_id = getattr(request, 'session_id', None)
        return self._generate_token_ids(
            output_length,
            f"output:{session_id}:{index}:{output_length}",
        )

    def _normalize_jsonl_row(self, row):
        norm = {
            'input_length': int(row['input_toks']),
            'output_length': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
        }
        if 'input_tok_ids' in row:
            norm['input_tok_ids'] = row['input_tok_ids']
        if 'output_tok_ids' in row:
            norm['output_tok_ids'] = row['output_tok_ids']
        return norm

    def _iter_jsonl_dataset_rows(self, path, enable_prefix_caching, window_spec):
        data = pd.read_json(self._dataset_path(path), lines=True)
        if window_spec is not None:
            if window_spec['mode'] == 'index':
                data = data.iloc[window_spec['start']:window_spec['end']]
            else:
                start_ns = int(window_spec['start'] * 1e9)
                end_ns = int(window_spec['end'] * 1e9)
                data = data[(data['arrival_time_ns'] >= start_ns) & (data['arrival_time_ns'] <= end_ns)]
                if not data.empty:
                    data = data.copy()
                    data['arrival_time_ns'] = data['arrival_time_ns'] - start_ns
        if self.req_num is not None:
            data = data.iloc[:self.req_num]

        for _, row in data.iterrows():
            norm = self._normalize_jsonl_row(row)
            if enable_prefix_caching and (
                'input_tok_ids' not in norm or 'output_tok_ids' not in norm
            ):
                raise ValueError(
                    f"Dataset '{path}' is missing token ids required for prefix caching"
                )
            yield norm

    def _iter_synthetic_dataset_rows(self, dataset_spec, load_scale, window_spec):
        from dataset.dataset import ArrivalTimes, Requests

        arrival_name, length_name = self._parse_synthetic_dataset_spec(dataset_spec)
        if window_spec is not None and window_spec['mode'] == 'index':
            requests = Requests.load(
                length_name,
                window_start=window_spec['start'],
                window_end=window_spec['end'],
            )
            arrivals = ArrivalTimes.load(
                arrival_name,
                load_scale=load_scale,
                window_start=window_spec['start'],
                window_end=window_spec['end'],
            )
            request_list = list(requests.requests)[:self.req_num]
            arrival_list = list(arrivals.arrival_times)[:self.req_num]
        else:
            requests = Requests.load(length_name)
            arrivals = ArrivalTimes.load(arrival_name, load_scale=load_scale)
            request_list = list(requests.requests)
            arrival_list = list(arrivals.arrival_times)
            if window_spec is not None and window_spec['mode'] == 'time':
                arrival_list = self._apply_time_window_to_arrivals(arrival_list, window_spec)
                request_list = request_list[:len(arrival_list)]
            else:
                if self.req_num is not None:
                    request_list = request_list[:self.req_num]
                    arrival_list = arrival_list[:self.req_num]

        pair_count = min(len(request_list), len(arrival_list))
        if self.req_num is not None:
            pair_count = min(pair_count, self.req_num)
        if pair_count == 0:
            raise ValueError(
                f"No requests remain after applying dataset window to '{dataset_spec}'"
            )

        for index in range(pair_count):
            request = request_list[index]
            arrival_time_s = float(arrival_list[index])
            yield {
                'input_length': int(request.input_length),
                'output_length': int(request.input_length + request.output_length),
                'arrival_time_ns': int(arrival_time_s * 1e9),
                'input_tok_ids': self._build_input_token_ids(request, index),
                'output_tok_ids': self._build_output_token_ids(request, index),
            }

    def count_dataset_rows(self, dataset_spec, load_scale=1.0, window=None):
        window_spec = self._parse_window_spec(window)
        if self._is_jsonl_dataset(dataset_spec):
            data = pd.read_json(self._dataset_path(dataset_spec), lines=True)
            if window_spec is not None:
                if window_spec['mode'] == 'index':
                    data = data.iloc[window_spec['start']:window_spec['end']]
                else:
                    start_ns = int(window_spec['start'] * 1e9)
                    end_ns = int(window_spec['end'] * 1e9)
                    data = data[(data['arrival_time_ns'] >= start_ns) & (data['arrival_time_ns'] <= end_ns)]
            return len(data) if self.req_num is None else min(len(data), self.req_num)

        from dataset.dataset import ArrivalTimes, Requests

        arrival_name, length_name = self._parse_synthetic_dataset_spec(dataset_spec)
        if window_spec is not None and window_spec['mode'] == 'index':
            requests = Requests.load(
                length_name,
                window_start=window_spec['start'],
                window_end=window_spec['end'],
            )
            arrivals = ArrivalTimes.load(
                arrival_name,
                load_scale=load_scale,
                window_start=window_spec['start'],
                window_end=window_spec['end'],
            )
            count = min(len(requests.requests), len(arrivals.arrival_times))
        else:
            requests = Requests.load(length_name)
            arrivals = ArrivalTimes.load(arrival_name, load_scale=load_scale)
            arrival_count = len(arrivals.arrival_times)
            if window_spec is not None and window_spec['mode'] == 'time':
                start = window_spec['start']
                end = window_spec['end']
                arrival_count = sum(
                    1 for t in arrivals.arrival_times
                    if start <= float(t) <= end
                )
            count = min(len(requests.requests), arrival_count)
        return count if self.req_num is None else min(count, self.req_num)

    def _resolve_dataset_rows(self, dataset_spec, enable_prefix_caching, load_scale, window):
        window_spec = self._parse_window_spec(window)
        if self._is_jsonl_dataset(dataset_spec):
            return self._iter_jsonl_dataset_rows(dataset_spec, enable_prefix_caching, window_spec)
        if load_scale <= 0:
            raise ValueError("--load-scale must be > 0")
        return self._iter_synthetic_dataset_rows(dataset_spec, load_scale, window_spec)

    def transfer_prefill_request(self, requests):
        for req in requests:
            instance_id = self.routing_fn(self.decode_rr_counter, self.decode_instances)
            self.decode_schedulers[instance_id].add_decode(req)
            self.decode_rr_counter += 1

    # generate request to each instance with routing policy
    def generate(self, path, enable_prefix_caching=False, is_init=True, load_scale=1.0, window=None):
        rows = self._resolve_dataset_rows(path, enable_prefix_caching, load_scale, window)
        accepted_index = 0
        skipped_unschedulable = 0
        max_input_tokens = self._max_schedulable_input_tokens()

        for source_index, row in enumerate(rows):
            if not self._is_schedulable_row(row):
                skipped_unschedulable += 1
                self.logger.warning(
                    "Skipping request[%d] with input_length=%d because it exceeds max schedulable input tokens (%d).",
                    source_index,
                    row['input_length'],
                    max_input_tokens,
                )
                continue

            input_length = row['input_length']
            output_length = row['output_length']
            arrival_time_ns = row['arrival_time_ns']
            if enable_prefix_caching:
                # using token ids as hash ids for simplicity
                # change this to add your own hash function
                input_hash_ids = row['input_tok_ids']
                output_hash_ids = row['output_tok_ids']

            if accepted_index == 0:
                # set first arrival time
                for scheduler in self.schedulers:
                    scheduler.first_arrival_time = arrival_time_ns
            
            instance_id = self.routing_fn(self.prefill_rr_counter, self.prefill_instances)
            # add only if instance id matches & add to only prefill schedulers
            if instance_id < 0 or instance_id >= self.prefill_instances:
                raise ValueError(f"Invalid instance_id {instance_id}")
            
            if enable_prefix_caching:
                self.prefill_schedulers[instance_id].add_request([accepted_index, self.prefill_schedulers[instance_id].model, input_length, output_length, arrival_time_ns, instance_id, input_hash_ids, output_hash_ids], is_init=is_init)
            else:
                self.prefill_schedulers[instance_id].add_request([accepted_index, self.prefill_schedulers[instance_id].model, input_length, output_length, arrival_time_ns, instance_id], is_init=is_init)
            self.prefill_rr_counter += 1
            accepted_index += 1

        if skipped_unschedulable > 0:
            self.logger.warning(
                "Skipped %d unschedulable requests while loading dataset '%s'.",
                skipped_unschedulable,
                path,
            )
        
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type) ",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )
        return
