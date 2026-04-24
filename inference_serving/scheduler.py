import pandas as pd
from time import time
import csv

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .graph_generator import *
from .trace_generator import *
from .pim_model import *
import numpy as np

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, node_id, instance_id, max_batch, max_num_batched_tokens,
                 npu_num, npu_group, npu_mem, cpu_mem,
                 start_npu, pd_type, fp, block_size, req_num,
                 prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0,
                 hbf_mem=0, enable_hbf_offloading=False, enable_hbf_kv=False):
        # all time realated variables are in using tick (system tick)
        # LLMServingSim uses Orca, vLLM technique at deafult
        self.model = model
        self.config = get_config(model)
        self.node_id = node_id
        self.instance_id = instance_id
        self.max_batch = max_batch
        self.max_num_batched_tokens = min(max_num_batched_tokens, self.config['max_position_embeddings'])
        self.npu_num = npu_num
        self.npu_group = npu_group
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage
        self.prioritize_prefill = prioritize_prefill
        self.enable_hbf_kv = enable_hbf_kv
        # lists are sorted in arrival time manner
        self.request = [] # list of requests
        self.inflight = [] # list of batches
        self.done = [] # list of requests
        # self.req_ids = -1
        self.batch_ids = -1
        self.first_arrival_time = 0

        # memory model
        self.memory = MemoryModel(model, instance_id, node_id, npu_num, npu_group, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem,
                                  hbf_mem, enable_hbf_offloading, enable_hbf_kv)

        # logger
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)
    
 
    def schedule(self, current, sys, batch_id=-1):
        if self.enable_prefix_caching:
            result = self.schedule_with_prefix(current, sys, batch_id)
            # Apply all queued KV cache events (from cache_*/evict_* calls
            # made during scheduling) in a single batched pass.
            self.memory.apply_kv_cache_events()
            return result
        else:
            return self.schedule_base(current, sys, batch_id)

    # batch the request scheduling method
    def schedule_base(self, current, sys, batch_id=-1):
        # first NPU to process new batch
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.npu_group:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            kv_size = 0
            evict_size = 0
            gen_req = [req for req in batch_req if not req.is_init]
            
            if self.prioritize_prefill:
                prefill_req = [req for req in batch_req if req.is_init]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch
                    batch_req = batch_req[:batch_len]
            
            # check if there is request that need to enlarge the block
            temp_len = batch_len
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i) # includes evicted input, and initiation input
                if self._kv_can_fit(kv_size):
                    temp_len = i
                    break

            # no memory to batch
            while temp_len == 0:
                # preempt request one by one untill there is enough space
                if len(gen_req) == 0:
                    return None

                # check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue

                # else
                evict_size += self._evict_req_kv(gen_req[-1])
                gen_req[-1].evict = True
                self.logger.info("Eviction of the request #%d", gen_req[-1].id)
                gen_req = gen_req[:-1]

                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)

                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i)
                    if self._kv_can_fit(kv_size):
                        temp_len = i
                        break

            batch_len = temp_len
            batch_req = batch_req[:batch_len]
            load_size = 0

            # check max_num_batched_tokens constraint
            total_len = 0
            for req in batch_req:
                if req.is_init:
                    total_len += req.input
                else:
                    total_len += 1

            while total_len > self.max_num_batched_tokens:
                if batch_req[-1].is_init:
                    total_len -= batch_req[-1].input
                else:
                    total_len -= 1
                
                batch_req = batch_req[:-1]
                batch_len -= 1

            # recompute kv_size
            kv_size = self.memory.get_block_kv(batch_req, batch_len) # includes evicted input, and initiation input

            # delete from request queue
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.evict:
                    # load evicted kv cache
                    load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            # Allocate Needed KV caches for current batch
            if self.enable_hbf_kv:
                self._allocate_kv_blocks(batch_req, batch_len)
            else:
                if kv_size > 0:
                    self.memory.allocate(kv_size, Device.NPU)

            # load memory from cpu (host)
            if load_size > 0:
                self.memory.free(load_size, Device.CPU)
            
            total_len = 0
            kv_len = 0
            hit_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            for req in batch_req:
                if req.is_init:
                    total_len += req.input
                    req.set_que_delay(current)
                    q_list.append(req.input)
                    prefill_q_list.append(req.input)
                    # For now, we don't assume chunked prefill
                    prefill_k_list.append(0)
                    num_prefill += 1
                else:
                    total_len += 1
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.input
                    decode_k_list.append(req.input)
                k_list.append(req.input)

            # compute hbf_kv_len for trace generation (token-granularity)
            hbf_kv_len = 0
            if self.enable_hbf_kv:
                for req in batch_req:
                    if not req.is_init:
                        total_blocks = req.npu_kv_blocks + req.hbf_kv_blocks
                        if total_blocks > 0:
                            hbf_kv_len += req.input * req.hbf_kv_blocks // total_blocks

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, hit_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, load_size, hbf_kv_len)
            # add alredy fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            return batch
        
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch == None:
                    return None
                # check if this has been runned in the system
                if sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
    
    def schedule_with_prefix(self, current, sys, batch_id=-1):
        if sys == self.start_npu:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.npu_group:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            if self.prioritize_prefill:
                prefill_req = [req for req in batch_req if req.is_init]

                if len(prefill_req) != 0:
                    batch_req = prefill_req
                    batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch
                    batch_req = batch_req[:batch_len]
        
            for req in batch_req:
                if req.is_init:
                    self.memory.prefix_match(req)
                    # self.memory.npu_lock_prefix(req)
                    self.memory.lock_prefix(req, Device.NPU)
            
            kv_size = 0
            evict_size = 0
            gen_req = [req for req in batch_req if not req.is_init]
            # check if there is request that need to enlarge the block
            temp_len = batch_len
            total_useable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)
            
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i) # includes evicted input, and initiation input
                if total_useable_size >= kv_size:
                    temp_len = i
                    break
            
            evicted_req = []
            # no memory to batch
            while temp_len == 0:
                if len(gen_req) == 0: # there is no request to evict but no memory
                    # rollback prefix cache lock ref
                    for req in batch_req:
                        if req.is_init:
                            # self.memory.npu_unlock_prefix(req)
                            self.memory.unlock_prefix(req, Device.NPU)
                            self.memory.erase_prefix_info(req)
                    return None
                
                # check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue
                
                # else
                # self.memory.npu_unlock_prefix(gen_req[-1])
                self.memory.unlock_prefix(gen_req[-1], Device.NPU)
                self.memory.erase_prefix_info(gen_req[-1])

                current_usable_size = self.memory.avail_size(Device.NPU) + self.memory.evictable_size(Device.NPU)

                gen_req[-1].evict = True
                evicted_req.append(gen_req[-1])
                self.logger.info("Eviction of the request #%d", gen_req[-1].id)
                gen_req = gen_req[:-1]

                if len(gen_req) < batch_len: # prefill is always at last
                    batch_len = len(gen_req)
                
                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i)
                    if current_usable_size >= kv_size:
                        temp_len = i
                        break

            for req in batch_req[temp_len:]:
                if req.is_init:
                    # self.memory.npu_unlock_prefix(req)
                    self.memory.unlock_prefix(req, Device.NPU)
                    self.memory.erase_prefix_info(req)

            batch_len = temp_len
            batch_req = batch_req[:batch_len]

            # check max_num_batched_tokens constraint
            total_len = 0
            for req in batch_req:
                if req.is_init:
                    total_len += req.input
                else:
                    total_len += 1

            while total_len > self.max_num_batched_tokens:
                if batch_req[-1].is_init:
                    total_len -= batch_req[-1].input
                else:
                    total_len -= 1
                
                if batch_req[-1].is_init:
                    # self.memory.npu_unlock_prefix(batch_req[-1])
                    self.memory.unlock_prefix(batch_req[-1], Device.NPU)
                    self.memory.erase_prefix_info(batch_req[-1])

                batch_req = batch_req[:-1]
                batch_len -= 1

            # recompute kv_size
            kv_size = self.memory.get_block_kv(batch_req, batch_len) # includes evicted input, and initiation input
            npu_avail = self.memory.avail_size(Device.NPU)
            evict_size = (kv_size - npu_avail) if kv_size > npu_avail else 0

            if evict_size > 0:
                # self.memory.npu_evict_prefix_cache(evict_size)
                self.memory.evict_prefix_cache(evict_size, Device.NPU)

            evict_load_size = 0
            prefix_load_size = 0
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.is_init and req.storage_cache_hit > req.prefix_cache_hit:
                    # load prefix cache
                    prefix_load_size += (req.storage_cache_hit - req.prefix_cache_hit) * self.memory.get_kv(1)

                if req.evict:
                    # load evicted kv cache
                    self.memory.prefix_match(req)
                    # self.memory.npu_lock_prefix(req)
                    self.memory.lock_prefix(req, Device.NPU)
                    # self.memory.cpu_unlock_prefix(req)
                    if self.prefix_storage is not None:
                        self.memory.unlock_prefix(req, Device.CPU)
                    evict_load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    self.logger.info("Loading the request #%d", req.id)

            total_len = 0
            kv_len = 0
            hit_len = 0
            num_prefill = 0
            num_decode = 0
            q_list = []
            k_list = []
            prefill_q_list = []
            prefill_k_list = []
            decode_k_list = []
            
            # evict cpu prefix cache if needed
            total_size = 0
            for req in batch_req:
                total_size += self.memory.get_total_kv(req) * self.npu_num
            for req in evicted_req:
                total_size += self.memory.get_total_kv(req) * self.npu_num
            
            if self.prefix_storage is not None:
                storage_avail = self.memory.avail_size(self.prefix_storage)
                storage_evict_size = (total_size - storage_avail) if total_size > storage_avail else 0
                
                if storage_evict_size > 0:
                    # self.memory.cpu_evict_prefix_cache(cpu_evict_size)
                    self.memory.evict_prefix_cache(storage_evict_size, self.prefix_storage)

            for req in batch_req:
                # Update the prefix cache for incoming batch
                self.memory.cache_unfinished_req(req, Device.NPU)
                if self.prefix_storage is not None:
                    self.memory.cache_unfinished_req(req, self.prefix_storage)
                if req.is_init:
                    total_len += req.input
                    req.set_que_delay(current)
                    if self.enable_prefix_caching and req.prefix_cache_hit > 0:
                        hit_len += req.prefix_cache_hit
                    q_list.append(max(req.input - req.prefix_cache_hit, 1))
                    num_prefill += 1
                    prefill_q_list.append(max(req.input - req.prefix_cache_hit, 1))
                    prefill_k_list.append(0)
                else:
                    total_len += 1    
                    q_list.append(1)
                    num_decode += 1
                    kv_len += req.input
                    decode_k_list.append(req.input)
                k_list.append(req.input)
            
            # cpu need to hold evicted cache
            if self.prefix_storage is not None:
                for req in evicted_req:
                    self.memory.storage_cache_evicted_req(req)

            # compute hbf_kv_len for trace generation (proportional estimate)
            hbf_kv_len = 0
            if self.enable_hbf_kv and kv_len > 0:
                hbf_kv_bytes = self.memory.hbf_used - self.memory.hbf_floor
                npu_kv_bytes = self.memory.npu_used - self.memory.npu_floor
                total_kv_bytes = npu_kv_bytes + hbf_kv_bytes
                if total_kv_bytes > 0:
                    hbf_kv_len = kv_len * hbf_kv_bytes // total_kv_bytes

            # For debugging
            # self.memory.npu_prefix_cache.pretty_print()
            # self.memory.npu_prefix_cache.print_prefix_info()
            batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, hit_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, current, kv_size, evict_size, evict_load_size + prefix_load_size, hbf_kv_len=hbf_kv_len)
            # add alredy fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            self.logger.info(
                "Scheduling new batch #%d to NPU[%d]",
                batch.batch_id,
                sys,
            )
            return batch
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch is None or sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    self.logger.info(
                        "Scheduling existing batch #%d to NPU[%d]",
                        batch.batch_id,
                        sys,
                    )
                    return batch
        
    # pop inflight, add to done
    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs
        batch = None
        # find batch
        id -= 1
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            return prompt_t, gen_t, end_reqs
        # already done
        if sys in batch.end:
            return prompt_t, gen_t, end_reqs
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            if self.pd_type != "prefill":
                if self.start_npu not in batch.end or (self.start_npu + self.npu_num - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
            else:
                if self.start_npu not in batch.end or (self.start_npu + self.npu_num * 2 - 1) not in batch.end:
                    return prompt_t, gen_t, end_reqs
        self.logger.info(
            "Batch #%d is done",
            batch.batch_id,
        )
                
        pool = []
        for req in batch.requests:
            # change phase
            if req.is_init:
                req.is_init = False
                if self.pd_type != "prefill":
                    prompt_t += req.input
                    gen_t += 1
                    req.set_ttft(finish)
                else: # prefill instance
                    prompt_t += req.input
                    gen_t += 1
                    req.set_ttft(finish)
                    self.logger.info(
                    "Request #%d is prefill done",
                    req.id,
                    )
                        
                    # sending is done. clean this batch in prefill instance
                    self.logger.info("Request #%d is sent to decode instance", req.id)
                    req.input += 1
                    
                    # remove kv cache here
                    if self.enable_prefix_caching:
                        self.memory.unlock_prefix(req, Device.NPU)
                    else:
                        self._free_req_kv(req)

                    end_reqs.append(req)
                    continue # pass generation phase and continue
            else:
                gen_t += 1
                req.add_itl(finish)

            req.input += 1

            # check done
            if req.output <= req.input:
                self.logger.info("Request #%d is done", req.id)
                # remove kv cache here
                if self.enable_prefix_caching:
                    self.memory.cache_finished_req(req, Device.NPU) # insert happens here
                    if self.prefix_storage is not None:
                        self.memory.cache_finished_req(req, Device.CPU)
                else:
                    self._free_req_kv(req)
                req.add_latency(finish)
                self.done.append(req)
                end_reqs.append(req)

            # return to pool
            else:
                pool.append(req)
        # return to request pool, both are already sorted with arrival_time
        if self.prioritize_prefill:
            self.request = self._merge_by_arrival_id(pool, self.request)
        else:
            self.request = pool + self.request

        del self.inflight[idx]
        del batch
        # Apply KV cache events queued by cache_finished_req / unlock_prefix
        # above in a single batched pass.
        if self.enable_prefix_caching:
            self.memory.apply_kv_cache_events()
        return prompt_t, gen_t, end_reqs
    

    ##### Helper Functions ######

    # --- HBF KV cache helpers ---

    def _one_block_kv_size(self):
        return self.memory.get_kv(self.memory.block_size)

    def _kv_can_fit(self, kv_size):
        """Check if kv_size fits in NPU (+ HBF when enabled)."""
        if self.enable_hbf_kv:
            one_blk = self._one_block_kv_size()
            npu_blocks = (self.memory.npu_mem - self.memory.npu_used) // one_blk
            hbf_blocks = (self.memory.hbf_mem - self.memory.hbf_used) // one_blk
            needed_blocks = (kv_size + one_blk - 1) // one_blk
            return (npu_blocks + hbf_blocks) >= needed_blocks
        return self.memory.is_avail(kv_size, Device.NPU)

    def _allocate_kv_blocks(self, batch_req, batch_len):
        """Allocate KV blocks per-request: NPU first, overflow to HBF."""
        one_blk = self._one_block_kv_size()
        for i in range(batch_len):
            if batch_req[i].evict or batch_req[i].is_init:
                hit = getattr(batch_req[i], 'npu_cache_hit', 0) if self.enable_prefix_caching else 0
                needed = max(0, batch_req[i].input - hit)
                num_blocks = needed // self.memory.block_size + 1
            else:
                num_before = (batch_req[i].input - 1) // self.memory.block_size + 1
                num_after = batch_req[i].input // self.memory.block_size + 1
                num_blocks = max(0, num_after - num_before)

            for _ in range(num_blocks):
                if self.memory.is_avail(one_blk, Device.NPU):
                    self.memory.allocate(one_blk, Device.NPU)
                    batch_req[i].npu_kv_blocks += 1
                else:
                    self.memory.allocate(one_blk, Device.HBF)
                    batch_req[i].hbf_kv_blocks += 1

    def _evict_req_kv(self, req):
        """Evict a request's KV cache from NPU/HBF to CPU. Returns total evicted size."""
        if self.enable_hbf_kv:
            one_blk = self._one_block_kv_size()
            npu_free = req.npu_kv_blocks * one_blk
            hbf_free = req.hbf_kv_blocks * one_blk
            total = npu_free + hbf_free
            if npu_free > 0:
                self.memory.free(npu_free, Device.NPU)
            if hbf_free > 0:
                self.memory.free(hbf_free, Device.HBF)
            self.memory.allocate(total, Device.CPU)
            req.npu_kv_blocks = 0
            req.hbf_kv_blocks = 0
            return total
        else:
            evict_size = self.memory.get_evict_kv(req)
            self.memory.free(evict_size, Device.NPU)
            self.memory.allocate(evict_size, Device.CPU)
            return evict_size

    def _free_req_kv(self, req):
        """Free a completed/prefilled request's KV cache from NPU/HBF."""
        if self.enable_hbf_kv:
            one_blk = self._one_block_kv_size()
            npu_kv = req.npu_kv_blocks * one_blk
            hbf_kv = req.hbf_kv_blocks * one_blk
            if npu_kv > 0:
                self.memory.free(npu_kv, Device.NPU)
            if hbf_kv > 0:
                self.memory.free(hbf_kv, Device.HBF)
            req.npu_kv_blocks = 0
            req.hbf_kv_blocks = 0
        else:
            kv_size = self.memory.get_evict_kv(req)
            self.memory.free(kv_size, Device.NPU)

    # get new batch id
    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    # add a request
    def add_request(self, req, is_init=True):
        new_req = Request(*(req), is_init=is_init)
        self.request.append(new_req)
        return
    
    # add decode request to decode instance from prefill instnace
    def add_decode(self, req):
        self.request.append(req)
        if self.enable_hbf_kv:
            num_blocks = (req.input - 1) // self.memory.block_size + 1
            one_blk = self._one_block_kv_size()
            for _ in range(num_blocks):
                if self.memory.is_avail(one_blk, Device.NPU):
                    self.memory.allocate(one_blk, Device.NPU)
                    req.npu_kv_blocks += 1
                else:
                    self.memory.allocate(one_blk, Device.HBF)
                    req.hbf_kv_blocks += 1
        else:
            kv_size = self.memory.get_total_kv(req)
            self.memory.allocate(kv_size, Device.NPU)
    
    # get first request's arrival time
    def get_first_arrival_time(self):
        return self.first_arrival_time if self.first_arrival_time != 0 else 1 # need to add event handler at first
    
    # merge requests in the request pool, ensuring they are sorted by arrival time
    def _merge_by_arrival_id(self, left, right):
        if not left:  
            return right
        if not right: 
            return left

        # Fast path: if ranges don't overlap, just concatenate
        if (left[-1].arrival, left[-1].id) <= (right[0].arrival, right[0].id):
            return left + right
        if (right[-1].arrival, right[-1].id) <= (left[0].arrival, left[0].id):
            return right + left

        # General merge
        i = j = 0
        out = []
        while i < len(left) and j < len(right):
            li, rj = left[i], right[j]
            if (li.arrival, li.id) <= (rj.arrival, rj.id):
                out.append(li); i += 1
            else:
                out.append(rj); j += 1
        if i < len(left):  
            out.extend(left[i:])
        if j < len(right): 
            out.extend(right[j:])
        return out
    
    # print total system request metrics (TTFT, TPOT, ITL)
    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        print("------------------------------Time to First Token-------------------------------")
        if ttft_values:
            mean = np.mean(ttft_values) / 1000_000
            median = np.median(ttft_values) / 1000_000
            p99 = np.percentile(ttft_values, 99) / 1000_000
            print(f"Mean TTFT (ms):                                                     {mean:.2f}")
            print(f"Median TTFT (ms):                                                   {median:.2f}")
            print(f"P99 TTFT (ms):                                                      {p99:.2f}")
        else:
            print("No TTFT data available")

        print("--------------------Time per Output Token (excl. 1st token)---------------------")
        if tpot_values:
            mean = np.mean(tpot_values) / 1000_000
            median = np.median(tpot_values) / 1000_000
            p99 = np.percentile(tpot_values, 99) / 1000_000
            print(f"Mean TPOT (ms):                                                     {mean:.2f}")
            print(f"Median TPOT (ms):                                                   {median:.2f}")
            print(f"P99 TPOT (ms):                                                      {p99:.2f}")
        else:
            print("No TPOT data available")

        print("------------------------------Inter-token Latency-------------------------------")
        if itl_values:
            mean = np.mean(itl_values) / 1000_000
            median = np.median(itl_values) / 1000_000
            p99 = np.percentile(itl_values, 99) / 1000_000
            print(f"Mean ITL (ms):                                                      {mean:.2f}")
            print(f"Median ITL (ms):                                                    {median:.2f}")
            print(f"P99 ITL (ms):                                                       {p99:.2f}")
        else:
            print("No ITL data available")

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def is_request_empty(self):
        if len(self.request) == 0 and len(self.inflight) == 0:
            return True
        else:
            return False
        
    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        output_file = f'../{output_file}'
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow(['instance id', 'request id', 'model', 'input', 'output', 
                                'arrival', 'end_time', 'latency', 
                                'queuing_delay', 'TTFT', 'TPOT', 'ITL'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.model,
                    req.input,
                    req.output,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot,
                    req.itl
                ])