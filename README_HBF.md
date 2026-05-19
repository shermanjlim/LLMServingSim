## Dataset creation 
```sh
# this takes 30 min
PYTHONPATH=. python dataset/download_dataset.py 
```
## Running thebenchmark
```sh
python main.py 
...
--load-scale 1.0    # the scale of the arrival rate, suppose original arrival rate is 2.5 RPS, scale of 2.0 makes it 5.0 RPS;  
--window [t]START:END # the request window for benchmarking; START/END is interpreted as indices by default; with t prefix, START/END is interpreted as time;  
--dataset ARRIVAL:LENGTH # the length and arrival patterns; available length patterns: azure_chat_23, azure_code_23, azure_chat, azure_code, reasoning, deepseek-r1; available arrival pattern azure_chat_23, azure_code_23, azure_chat, azure_code;
```

Example Command 

```sh
python main.py \
    --cluster-config 'cluster_config/single_node_single_instance_hbf.json' \
    --fp 16 --block-size 16 \
    --dataset azure_chat_23:deepseek-r1 \
    --window 0:1000 \
    --load-scale 1.0 \
    --steady-min-s 60 \
    --steady-window 120 \
    --output 'output/example_single_run.csv' \
    --log-interval 1.0 \
    --enable-hbf-offloading \
    --enable-hbf-kv
```

`--steady-min-s` and `--steady-window` switch the simulator into a bounded
steady-state profiling mode. The run warms up for `steady-min-s`, resets
throughput / latency / per-request output / HBF write counters, profiles only
the interval `[steady-min-s, steady-min-s + steady-window)`, then exits.

`--window 0:60` is a request-index slice, not a 60-second slice. Use `--window t0:60`
if you want the first 60 seconds of arrivals instead of the first 60 requests.

`--enable-hbf-offloading` and `--enable-hbf-kv` do different things:

- `--enable-hbf-offloading` places model weights on HBF.
- `--enable-hbf-kv` allows KV blocks to spill to HBF only after NPU KV capacity is exhausted.

With the provided `cluster_config/single_node_single_instance_hbf.json`, Llama-3.1-8B
on an A6000 still has substantial NPU KV headroom after weights are placed, so
`--enable-hbf-kv` may show no change unless the workload is long enough or the load is high
enough to force KV spill.

## Comparing the tradeoff

To compare throughput and latency across all HBF flag combinations in one shot:

```sh
python3 script/hbf_tradeoff.py \
    --cluster-config 'cluster_config/single_node_single_instance_hbf.json' \
    --dataset azure_chat_23:deepseek-r1 \
    --window t0:60 \
    --load-scale 1.0 \
    --log-interval 1.0
```

This runs four scenarios:

- `baseline`
- `hbf_kv_only`
- `hbf_weights_only`
- `hbf_weights_kv`

and writes per-scenario logs/CSVs plus `output/hbf_tradeoff/summary.csv`.

To sweep request-rate multipliers and plot the tradeoff across the three main
scenarios (`no spill`, `weight spill`, `weight+KV spill`):

```sh
python3 script/hbf_tradeoff_sweep.py \
    --cluster-config 'cluster_config/single_node_single_instance_hbf.json' \
    --dataset azure_chat_23:deepseek-r1 \
    --window 0:1000 \
    --request-rates 0.50,0.75,1.00,1.25,1.50
```

This writes:

- `output/hbf_tradeoff_sweep/summary.csv`
- `output/hbf_tradeoff_sweep/tradeoff.png`

The plot contains:

- offered load scale vs average generation throughput
- throughput vs latency (default: mean TPOT)
- throughput vs HBF writes

Failed runs are kept in the summary and marked on the load-scale panel so the
unsupported high-load region is visible instead of stopping the sweep.

For `azure_chat_23` with `--window 0:1000`, the baseline offered arrival rate at
`--load-scale 1.0` is about `4.62 req/s`, so a compute limit near `1 req/s`
corresponds to roughly `--load-scale 0.216`.

This runs for ~60s, example output 
```
--------------------------------------------------------------------------------
                                  Instance [0]                                  
--------------------------------------------------------------------------------
------------------------------Time to First Token-------------------------------
Mean TTFT (ms):                                                     205.84
Median TTFT (ms):                                                   211.61
P99 TTFT (ms):                                                      441.36
--------------------Time per Output Token (excl. 1st token)---------------------
Mean TPOT (ms):                                                     34.87
Median TPOT (ms):                                                   34.92
P99 TPOT (ms):                                                      46.57
------------------------------Inter-token Latency-------------------------------
Mean ITL (ms):                                                      34.35
Median ITL (ms):                                                    27.62
P99 ITL (ms):                                                       213.57
--------------------------------------------------------------------------------
```

with HBF offloading
```
--------------------------------------------------------------------------------
------------------------------Time to First Token-------------------------------
Mean TTFT (ms):                                                     206.74
Median TTFT (ms):                                                   207.90
P99 TTFT (ms):                                                      438.29
--------------------Time per Output Token (excl. 1st token)---------------------
Mean TPOT (ms):                                                     36.52
Median TPOT (ms):                                                   36.53
P99 TPOT (ms):                                                      48.03
------------------------------Inter-token Latency-------------------------------
Mean ITL (ms):                                                      35.97
Median ITL (ms):                                                    29.03
P99 ITL (ms):                                                       214.95
--------------------------------------------------------------------------------
```


```
-------------------------------------------------------------------------------
                              HBF Write Statistics                              
--------------------------------------------------------------------------------
Instance [0] HBF writes: 0, Total bytes written: 0.00 MB
Total HBF writes (all instances):                                   0
Total HBF bytes written (all instances):                            0.00 MB
--------------------------------------------------------------------------------
                                  Instance [0]                                  
--------------------------------------------------------------------------------
------------------------------Time to First Token-------------------------------
Mean TTFT (ms):                                                     206.74
Median TTFT (ms):                                                   207.90
P99 TTFT (ms):                                                      438.29
--------------------Time per Output Token (excl. 1st token)---------------------
Mean TPOT (ms):                                                     36.52
Median TPOT (ms):                                                   36.53
P99 TPOT (ms):                                                      48.03
------------------------------Inter-token Latency-------------------------------
Mean ITL (ms):                                                      35.97
Median ITL (ms):                                                    29.03
P99 ITL (ms):                                                       214.95
--------------------------------------------------------------------------------
```
