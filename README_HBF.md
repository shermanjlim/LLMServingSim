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
    --window 0:60 \
    --load-scale 1.0 \
    --output 'output/example_single_run.csv' \
    --log-interval 1.0 \
    --enable-hbf-offloading \
    --enable-hbf-kv
```

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