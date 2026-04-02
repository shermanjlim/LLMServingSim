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
    --dataset azure_chat_23:azure_chat_23 \
    --window 0:60 \
    --load-scale 1.0 \
    --output 'output/example_single_run.csv' \
    --log-interval 1.0
```