#!/bin/sh

model="configs/abide_schaefer100/TUs_graph_classification_ContrastPool_abide_schaefer100_100k.json"
echo ${model}
python main.py --config  $model --gpu_id 0 --node_feat_transform pearson --max_time 60 --init_lr 1e-2 --threshold 0.0 --batch_size 20 --dropout 0.0 --contrast --pool_ratio 0.5 --lambda1 1e-3 --L 2
