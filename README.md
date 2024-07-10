# ContrastPool
This is the official PyTorch implementation of ContrastPool from the paper 
*"Contrastive Graph Pooling for Explainable Classification of Brain Networks"* published in IEEE Transactions on Medical Imaging (TMI) 2024.

Link: [Arxiv](https://arxiv.org/abs/2307.11133).

<img alt="Model" src="figs/framework.png" title="Framework"/>


## Data
All Preprocessed data used in this paper are published in [this paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/44e3a3115ca26e5127851acd0cedd0d9-Paper-Datasets_and_Benchmarks.pdf). 
Data splits and configurations are stored in `./data/` and `./configs/`. If you want to process your own data, please check the dataloader script `./data/BrainNet.py`.

## Usage

Please check `baseline.sh` on how to run the project.

## Citation

If you find this code useful, please consider citing our paper:

```
@ARTICLE{10508252,
  author={Xu, Jiaxing and Bian, Qingtian and Li, Xinhang and Zhang, Aihu and Ke, Yiping and Qiao, Miao and Zhang, Wei and Sim, Wei Khang Jeremy and Gulyás, Balázs},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Contrastive Graph Pooling for Explainable Classification of Brain Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Functional magnetic resonance imaging;Feature extraction;Task analysis;Data mining;Alzheimer's disease;Message passing;Brain modeling;Brain Network;Deep Learning for Neuroimaging;fMRI Biomarker;Graph Classification;Graph Neural Network},
  doi={10.1109/TMI.2024.3392988}}
```

## Contact

If you have any questions, please feel free to reach out at `jiaxing003@e.ntu.edu.sg`.
