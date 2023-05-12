# MUSCLE: Multi-view and multi-scale attentional feature fusion for microRNA-disease associations prediction
To better predict miRNA-disease association, we propose a novel computational method called MUSCLE. These methods can not only improve the accuracy of miRNA-disease association but also provide more effective means for disease diagnosis, treatment, and prevention.This repository contains codes and datas for MUSCLE model.

![image](github.com/zht-code/MUSCLE/blob/main/MVGMDA.png)

# Data description

| File name  | Description |
| ------------- | ------------- |
| miRNA.csv    | microRNA name file  |
| disease.csv  | disease name file   |
| all_sample.csv  | all miRNA-disease sample  |
| miRNA_disease_feature.csv | feature of miRNAs and diseaseases fused with GIP |
| m_drug_d_sample.csv| all miRNA、drug and disease name | 
| m_drug_drug_d_sample.csv|  all miRNA-drug and drug-disease association sample| 
| m_mRNA_d_sample.csv| all miRNA、mRNA and disease name | 
| m_mRNA_mRNA_d_sample.csv|  all miRNA-mRNA and mRNA-disease association sample| 
| m_lncRNA_d_sample.csv| all miRNA、lncRNA and disease name | 
| m_lncRNA_lncRNA_d_sample.csv|  all miRNA-lncRNA and lncRNA-disease association sample| 
| xm_drug_d_adj.csv |  5-fold miRNA-drug-disease Adjacency matrix | 
| xm_mRNA_d_adj.csv |  5-fold miRNA-mRNA-disease Adjacency matrix | 
| xm_lncRNA_d_adj.csv |  5-fold miRNA-lncRNA-disease Adjacency matrix |
| x_m_drug_d_adj.csv |  10-fold miRNA-drug-disease Adjacency matrix | 
| x_m_mRNA_d_adj.csv |  10-fold miRNA-mRNA-disease Adjacency matrix | 
| x_m_lncRNA_d_adj.csv |  10-fold miRNA-lncRNA-disease Adjacency matrix |

# Requirements
MUSCLE is tested to work under:

python == 3.6

pytorch == 1.10.2+cu113

scipy == 1.5.4

numpy == 1.19.5

sklearn == 0.24.2

pandas == 1.1.5

matplotlib == 3.3.4

networkx == 2.5.1
# Quick start
To reproduce our results:

1, Download the environment required by MUSCLE
```
pip install pytorch == 1.10.2+cu113
```
2, Run train.py to generate train_model and performance score, the options are:
```
python ./train.py
```
3, Ablation experiment：Run GAT_Ave_MDA.py GAT_CAT_MDA.py GAT_dot_MDA.py GAT_MS_MDA.py GAT_MSCAM_MDA.py to generate performance score for everyone, the options are:
```
python ./Ablation/GAT_Ave_MDA.py

python ./Ablation/GAT_CAT_MDA.py

python ./Ablation/GAT_dot_MDA.py

python ./Ablation/GAT_MS_MDA.py

python ./Ablation/GAT_MSCAM_MDA.py
```
4, Run 5_Fold.py and 10_Fold.py to generate 5-CV and 10-CV scores, the options are:
```
python ./5_Fold.py

python ./10_Fold.py
```
5, embedding size: Run train.py in the embedding_size file , the options are:
```
python  ./embedding_size/902/train.py

python  ./embedding_size/878/train.py

python  ./embedding_size/700/train.py

python  ./embedding_size/600/train.py
```
6, MLP Layers: Run train.py in the MLPlayer file , the options are:
```
python  ./MLPlayer/MLP1/train.py

python  ./MLPlayer/MLP2/train.py

python  ./MLPlayer/MLP3/train.py

python  ./MLPlayer/MLP4/train.py
```
7, case_study: Run casestudies.py to generate three diseases prediction, the options are:
```
python  ./casestudies.py
```
# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

