# MUSCLE: Multi-view and multi-scale attentional feature fusion for microRNA-disease associations prediction
To better predict miRNA-disease association, we propose a novel computational method called MUSCLE. These methods can not only improve the accuracy of miRNA-disease association but also provide more effective means for disease diagnosis, treatment, and prevention.This repository contains codes and datas for MUSCLE model.

![image](github.com/zht-code/MUSCLE/IMG/liuchengtu.png)

# Data description
all data downdload from data.zip
| File name  | Description |
| ------------- | ------------- |
| miRNA.csv    | MicroRNA name  |
| disease.csv  | Disease name   |
| all_sample.csv  | All miRNA-disease samples  |
| miRNA_disease_feature.csv | Feature of miRNAs and diseases fused with Gaussian interaction profile kernel(GIP) |
| m_drug_d_sample.csv| Names of all miRNAs, drugs and diseases | 
| m_drug_drug_d_sample.csv|  Samples of all miRNA-drug and drug-disease associations| 
| m_mRNA_d_sample.csv| Names of all miRNAs, mRNA and diseases | 
| m_mRNA_mRNA_d_sample.csv|  Sample of all miRNA-mRNA and mRNA-disease associations | 
| m_lncRNA_d_sample.csv| Names of all miRNAs, lncRNA and diseases | 
| m_lncRNA_lncRNA_d_sample.csv|  Sample of all miRNA-lncRNA and lncRNA-disease associations| 
| xm_drug_d_adj.csv |  Each miRNA-drug-disease adjacency matrix used in the 5-fold CV | 
| xm_mRNA_d_adj.csv |  Each miRNA-mRNA-disease adjacency matrix used in the 5-fold CV | 
| xm_lncRNA_d_adj.csv |  Each miRNA-lncRNA-disease adjacency matrix used in the 5-fold CV|
| n_m_drug_d_adj.csv |  Each miRNA-drug-disease adjacency matrix used in the 10-fold CV | 
| n_m_mRNA_d_adj.csv |  Each miRNA-mRNA-disease adjacency matrix used in the 10-fold CV | 
| n_m_lncRNA_d_adj.csv | Each miRNA-lncRNA-disease adjacency matrix used in the 10-fold CV |

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


1, Run train.py to generate train_model and performance score, the options are:
```
python ./train.py
```
2, Ablation experiment：Run GAT_Ave_MDA.py GAT_CAT_MDA.py GAT_dot_MDA.py GAT_MS_MDA.py GAT_MSCAM_MDA.py to generate performance score for everyone, the options are:
```
python ./Ablation/GAT_Ave_MDA.py

python ./Ablation/GAT_CAT_MDA.py

python ./Ablation/GAT_dot_MDA.py

python ./Ablation/GAT_MS_MDA.py

python ./Ablation/GAT_MSCAM_MDA.py
```
3, Run 5_Fold.py and 10_Fold.py to generate 5-CV and 10-CV scores, the options are:
```
python ./5_Fold.py

python ./10_Fold.py
```
4, embedding size: Run train.py in the embedding_size file , the options are:
```
python  ./embedding_size/902/train.py

python  ./embedding_size/878/train.py

python  ./embedding_size/700/train.py

python  ./embedding_size/600/train.py
```
5, MLP Layers: Run train.py in the MLPlayer file , the options are:
```
python  ./MLPlayer/MLP1/train.py

python  ./MLPlayer/MLP2/train.py

python  ./MLPlayer/MLP3/train.py

python  ./MLPlayer/MLP4/train.py
```
6, case_study: Run casestudies.py to generate three diseases prediction, the options are:
```
python  ./casestudies.py
```
# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

