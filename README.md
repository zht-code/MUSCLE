# MUSCLE: Multi-view and multi-scale attentional feature fusion for microRNA-disease associations prediction
MicroRNAs (miRNAs) synergize with various biomolecules in human cells resulting in diverse functions in regulating a wide range of biological processes associated with human diseases. Predicting potential disease-associated miRNAs as valuable biomarkers contributes to the diagnosis, prevention, and treatment of human diseases. However, few previous methods take a holistic perspective and only concentrate on isolated miRNA and disease objects, thereby ignoring that human cells are responsible for multiple relationships. In this paper, we first constructed a multi-view graph based on the relationships between miRNAs and various biomolecules, and then utilized graph attention neural network to learn the graph topology features of miRNAs and diseases for each view. Next, we added an attention mechanism again, and developed a multi-scale feature fusion module, aiming to determine the optimal fusion results for the multi-view topology features of miRNAs and diseases. In addition, the prior attribute knowledge of miRNAs and diseases was simultaneously added to achieve better prediction results and solve the cold start problem. Finally, the learned miRNA and disease representations were then concatenated and fed into a multi-layer perceptron (MLP) for end-to-end training and predicting potential miRNA-disease associations. To assess the efficacy of our model (called MUSCLE), we performed 5-fold and 10-fold cross-validation, which got average AUCs (The area under ROC curves) of 0.966±0.0102 and 0.973±0.0135 respectively, outperforming most current state-of-the-art models. We then examined the impact of crucial parameters on prediction performance and performed ablation experiments on the feature combination and model architecture. Furthermore, the case studies about colon cancer, lung cancer, and breast cancer also fully demonstrate the good inductive capability of MUSCLE.

![image](https://github.com/zht-code/MUSCLE/blob/main/IMG/liuchengtu.png)
The flowchart of MUSCLE. **A.** Data sources and some symbols in this study. **B.** The computation and integration for the prior attribute features of miRNAs and diseases. \textbf{C.} Multiple heterogeneous graph construction and multi-view graph attention network for graph topology feature extraction of miRNAs and diseases. \textbf{D.} Multi-scale attentional feature fusion mechanism for efficiently fuse these multiple graph topology features. \textbf{E.} Multi-layer perceptron for trainning and prediction with attribute and graph topology features of miRNAs and diseases.


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

