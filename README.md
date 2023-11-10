# MUSCLE: Multi-view and multi-scale attentional feature fusion for microRNA-disease associations prediction
MicroRNAs (miRNAs) synergize with various biomolecules in human cells resulting in diverse functions in regulating a wide range of biological processes associated with human diseases. Predicting potential disease-associated miRNAs as valuable biomarkers contributes to the diagnosis, prevention, and treatment of human diseases. However, few previous methods take a holistic perspective and only concentrate on isolated miRNA and disease objects, thereby ignoring that human cells are responsible for multiple relationships. In this paper, we first constructed a multi-view graph based on the relationships between miRNAs and various biomolecules, and then utilized graph attention neural network to learn the graph topology features of miRNAs and diseases for each view. Next, we added an attention mechanism again, and developed a multi-scale feature fusion module, aiming to determine the optimal fusion results for the multi-view topology features of miRNAs and diseases. In addition, the prior attribute knowledge of miRNAs and diseases was simultaneously added to achieve better prediction results and solve the cold start problem. Finally, the learned miRNA and disease representations were then concatenated and fed into a multi-layer perceptron (MLP) for end-to-end training and predicting potential miRNA-disease associations. To assess the efficacy of our model (called MUSCLE), we performed 5-fold and 10-fold cross-validation, which got average AUCs (The area under ROC curves) of 0.966±0.0102 and 0.973±0.0135 respectively, outperforming most current state-of-the-art models. We then examined the impact of crucial parameters on prediction performance and performed ablation experiments on the feature combination and model architecture. Furthermore, the case studies about colon cancer, lung cancer, and breast cancer also fully demonstrate the good inductive capability of MUSCLE.

![image](https://github.com/zht-code/MUSCLE/blob/main/IMG/liuchengtu.png)
The flowchart of MUSCLE. **A.** Data sources and some symbols in this study. **B.** The computation and integration for the prior attribute features of miRNAs and diseases. **C.** Multiple heterogeneous graph construction and multi-view graph attention network for graph topology feature extraction of miRNAs and diseases. **D.** Multi-scale attentional feature fusion mechanism for efficiently fuse these multiple graph topology features. **E.** Multi-layer perceptron for trainning and prediction with attribute and graph topology features of miRNAs and diseases.

## Table of Contents

- [Data description](#Data-description)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)


# Data description
All users can get our data from data.zip under the data folder.
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


1, Run train.py to train the MUSCLE model, the options are:
```
python ./src/train.py
```
2, Run 5_fold_cross_validataion.py and 10_fold_cross_validataion.py to reproduce the predicted results of MUSCLE under 5-fold and 10 fold cross validataion, the options are:
```
python ./src/5_fold_cross_validataion.py

python ./src/10_fold_cross_validataion.py
```


3, To achieve optimal results of classification, we performed a parameter analysis of the MUSCLE method, focusing on two crucial parameters: the embedding dimensions generated by the graph attention network and the number of layers of the multi-layer perceptron (MLP). To ensure fairness, we changed only one parameter at a time and kept the other parameters unchanged. Furthermore, for enhanced experiment reliability and accuracy, we conducted 5-fold cross-validation for each parameter. Run src/parameter_analysis to reproduce the parameter analysis results, the options are:
```
python  ./embedding_size/902/train.py

python  ./embedding_size/878/train.py

python  ./embedding_size/700/train.py

python  ./embedding_size/600/train.py

python  ./MLPlayer/MLP2/train.py

python  ./MLPlayer/MLP3/train.py

python  ./MLPlayer/MLP4/train.py

python  ./MLPlayer/MLP5/train.py
```

4, In this study, we integrated the biological attribute and three topological features to represent miRNA and disease nodes. In this section, we conducted ablation experiments to examine the effects of features. Furthermore, we also examined the validity of the multi-scale attentional feature fusion module through the ablation experiments. Similar to the previous experiments, we adopted a control variable method and used the average result under 5-fold cross-validation as the final evaluation metric. Run src/ablation to reproduce the ablation results, the options are:

```
python ./Ablation/GAT_Ave_MDA.py

python ./Ablation/GAT_CAT_MDA.py

python ./Ablation/GAT_dot_MDA.py

python ./Ablation/GAT_MS_MDA.py

python ./Ablation/GAT_MSCAM_MDA.py
```


5, To further examine the abili of MUSCLE in practical applications, we selected three common diseases for case studies, including lung cancer, breast cancer, and colon cancer. First, all the known miRNA-disease associations in our dataset were used to train the MUSCLE model. Second, we constructed all except known associations above between miRNAs and corresponding diseases as the test dataset. After that, MUSCLE predicted the three test dataset for the corresponding diseases and selected the top 50 miRNAs with the highest predicted scores. Finally, we checked the accuracy of the projected miRNAs using the dbDEMC and miRCancer databases. Run src/case_studies to reproduce the results, the options are:
```
python  ./src/case_studies.py
```
The validation results of all case studies are in the folder: ./case_study

# Contributing

XXL and SLP conceived the experiments,  BYJ, HTZ and LWX conducted the experiments, BYJ, XXL and SLP analysed the results.  BYJ and HTZ wrote and reviewed the manuscript.

# Cite
<p align="center">
  <a href="https://clustrmaps.com/site/1bpq2">
     <img width="200"  src="https://clustrmaps.com/map_v2.png?cl=ffffff&w=268&t=m&d=4hIDPHzBcvyZcFn8iDMpEM-PyYTzzqGtngzRP7_HkNs" />
   </a>
</p>

<p align="center">
  <a href="#">
     <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fjiboyalab%2FscDecipher&labelColor=%233499cc&countColor=%2370c168" />
   </a>
</p>


# Contacts
If you have any questions or comments, please feel free to email: byj@hnu.edu.cn.

# License

[MIT ? Richard McRichface.](../LICENSE)

