import scipy.sparse as sp
import torch
import torch.utils.data as Data
import numpy as np
import torch_geometric.utils as utils
from torch import nn
import csv
from sklearn.metrics import roc_curve, roc_auc_score,precision_score,f1_score,recall_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
import argparse
import os
import torch.nn.functional as F
import math

import pandas as pd
from sklearn.metrics import auc
from torch_geometric.nn import GATConv
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）
# 如果GPU可用，利用GPU进行训练
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument("--hid_feats", type=int, default=1500, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=901, help='Output layer dimensionalities.')
parser.add_argument("--method", default='sum', help='Merge feature method')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--input_dropout", type=float, default=0, help='Dropout applied at input layer.')
parser.add_argument("--layer_dropout", type=float, default=0, help='Dropout applied at hidden layers.')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--early_stopping', type=int, default=200, help='stop')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--mlp', type=list, default=[64, 2], help='mlp layers')
parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
parser.add_argument('--save_score', default='True', help='save_score')


args = parser.parse_args()  # 会将命令行参数转换为Python对象，可以通过属性来访问这些参数的值。
args.dd2 = True
args.data_dir = 'data/data/'
args.result_dir = 'result/'         #保存结果
args.save_score = True if str(args.save_score) == 'True' else False
args.m_drug_d_num = 2060
args.m_mRNA_d_num = 3929
args.m_incRNA_d_num = 2459



class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=2, r=2):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class MultiGraphConvolution_Layer(nn.Module):

    def __init__(self, in_features, out_features):
        super(MultiGraphConvolution_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # self.view_conv1 = GATConv(in_features, out_features, heads=3, dropout=0.05, concat=False)
        # self.view_conv2 = GATConv(out_features, out_features, heads=3, dropout=0.05, concat=False)
        self.view_conv1 = GATConv(in_features, out_features)
        # self.view_conv2 = GATConv(out_features, out_features)

    def forward(self, input_x, adj, device):
        sum_x = torch.zeros((0, input_x.shape[0], self.out_features)).to(device)
        adj_temp = adj.numpy()

        adj_temp = sp.coo_matrix(adj_temp)
        # 转换为 PyTorch Geometric 中的数据对象
        edge_index, edge_weight = utils.from_scipy_sparse_matrix(adj_temp)
        input_x = input_x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        input_x_view_conv1 = F.relu(self.view_conv1(input_x, edge_index, edge_weight))
        input_x_view_conv2 = torch.unsqueeze(input_x_view_conv1, dim=0)
        sum_x = torch.cat((sum_x, input_x_view_conv2), 0)
        return sum_x


class MGC_Model(nn.Module):

    def __init__(self, feature_num, hidden_num, out_num):
        super(MGC_Model, self).__init__()
        self.feature_num = feature_num
        self.hidden_num = hidden_num  # hidden_num = 1 or feature_num
        self.mgc = MultiGraphConvolution_Layer(in_features=feature_num, out_features=out_num)

    def forward(self, input_x, adj, device):
        x = self.mgc(input_x, adj, device)
        x = F.relu(x)

        # x_ce = self.linear_ce(x)
        # x_b = self.linear_b(input_x)
        # x = x_b + x_ce

        # x = torch.sigmoid(x)
        return x


class MDA(nn.Module):
    def __init__(self, args):
        super(MDA, self).__init__()
        self.args = args
        self.gat_m_drug_m = MGC_Model(args.m_drug_d_num, args.hid_feats, args.out_feats)
        self.gat_m_mRNA_d = MGC_Model(args.m_mRNA_d_num, args.hid_feats, args.out_feats)
        self.gat_m_incRNA_d = MGC_Model(args.m_incRNA_d_num, args.hid_feats, args.out_feats)
        self.mlp = nn.Sequential(
                nn.Linear(901*2, 1024),
                nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.Dropout(0.1),
        nn.Linear(512, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 1),nn.Sigmoid())
        # self.mlp = nn.Sequential(nn.Linear(1802 * 2, 1024),
        #                          nn.Linear(1024, 512),
        #                          nn.Linear(512, 64),
        #                          nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, data, train_sample, test_sample, device):
        # 随机生成drug、mRNA、incRNA特征
        m_drug_d_feature = torch.randn((self.args.m_drug_d_num, self.args.m_drug_d_num))

        # 构图
        m_drug_d_ass = self.gat_m_drug_m(m_drug_d_feature, data['m_drug_d_adj'], device)  # 1x2060x901

        # 将drug、mRNA、incRNA中间层去掉
        m_drug_d_ass = torch.cat((m_drug_d_ass[0][:901], m_drug_d_ass[0][1183:]), dim=0).unsqueeze(0)  # 1x1778x901
        # 将四张图拼在一起
        sum_x = torch.cat((m_drug_d_ass, data['miRNA_disease_feature'].unsqueeze(0)), 0)
        sum_x = sum_x.unsqueeze(0)  # 1x2x1778x901
        sum_x = sum_x.permute(2, 1, 0, 3)

        fusion_mode = MS_CAM().to(device)
        sum_x = fusion_mode(sum_x)  # 1778 2 1 901
        sum_x = sum_x.permute(2, 1, 0, 3)  # 1x2x1778x901
        sum_x = torch.squeeze(sum_x, dim=0)  # 2x1778x901

        # 将图拼在一起
        sum_x = (sum_x[0] + sum_x[1]) / 2  # 1778x901

        train_sample = train_sample.int()
        train_emb = torch.empty(0).to(device)
        for i in range(len(train_sample)):
            a = torch.cat((sum_x[train_sample[i][0]], sum_x[train_sample[i][1]]), dim=0).unsqueeze(0)
            train_emb = torch.cat((train_emb, a), dim=0)
        train_score = self.mlp(train_emb)

        test_sample = test_sample.int()
        test_emb = torch.empty(0).to(device)
        for i in range(len(test_sample)):
            a = torch.cat((sum_x[test_sample[i][0]], sum_x[test_sample[i][1]]), dim=0).unsqueeze(0)
            test_emb = torch.cat((test_emb, a), dim=0)
        test_score = self.mlp(test_emb)

        return train_score, test_score
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
# 加载数据集
def loading():
    data = dict()
    # 读取融合了GIP的miRNA和disease的特征
    data['miRNA_disease_feature'] = pd.read_csv(args.data_dir + 'miRNA_disease_feature.csv', header=None).iloc[:,
                                    :].values
    data['m_drug_d_adj'] = pd.read_csv(args.data_dir + "1m_drug_d_adj.csv", header=None).iloc[:, :].values
    data['m_mRNA_d_adj'] = pd.read_csv(args.data_dir + "1m_mRNA_d_adj.csv", header=None).iloc[:, :].values
    data['m_incRNA_d_adj'] = pd.read_csv(args.data_dir + "1m_incRNA_d_adj.csv", header=None).iloc[:, :].values
    return data




dataset = loading()
test_feature = pd.read_csv('./data/data/test_sample_index_5.csv', header=None).iloc[:,:].values
test_label = pd.read_csv('./data/data/test_label_5.csv', header=None).iloc[:,:].values
val_feature = torch.FloatTensor(test_feature)
val_label = torch.FloatTensor(test_label)
# 将字典中的数组转换为张量
dataset['m_drug_d_adj'] = torch.FloatTensor(dataset['m_drug_d_adj'])
dataset['m_mRNA_d_adj'] = torch.FloatTensor(dataset['m_mRNA_d_adj'])
dataset['m_incRNA_d_adj'] = torch.FloatTensor(dataset['m_incRNA_d_adj'])
dataset['miRNA_disease_feature'] = torch.FloatTensor(dataset['miRNA_disease_feature']).to(device)
# 模型实例化
model = MDA(args).to(device)
# map_location:指定设备，cpu或者GPU
model.load_state_dict(torch.load('save_model/att_2/train_model.pth', map_location="cpu"))
val_data = Data.TensorDataset(val_feature, val_label)
val_loader = Data.DataLoader(dataset=val_data, batch_size=500, shuffle=True, num_workers=0)
loss_fn = nn.MSELoss()
def compute_accuracy_and_loss(model, data_loader, device):
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        test_pro = model(dataset, features, device)
        test_pro_int =  np.rint(test_pro.detach().cpu().numpy()).astype(np.int64)
        test_score = test_pro.squeeze(1)
        targets = targets.squeeze(1)
        # 处理输出数据
        currnet_loss = loss_fn(test_score, targets)
        CM = confusion_matrix(targets.cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64))
        FPR, TPR, _ = roc_curve(targets.cpu().numpy(), test_score.detach().cpu().numpy(), pos_label=1)
        AUC = roc_auc_score(targets.cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64))
        pre = precision_score(targets.cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
        recall = recall_score(targets.cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
        precision_1, recall_1, threshold = precision_recall_curve(targets.detach().cpu().numpy(), test_score.detach().cpu().numpy())
        AUPR = auc(recall_1,precision_1)
        f1 = f1_score(targets.detach().cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
        CM = CM.tolist()
        TN = CM[0][0]
        FP = CM[0][1]
        FN = CM[1][0]
        TP = CM[1][1]
        Acc = (TN + TP) / (TN + TP + FN + FP)
        Sen = TP / (TP + FN)
        # Spec = TN / (TN + FP)
        # MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return AUC, currnet_loss.item(), pre, recall, f1, FPR, TPR, Acc, Sen,  AUPR, test_pro, test_pro_int
# 计算测试精度
val_auc, val_loss, val_pre, val_recall, val_f1, val_FPR, val_TPR, val_Acc, val_Sen, val_AUPR, test_pro, test_pro_int = compute_accuracy_and_loss(model, val_loader, device=device)
print(f'Test Loss.: {val_loss:.2f}')
print(f'Test Auc.: {val_auc:.4f}')
print(f'Test AUPR.: {val_AUPR:.4f}')
print(f'Test pre.: {val_pre:.2f}')
print(f'Test recall.: {val_recall:.2f}')
print(f'Test F1.: {val_f1:.2f}')
print(f'Test acc.: {val_Acc:.2f}')
print(f'Test sen.: {val_Sen:.2f}')
dataset['att_2_pro'] = np.concatenate((test_label,test_pro),axis=1).tolist()
dataset['att_2_int'] = np.concatenate((test_label,test_pro_int),axis=1).tolist()
StorFile(dataset['att_2_pro'], 'Graph_Attention_MDA/Ablation2/att_2/att_2_pro.csv')
StorFile(dataset['att_2_int'], 'Graph_Attention_MDA/Ablation2/att_2/att_2_int.csv')
# print(f'Validation spec.: {val_Spec:.2f}')
# print(f'Validation mcc.: {val_MCC:.2f}')
# np.savetxt('./data/linear2_10FCV_FPR.txt',val_FPR)
# np.savetxt('./data/linear2_10FCV_TPR.txt',val_TPR)