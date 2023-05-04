from torch import nn
import torch as torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import GATConv
import scipy.sparse as sp
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv


# 多头注意力机制
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, emb_size, heads):
        super(SelfAttention, self).__init__()

        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        # 定义线性层
        self.Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, emb_size)

    def forward(self, x):
        batch_size, seq_len, emb_size = x.size()
        assert emb_size == self.emb_size

        # 将输入向量拆分为多个头
        multi_head_x = x.reshape(batch_size, seq_len, self.heads, self.head_dim)
        multi_head_x = multi_head_x.permute(0, 2, 1, 3)

        # 计算Q, K, V
        Q = self.Q(multi_head_x)
        K = self.K(multi_head_x)
        V = self.V(multi_head_x)

        # 计算注意力得分
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        # 计算输出向量
        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_len, self.heads * self.head_dim)

        # 压缩多个头到一个向量
        out = self.fc_out(out)

        return out


# 层级注意力机制
class Attention(nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        hidden_size = in_size

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3604, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return beta * z, beta


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
        return x


class MDA(nn.Module):
    def __init__(self, args):
        super(MDA, self).__init__()
        self.args = args
        self.gat_m_drug_m = MGC_Model(args.m_drug_d_num, args.hid_feats, args.out_feats)
        self.gat_m_mRNA_d = MGC_Model(args.m_mRNA_d_num, args.hid_feats, args.out_feats)
        self.gat_m_incRNA_d = MGC_Model(args.m_incRNA_d_num, args.hid_feats, args.out_feats)
        self.layer_att = Attention(3604)
        self.Self_Attention = SelfAttention(3604, 4)
        self.MultiHead_Attention = MultiHeadAttention()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(3604 * 2),
            nn.Linear(3604 * 2, 3604),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(3604),
            nn.Linear(3604, 1),
            nn.Sigmoid(),
        )
        # self.mlp1 = nn.Linear(7208, 1024)
        # self.mlp2 = nn.Linear(1024, 512)
        # self.mlp3 = nn.Linear(512, 64)
        # self.mlp4 = nn.Linear(64, 1)

    def forward(self, data, train_sample, test_sample, device):
        # 随机生成drug、mRNA、incRNA特征
        m_drug_d_feature = torch.randn((self.args.m_drug_d_num, self.args.m_drug_d_num))
        m_mRNA_d_feature = torch.randn((self.args.m_mRNA_d_num, self.args.m_mRNA_d_num))
        m_incRNA_d_feature = torch.randn((self.args.m_incRNA_d_num, self.args.m_incRNA_d_num))
        # 构图
        m_drug_d_ass = self.gat_m_drug_m(m_drug_d_feature, data['m_drug_d_adj'], device)  # 1x2060x901
        m_incRNA_d_ass = self.gat_m_incRNA_d(m_incRNA_d_feature, data['m_incRNA_d_adj'], device)  # 1x2459x901
        m_mRNA_d_ass = self.gat_m_mRNA_d(m_mRNA_d_feature, data['m_mRNA_d_adj'], device)  # 1x3929x901
        # 将drug、mRNA、incRNA中间层去掉
        m_drug_d_ass = torch.cat((m_drug_d_ass[0][:901], m_drug_d_ass[0][1183:]), dim=0).unsqueeze(0)  # 1x1778x901
        m_incRNA_d_ass = torch.cat((m_incRNA_d_ass[0][:901], m_incRNA_d_ass[0][1582:]), dim=0).unsqueeze(
            0)  # 1x1778x901
        m_mRNA_d_ass = torch.cat((m_mRNA_d_ass[0][:901], m_mRNA_d_ass[0][3052:]), dim=0).unsqueeze(0)  # 1x1778x901
        # 将四张图拼在一起
        result1 = torch.cat((m_drug_d_ass, m_incRNA_d_ass), 2)
        result2 = torch.cat((result1, m_mRNA_d_ass), 2)
        sum_x = torch.cat((result2, data['miRNA_disease_feature'].unsqueeze(0)), 2)  # 1x1778x3604
        # sum_x = torch.squeeze(sum_x, dim=0)                         #1778x3604
        x = self.Self_Attention(sum_x)
        sum_x = torch.squeeze(x, dim=0)  # 1778x3604
        # x, _ = self.layer_att(sum_x)
        # x = torch.unsqueeze(x, dim=1)

        train_sample = train_sample.int()
        train_emb = torch.empty(0).to(device)
        for i in range(len(train_sample)):
            a = torch.cat((sum_x[train_sample[i][0]], sum_x[train_sample[i][1]]), dim=0).unsqueeze(0)
            train_emb = torch.cat((train_emb, a), dim=0)
        # train_score = F.sigmoid(self.mlp4(self.mlp3(self.mlp2(self.mlp1(train_emb)))))
        train_score = self.mlp(train_emb)

        test_sample = test_sample.int()
        test_emb = torch.empty(0).to(device)
        for i in range(len(test_sample)):
            a = torch.cat((sum_x[test_sample[i][0]], sum_x[test_sample[i][1]]), dim=0).unsqueeze(0)
            test_emb = torch.cat((test_emb, a), dim=0)
        # test_score = F.sigmoid(self.mlp4(self.mlp3(self.mlp2(self.mlp1(test_emb)))))
        test_score = self.mlp(test_emb)
        return train_score, test_score