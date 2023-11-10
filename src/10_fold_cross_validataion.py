import random
import math
from model import MDA
import pandas as pd
import numpy as np
from torch import optim,nn
import torch as t
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
import argparse
import os
import warnings
import csv
warnings.filterwarnings("ignore")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  

device = t.device('cuda:0' if t.cuda.is_available() else "cpu")
t.backends.cudnn.enabled = False

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


args = parser.parse_args()  
args.dd2 = True
args.data_dir = 'data/data/'
args.result_dir = 'result/'         
args.save_score = True if str(args.save_score) == 'True' else False


def loading():
    data = dict()
 
    data['all_sample'] = pd.read_csv(args.data_dir + 'all_sample.csv', header=None).iloc[:,:].values
  
    data['miRNA'] = pd.read_csv(args.data_dir + 'miRNA.csv', header=None).iloc[:, :].values
    data['disease'] = pd.read_csv(args.data_dir + 'disease.csv', header=None).iloc[:, :].values
    data['miRNA_disease'] = np.concatenate((data['miRNA'], data['disease']), axis=0)
   
    data['miRNA_disease_feature'] = pd.read_csv(args.data_dir + 'miRNA_disease_feature.csv', header=None).iloc[:,:].values
   
    data['m_drug_d_sample'] = pd.read_csv(args.data_dir + 'm_drug_d_sample.csv', header=None).iloc[:,:].values
    data['m_drug_drug_d_sample'] = pd.read_csv(args.data_dir + 'm_drug_drug_d_sample.csv', header=None).iloc[:, :].values

    data['m_mRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_mRNA_d_sample.csv', header=None).iloc[:, :].values
    data['m_mRNA_mRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_mRNA_mRNA_d_sample.csv', header=None).iloc[:, :].values

    data['m_incRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_incRNA_d_sample.csv', header=None).iloc[:, :].values
    data['m_incRNA_incRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_incRNA_incRNA_d_sample.csv', header=None).iloc[:,:].values
    return data

def make_index(data,sample):
    sample_index = []
    for i in range(sample.shape[0]):
        idx = np.where(sample[i][0] == data['miRNA_disease'])
        idy = np.where(sample[i][1] == data['miRNA_disease'])
        sample_index.append([idx[0].item(), idy[0].item()])
    sample_index = np.array(sample_index)
    return sample_index
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

if __name__ == '__main__':

    dataset = loading()
    args.m_drug_d_num = dataset['m_drug_d_sample'].shape[0]
    args.m_mRNA_d_num = dataset['m_mRNA_d_sample'].shape[0]
    args.m_incRNA_d_num = dataset['m_incRNA_d_sample'].shape[0]
    dataset['miRNA_disease_feature'] = t.FloatTensor(dataset['miRNA_disease_feature']).to(device)

    model = MDA(args).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    cross_entropy = nn.MSELoss(reduction='mean')
    file_num = 1


    auc = 0
    auprc = 0
    acc = 0
    f1 = 0
    recall = 0
    pre = 0


    max_test_auc = 0
    kfold = KFold(n_splits=10, shuffle=True, random_state=123)
    for train_index, test_index in kfold.split(dataset['all_sample'][:,:2]):
        tran_sample = dataset['all_sample'][train_index][:,:2]
        tran_sample_index = make_index(dataset, tran_sample)
        tran_label = dataset['all_sample'][train_index][:,2]
        test_sample = dataset['all_sample'][test_index][:,:2]
        test_sample_index = make_index(dataset, test_sample)
        test_label = dataset['all_sample'][test_index][:, 2]
        # StorFile(test_sample_index.tolist(), 'data/data/test_sample_index_10.csv')
        # StorFile([str(x) for x in test_label], 'data/data/test_label_10.csv')
        dataset['m_drug_drug_d_sample'] = np.concatenate((dataset['m_drug_drug_d_sample'], tran_sample), axis=0)
        dataset['m_mRNA_mRNA_d_sample'] = np.concatenate((dataset['m_mRNA_mRNA_d_sample'], tran_sample), axis=0)
        dataset['m_incRNA_incRNA_d_sample'] = np.concatenate((dataset['m_incRNA_incRNA_d_sample'], tran_sample), axis=0)
  
        file_name1 = str(file_num) + "_m_drug_d_adj.csv"
        dataset['m_drug_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:,:].values
 
        file_name1 = str(file_num) + "_m_mRNA_d_adj.csv"
        dataset['m_mRNA_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:, :].values

        file_name1 = str(file_num) + "_m_incRNA_d_adj.csv"
        dataset['m_incRNA_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:, :].values

        dataset['m_drug_d_adj'] = t.FloatTensor(dataset['m_drug_d_adj'])
        dataset['m_mRNA_d_adj'] = t.FloatTensor(dataset['m_mRNA_d_adj'])
        dataset['m_incRNA_d_adj'] = t.FloatTensor(dataset['m_incRNA_d_adj'])
        tran_sample_index = t.FloatTensor(tran_sample_index).to(device)
        tran_label = t.FloatTensor(tran_label.astype(int)).to(device)
        test_sample_index = t.FloatTensor(test_sample_index).to(device)
        test_label = t.FloatTensor(test_label.astype(int)).to(device)
        for i in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            train_score, test_score = model(dataset, tran_sample_index, test_sample_index, device)
            train_score = train_score.squeeze(1)
            train_label = tran_label.to(device)
            train_loss = cross_entropy(train_score, train_label)
            train_loss.backward()
            train_auc = roc_auc_score(train_label.detach().cpu().numpy(),
                                      train_score.detach().cpu().numpy())
            train_acc = accuracy_score(train_label.detach().cpu().numpy().astype(np.int64),
                                       np.rint(train_score.detach().cpu().numpy()).astype(np.int64))
            optimizer.step()
            model.eval()
            test_score = test_score.squeeze(1)
            test_label = test_label.to(device)
            test_loss = cross_entropy(test_score, test_label)
            test_auc = roc_auc_score(test_label.detach().cpu().numpy(),
                                     test_score.detach().cpu().numpy())
            test_acc = accuracy_score(test_label.detach().cpu().numpy().astype(np.int64),
                                      np.rint(test_score.detach().cpu().numpy()).astype(np.int64))
            test_aupr = average_precision_score(test_label.detach().cpu().numpy(), test_score.detach().cpu().numpy())
            test_f1 = f1_score(test_label.detach().cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
            test_recall = recall_score(test_label.detach().cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
            test_pre = precision_score(test_label.detach().cpu().numpy(), np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
            print(f'Epoch: {i + 1:03d}/{args.epochs:03d}' f'   | Learning Rate {scheduler.get_last_lr()[0]:.6f}')
            # print(f'Epoch: {i + 1:03d}/{args.epochs:03d}')
            print(f'Train Auc.: {train_auc:.4f}' f' | Test Auc.: {test_auc:.4f}')
            print(f'Train Loss.: {train_loss.item():.4f}' f' | Test Loss.: {test_loss.item():.4f}')
            print(f'Train Acc.: {train_acc:.4f}' f' | Test Acc.: {test_acc:.4f}')

   
            if test_auc > max_test_auc:
                t.save(model.state_dict(), "./save_model/10_fold/10fold_train_model.pth")
                max_test_auc = test_auc
                auc = test_auc
                auprc = test_aupr
                acc = test_acc
                f1 = test_f1
                recall = test_recall
                pre = test_pre
    
            scheduler.step()
        file_num += 1
    print(f' | Test Auc.: {auc:.4f}')
    print(f' | Test Auprc.: {auprc:.4f}')
    print(f' | Test Acc.: {acc:.4f}')
    print(f' | Test F1.: {f1:.4f}')
    print(f' | Test Recall.: {recall:.4f}')
    print(f' | Test Precision.: {pre:.4f}')