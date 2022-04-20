from torch.utils.data import DataLoader
from dgl.nn.pytorch import GraphConv
from dgl.data import MiniGCDataset
from dgl.data import DGLDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import urllib.request
import pandas as pd
import numpy as np
import torch
import dgl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain
import random
import networkx as nx

result_MAE = []
result_RMSE = []
for i in range(10):
    print(i)


    class SyntheticDataset(DGLDataset):
        def __init__(self):
            super().__init__(name='synthetic')

        def process(self):
            self.l = []
            self.graphs = []
            self.labels = []
            edge = pd.read_csv('E:\\学习\\科研论文\\DLP\\Dataset\\Data\\ULD.csv')
            edges = edge.sample(n=None, frac=1, replace=False, weights=None, random_state=1, axis=None)  # # 稀疏性

            src = edges['src'].to_numpy()
            dst = edges['dst'].to_numpy() + max(src)
            u = np.concatenate([src, dst])
            v = np.concatenate([dst, src])
            g = dgl.graph((u, v))
            # ###############
            # #数据集的划分
            # #取正例
            index1 = []
            data1 = np.array(edges[['src', 'dst']])
            for h in data1:
                index1.append([h[0], h[1] + max(src)])
            # print(index1)
            # #########
            # #取全部负例
            index0_ = []
            for h in src:
                for k in dst:
                    if [h, k] not in index1:
                        index0_.append([h, k])
            # print(index0_)
            # #取和正例等量的负例
            sample_num = len(index1)  # 假设取100%的数据
            sample_list = [i for i in range(len(index0_))]
            sample_list = random.sample(sample_list, sample_num)
            index0 = []
            for i in sample_list:
                index0.append(index0_[i])
            # print(index0)
            # ###########
            # #######连同的节点######
            # ###9.15####
            graph = g.to_networkx()
            hop1 = []
            hop2 = []
            for n in index0:
                path_u = []
                path3 = "获取目标节点之间的所有路径"  # all simple paths
                for j in path3:
                    if len(j) == 4:
                        path_u.append(j)
                if len(path_u) > 0:
                    for h in path_u:
                        hop1.append(h[1])
                        hop2.append(h[2])
                nodes = "重复节点去重"
                if nodes:
                    sub = "根据节点获取局部结构"
                    sub.ndata['x'] = torch.zeros([len(nodes), 6])
                    pid = sub.ndata[dgl.NID]
                    for i in range(pid.shape[0]):
                        # # #四位特征
                        # if pid[i] == n[0]:
                        #     sub.ndata['x'][i, 0] = 1
                        # elif pid[i] == n[1]:
                        #     sub.ndata['x'][i, 1] = 1
                        # elif pid[i] in hop1:
                        #     sub.ndata['x'][i, 2] = 1
                        # elif pid[i] in hop2:
                        #     sub.ndata['x'][i, 3] = 1
                        # #六位特征
                        if pid[i] in src:
                            sub.ndata['x'][i, 0] = 1
                        if pid[i] in dst:
                            sub.ndata['x'][i, 1] = 1
                        if pid[i] == n[0]:
                            sub.ndata['x'][i, 2] = 1
                        elif pid[i] == n[1]:
                            sub.ndata['x'][i, 5] = 1
                        elif pid[i] in hop1:
                            sub.ndata['x'][i, 3] = 1
                        elif pid[i] in hop2:
                            sub.ndata['x'][i, 4] = 1
                        # #两位特征
                        # if pid[i] in dst:
                        #     sub.ndata['x'][i, 1] = 1
                        # elif pid[i] in src:
                        #     sub.ndata['x'][i, 2] = 1
                    # sub.ndata['x'] = torch.randn(len(nodes), 1)
                    self.l.append([sub, 0])
                else:
                    sub = "一个节点的局部结构"
                    sub.ndata['x'] = torch.zeros([1, 6])
                    self.l.append([sub, 0])
            # print(len(index0))
            # print(len(self.l))
            # ###########

            for n in index1:
                path_v = []
                path3 = "获取目标节点之间的所有路径"  # all simple paths
                for j in path3:
                    if len(j) == 4:
                        path_v.append(j)
                    if len(path_v) > 0:
                        for h in path_v:
                            hop1.append(h[1])
                            hop2.append(h[2])
                nodes = "重复节点去重"
                if nodes:
                    sub = "根据节点获取局部结构"
                    sub.ndata['x'] = torch.zeros([len(nodes), 6])
                    pid = sub.ndata[dgl.NID]
                    for i in range(pid.shape[0]):
                        # # #四位特征
                        # if pid[i] == n[0]:
                        #     sub.ndata['x'][i, 0] = 1
                        # elif pid[i] == n[1]:
                        #     sub.ndata['x'][i, 1] = 1
                        # elif pid[i] in hop1:
                        #     sub.ndata['x'][i, 2] = 1
                        # elif pid[i] in hop2:
                        #     sub.ndata['x'][i, 3] = 1
                        # #六位特征
                        if pid[i] in src:
                            sub.ndata['x'][i, 0] = 1
                        if pid[i] in dst:
                            sub.ndata['x'][i, 1] = 1
                        if pid[i] == n[0]:
                            sub.ndata['x'][i, 2] = 1
                        elif pid[i] == n[1]:
                            sub.ndata['x'][i, 5] = 1
                        elif pid[i] in hop1:
                            sub.ndata['x'][i, 3] = 1
                        elif pid[i] in hop2:
                            sub.ndata['x'][i, 4] = 1
                        # 两位特征
                        # if pid[i] in dst:
                        #     sub.ndata['x'][i, 1] = 1
                        # elif pid[i] in src:
                        #     sub.ndata['x'][i, 2] = 1
                    # sub.ndata['x'] = torch.randn(len(nodes), 1)
                    self.l.append([sub, 1])
                else:
                    sub = "一个节点的局部结构"
                    sub.ndata['x'] = torch.zeros([1, 6])
                    self.l.append([sub, 1])
            # print(len(self.l))
            # print(len(index1))
            for id in self.l:
                self.graphs.append(id[0])
                self.labels.append(id[1])

            # Convert the label list to tensor for saving.
            self.labels = torch.LongTensor(self.labels)

        def __getitem__(self, i):
            return self.graphs[i], self.labels[i]

        def __len__(self):
            return len(self.graphs)


    def collate(samples):
        # The input `samples` is a list of pairs
        #  (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)


    data = SyntheticDataset()

    train, test = train_test_split(data, test_size=0.2)
    data_loader = DataLoader(train, batch_size=5, shuffle=True, collate_fn=collate)
    data_test = DataLoader(test, batch_size=5, shuffle=True, collate_fn=collate)


    # print(data_loader)

    # print(data)
    # print(train)


    class Classifier(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_classes):
            super(Classifier, self).__init__()
            self.conv1 = GraphConv(in_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, hidden_dim)
            self.classify = nn.Linear(hidden_dim, n_classes)

        def forward(self, g):
            # 使用节点的入度作为初始特征
            # ####################
            # h = g.in_degrees().view(-1, 1).float()
            h = g.ndata["x"]
            # ####################
            h = F.relu(self.conv1(g, h))
            h = F.relu(self.conv2(g, h))
            g.ndata['h'] = h  # # 节点特征经过两层卷积的输出
            hg = dgl.mean_nodes(g, 'h')  # 图的特征是所有节点特征的均值
            # hg = dgl.max_nodes(g, 'h')
            y = self.classify(hg)
            return y


    model = Classifier(6, 25, 1)  # #4是特征的维度，25是隐藏层节点数，2是分类结果
    # print(model)
    # print(trainset.num_classes)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    model.train()

    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        epoch_rmse = 0
        epoch_mae = 0
        rmse_ = 0
        for iter, (bg, label) in enumerate(data_loader):
            # print(bg)
            prediction = model(bg)
            # label = label.long()
            label = label.to(torch.float32)
            label = label.view(-1, 1)
            mse = MSE(prediction, label)  # MSE
            mae = MAE(prediction, label)
            rmse = torch.sqrt(mse)
            ########
            # rmse_ = 0rmse_
            # rmse_ += rmse.detach().item()
            # if iter % 5 == 0:
            #     print(rmse_/5)
            #     rmse_ = 0
            #########
            # print(prediction)
            # print(label)
            loss = loss_func(prediction, label)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_mae += mae  #
            epoch_rmse += rmse  #
        epoch_loss /= (iter + 1)
        epoch_rmse /= (iter + 1)  #
        # test(data_test)
        model.eval()
        test_rmse = 0
        test_mae = 0
        for iter, (bg, label) in enumerate(data_test):
            # print(bg)
            prediction = model(bg)
            # label = label.long()
            label = label.to(torch.float32)
            label = label.view(-1, 1)
            mse = MSE(prediction, label)  # MSE
            mae = MAE(prediction, label)
            rmse = torch.sqrt(mse)
            # rmse_ = 0
            # rmse_ += rmse
            # if iter % 50 == 0:
            #     print(rmse_)
            test_rmse += rmse
            test_mae += mae
        test_rmse /= (iter + 1)
        test_mae /= (iter + 1)
        print('Epoch {}, loss {:.4f}, rmse {:.4f}, test_rmse {:.4f}, test_mae {:.4f}'.format(epoch, epoch_loss,
                                                                                             epoch_rmse,
                                                                                             test_rmse, test_mae))
        # epoch_losses.append(epoch_loss)
    result_RMSE.append(test_rmse.detach().item())
    result_MAE.append(test_mae.detach().item())
print(np.mean(result_RMSE))
print(np.mean(result_MAE))
