import torch
import random
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP*TN-FP*FN)/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    F1_score = 2*(precision*sensitivity)/(precision+sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(data_ID, output_dim):
    interaction = pd.read_csv('dataset/dataset{}/interaction_dataset{}.csv'.format(data_ID, data_ID),
                              delimiter=',', header=None)
    lncRNA_feature = pd.read_csv('dataset/dataset{}/ml.csv'.format(data_ID), delimiter=',', header=None)
    protein_feature = pd.read_csv('dataset/dataset{}/mp.csv'.format(data_ID), delimiter=',', header=None)

    m_emb = torch.Tensor(lncRNA_feature.values)
    print(m_emb.size())
    m_emb = torch.cat([m_emb, torch.zeros(m_emb.size(0), output_dim - m_emb.size(1))], dim=1)
    s_emb = torch.Tensor(protein_feature.values)
    print(s_emb.size())
    s_emb = torch.cat([s_emb, torch.zeros(s_emb.size(0), output_dim - s_emb.size(1))], dim=1)

    feature = torch.cat([m_emb, s_emb])

    l, p = interaction.values.nonzero()
    adj = torch.tensor([p, l + len(protein_feature)])
    data = Data(x=feature, edge_index=adj).cuda()

    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                                 is_undirected=True, split_labels=True,
                                                 add_negative_train_samples=True)(data)
    splits = dict(train=train_data, test=test_data)
    return splits


if __name__ == '__main__':
    data = get_data(2, 2048)
