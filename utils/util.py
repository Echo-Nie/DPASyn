import numpy as np
from torch_geometric.data import Data, Batch
import torch
from torch.utils.data import Dataset
import pickle


class MyDataset(Dataset):
    def __init__(self, data=None):
        if not data:
            with open('../data/data.pt', 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = data

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.data[item]

    def get_data(self, slice):
        d = [self.data[i] for i in slice]
        return MyDataset(d)


def collate(data):
    d1_list, d2_list, label_list = [], [], []
    for d in data:
        graph1, graph2, cell, label = d[0], d[1], d[2], d[3]

        graph1.cell = cell

        d1_list.append(graph1)
        d2_list.append(graph2)
        label_list.append(label)

    return Batch.from_data_list(d1_list), Batch.from_data_list(d2_list), torch.tensor(label_list)


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, AUCs)) + '\n')


if __name__ == '__main__':
    with open('../data/data.pt', 'rb') as f:
        data = pickle.load(f)
    d1_list, d2_list, label_list = [], [], []

    for d in data[15:30]:
        d1_list.append(d[0])
        d2_list.append(d[1])
    print(d2_list)
    # b1 = Batch.from_data_list(d1_list)
    b2 = Batch.from_data_list(d2_list)
