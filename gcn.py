import argparse
import copy
import logging
import pdb
import pickle
import random
from collections import namedtuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

logger = logging.getLogger(__name__)


Data = namedtuple('Data', ['x', 'a', 'y', 'num_classes', 'train_index', 'dev_index', 'test_index'])


class GraphConv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.bias.data.fill_(0)

    def forward(self, x, a):
        x = torch.mm(x, self.weight)
        x = torch.spmm(a, x)
        x = x + self.bias
        return x


class GCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate):
        super().__init__()
        self.conv1 = GraphConv(input_size, hidden_size)
        self.conv2 = GraphConv(hidden_size, output_size)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, a = data.x, data.a
        x = self.conv1(x, a)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, a)
        return F.log_softmax(x, dim=1)


def to_sparse_tensor(x):
    x = x.tocoo()
    i = torch.tensor(np.vstack((x.row, x.col)), dtype=torch.long)
    v = torch.tensor(x.data, dtype=torch.float)
    return torch.sparse_coo_tensor(i, v, torch.Size(x.shape))


def normalize(x):
    return scipy.sparse.diags(np.array(x.sum(1)).flatten() ** -1).dot(x)


def read_file(name):
    filename = f'data/ind.cora.{name}'
    if name == 'test.index':
        return np.loadtxt(filename, dtype=np.long)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f, encoding='latin1')


def load_data(device):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
    x, y, tx, ty, allx, ally, graph, test_index = [read_file(name) for name in names]
    num_classes = y.shape[1]
    train_index = torch.arange(y.shape[0]).to(device)
    dev_index = torch.arange(y.shape[0], y.shape[0] + 500).to(device)
    test_index_sorted = torch.tensor(np.sort(test_index)).to(device)
    test_index = torch.tensor(test_index).to(device)

    x = torch.tensor(normalize(scipy.sparse.vstack([allx, tx])).todense()).to(device)
    y = torch.tensor(np.vstack([ally, ty]).argmax(axis=1)).to(device)

    x[test_index] = x[test_index_sorted]
    y[test_index] = y[test_index_sorted]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    a = to_sparse_tensor(normalize(adj + scipy.sparse.eye(adj.shape[0]))).to(device)

    return Data(x, a, y, num_classes, train_index, dev_index, test_index)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config_data = f.read()
    config = yaml.load(config_data)

    logging.basicConfig(
        handlers=[logging.FileHandler(config['log_file'], mode='w'), logging.StreamHandler()],
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    logger.info('Configuration:\n' + config_data)

    if config['seed']:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    use_gpu = args.gpu is not None and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_gpu else 'cpu')
    logger.info(f'Device: {device}')

    return config, device


def evaluate(gcn, data, split_index):
    z = gcn(data)[split_index]
    y = data.y[split_index]
    loss = F.nll_loss(z, y)
    acc = torch.sum(torch.argmax(z, dim=1) == y).item() / y.shape[0]
    return loss, acc


def main():
    config, device = setup()
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    data = load_data(device)

    gcn = GCN(data.x.shape[1], data.num_classes, config['hidden_size'], config['dropout_rate']).to(device)
    optimizer = optim.Adam(gcn.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_dev_acc = 0

    for i in range(1, config['iterations'] + 1):
        gcn.train()
        loss, acc = evaluate(gcn, data, data.train_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gcn.eval()
        with torch.no_grad():
            dev_loss, dev_acc = evaluate(gcn, data, data.dev_index)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_model = copy.deepcopy(gcn)

        logger.info(
            'Iter: {:6d}    '
            'Train loss: {:.4f}    '
            'Train acc: {:.4f}    '
            'Dev loss: {:.4f}    '
            'Dev acc: {:.4f}'.format(i, loss.item(), acc, dev_loss.item(), dev_acc)
        )

    best_model.eval()
    with torch.no_grad():
        _, test_acc = evaluate(best_model, data, data.test_index)
        logger.info(f'Test acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()
