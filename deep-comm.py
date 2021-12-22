from __future__ import division
from __future__ import print_function

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
############model################
import torch.nn as nn
import torch.nn.functional as F
#############uuuutils#####################
import scipy.sparse as sp
import numpy as np
from pandas_ml import ConfusionMatrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
#from statistics import mean 
######################train###########


import time
import argparse

import torch.optim as optim

#from pygcn.layers import GraphConvolution
#Layyyyyyers##################################



class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


#MMMMMMMMMMMMMMMMMMMMModelsssssssssssssssssssss
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


###############Utilsssss#####################################

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


#def load_data(path="../data/cora/", dataset="cora"):
#def load_data(path="./data/email/", dataset="email"):
#def load_data(path="../data/enron/", dataset="enron"):
#def load_data(path="./data/p2p-g04/", dataset="p2p-g04"):
#def load_data(path="../data/ca/", dataset="ca"):
#def load_data(path="./data/fb/", dataset="fb"):
def load_data(path="./data/ca-gr/", dataset="ca-gr"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    #types = ['i4', 'f4', 'f4', 'f4', 'f4','f4', 'f4', 'f4','f4','f4','f4','f4','i4']
    #Graph node features
    idx_features_labels = np.genfromtxt("{}{}.csv".format(path, dataset),delimiter=',')
    #idx_features_labels=np.array(idx_features_labels)
    #print(idx_features_labels)
    #print(idx_features_labels.shape)
    #idx_features_labels.reshape((idx_features_labels.shape[0], 13))
    #print(idx_features_labels[:,:-1])

    #Identity Matrix
    #idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
    
    #print(idx_features_labels)
    #print(idx_features_labels[:,:-1])
    #f_features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    f_features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    f_labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    f_idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    print(f_idx)




    print(f_idx.shape)
    #j=1
    idx_map = {j: i for i, j in enumerate(f_idx)}
    print(idx_map)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    print(edges_unordered)
    print(edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    print(edges)
    print(edges.shape)
    f_adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(f_labels.shape[0], f_labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    f_adj = f_adj + f_adj.T.multiply(f_adj.T > f_adj) - f_adj.multiply(f_adj.T > f_adj)

    f_features = normalize(f_features)
    f_adj = normalize(f_adj + sp.eye(f_adj.shape[0]))

	##email
    #f_idx_train = range(200)#20%
    #f_idx_val = range(201, 300)
    #f_idx_test = range(300, 1005)


    #idx_train = range(500)#
    #idx_val = range(501, 700)
    #idx_test = range(700, 1005)


    ##fbbbb

    #f_idx_train = range (500)#12.3%
    #f_idx_val = range (500, 700)
    #f_idx_test = range (700, 4039)


    ##ca-gr

    f_idx_train = range (524)#10%
    f_idx_val = range (524, 700)
    f_idx_test = range (700, 5242)

                            #cora
    
    #idx_train = range(140)
    #idx_val = range(200, 500)
 #   idx_test = range(500, 1500)

	

	##ennnnronnn
	#	idx_train = range(3600)
	#    idx_val = range(3600, 4000)
	#    idx_test = range(4000, 36692)

 #   idx_train = range(10000)
  #  idx_val = range(10000, 15000)
   # idx_test = range(15000, 36692)

	#####p2p-g04
    #f_idx_train = range(1000)
    #f_idx_val = range(1000, 1500)
    #f_idx_test = range(1500, 10876)

	#####ca

    #idx_train = range(892)
    #idx_val = range(892, 1500)
    #idx_test = range(1500, 89209)

    f_features = torch.FloatTensor(np.array(f_features.todense()))
    f_labels = torch.LongTensor(np.where(f_labels)[1])
    f_adj = sparse_mx_to_torch_sparse_tensor(f_adj)

    f_idx_train = torch.LongTensor(f_idx_train)
    f_idx_val = torch.LongTensor(f_idx_val)
    f_idx_test = torch.LongTensor(f_idx_test)

    return f_adj, f_features, f_labels, f_idx_train, f_idx_val, f_idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def model_metrics(output, labels):
	preds = output.max(1)[1].type_as(labels)
#	y_actu = pd.Series(labels, name='Actual')
#	y_pred = pd.Series(preds, name='Predicted')
#	df_confusion = pd.crosstab(y_actu, y_pred)
	cm = ConfusionMatrix(labels, preds)
#	cm.stats()
	cm.print_stats()
	print(cm._avg_stat("TPR"))
	print("Accuracy", round(np.mean(cm.ACC),2))
	print("Recall/Sensitivity:", round(np.mean(cm.TPR),2))
	print("Precision:", round(np.mean(cm.PPV),2))
	print("Specificity:", round(np.mean(cm.TNR),2))
	print("MCC:",round(np.mean(cm.MCC),2) )


def performance_metrics(output, labels):
	preds = output.max(1)[1].type_as(labels)
	print("Accuracy Score: ", accuracy_score(labels, preds))
	print("Balanced Accuracy Score: ", balanced_accuracy_score(labels, preds))
	print("F1 Score: ", f1_score(labels, preds,average='weighted'))
	print("precision_recall_fscore_support Score: ", precision_recall_fscore_support(labels, preds,average='weighted'))
	#print("ROC AUC Score: ", roc_auc_score(labels, preds))



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)





#####################Traiiiining#############################
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#global adj, features, labels, idx_train, idx_val, idx_test
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
 #   print("Printing idx_train")
#    print (idx_train)
 #   print("Printing output[idx_train]")
  #  print(output[idx_train])
   # print("Printing labels[idx_train]")
    #print(labels[idx_train])

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    model_metrics(output[idx_test], labels[idx_test])
    performance_metrics(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
	
	
	
	
#    print("Printing idx_test")
 #   print (idx_test)
  #  print("Printing output[idx_test]")
   ###### print(output[idx_test])
   # print("Printing labels[idx_test]")
    #####print(labels[idx_test])


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


