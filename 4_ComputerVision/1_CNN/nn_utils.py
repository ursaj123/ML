import numpy as np
import torch 
import math 





def dropout_regularizer(X, device, dropout=0.3):
    assert 0 <= dropout <= 1# In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros(X.shape, dtype=torch.float32, device=device)# In this case, all elements are kept
    if dropout == 0:
        return X
    mask = torch.rand(X.shape, dtype=torch.float32, device=device) > dropout
    return torch.multiply(X, mask) / (1.0 - dropout)

def relu(x, device):
    return torch.maximum(torch.tensor([0], dtype=torch.float32, device=device), x)


def init_dense_params(in_nodes, out_nodes, device):
    # xavier initialization
    params = []
    a = math.sqrt(6/(in_nodes + out_nodes))
    params.append(torch.tensor(np.random.uniform(-a, a, (in_nodes, out_nodes)), dtype=torch.float32, device=device, requires_grad=True))   
    #params.append(torch.Tensor((layers[i], layers[i+1]), requires_grad=True).uniform_(-a, a))
    params.append(torch.tensor(np.random.uniform(-a, a, (out_nodes)), dtype=torch.float32, device=device, requires_grad=True))   
    #params.append(torch.randn(layers[i+1], dtype = torch.float32, requires_grad=True))
    return params

def dense_batch(x, params, device, dropout = False): # we will apply dropout only when training is going on
    op = x
    op = relu(torch.matmul(op, params[0]) + params[1], device)
    if dropout:
        op = dropout_regularizer(op, device)
    return op # shape will be (16, 3)


def accuracy(x, y, params, device):
    op = dense_batch(x, params, device) # shape is (210, 3)
    _, ind = torch.max(op, axis=1)
    return torch.sum(ind==y)/y.shape[0]

def pred(x, params, device):
    op = dense_batch(x, params, device)