import numpy as np
import math 


def dropout_regularizer(X, dropout=0.3):
    assert 0 <= dropout <= 1# In this case, all elements are dropped out
    if dropout == 1:
        return np.zeros(X.shape, dtype=np.float32)# In this case, all elements are kept
    if dropout == 0:
        return X
    mask = np.random.rand(X.shape, dtype=np.float32) > dropout
    return np.multiply(X, mask) / (1.0 - dropout)

def relu(x):
    return np.maximum(0, x)

def init_dense_params(in_nodes, out_nodes): # params are of only one layer
    # xavier initialization
    params = []
    a = math.sqrt(6/(layers[i] + layers[i+1]))
    params.append(np.random.uniform(-a, a, (in_nodes, out_nodes)))
    #params.append(torch.Tensor((layers[i], layers[i+1]), requires_grad=True).uniform_(-a, a))
    params.append(np.random.uniform(-a, a, (out_nodes)))   
    #params.append(torch.randn(layers[i+1], dtype = torch.float32, requires_grad=True))
    return params

def dense_batch(x, params, dropout = False): # we will apply dropout only when training is going on
    op = x
    op = relu(torch.matmul(op, params[0]) + params[1])
    if dropout:
        op = dropout_regularizer(op)
    return op # shape will be (16, 3)


def accuracy(x, y, params):
    op = forward(x, params) # shape is (210, 3)
    _, ind = torch.max(op, axis=1)
    return torch.sum(ind==y)/y.shape[0]

def pred(x, params):
    op = forward(x, params)


