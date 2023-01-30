import numpy as np
import math 
import torch

def pad_single_image(x, pad_size, device): # pad just with respect to one image # working fine
    '''
     pad_size will be a tuple.
     remember that x will be of shape (num_channels, height , width),  only be applying zero padding 
    '''
    op = x
    num_channels, height, width = x.shape 
    #print(x.shape)
    if pad_size[0]!=0:
        pad_vert = torch.zeros((num_channels, pad_size[0], width), dtype=torch.float32, device=device)
        #print("pad_vert.shape = ", pad_vert.shape)
        op = torch.cat((pad_vert, op, pad_vert), axis=1)
        #print("op = ", op.shape) # shape must be (num_channels, height+2*pad_size[0], width)
    if pad_size[1]!=0:
        pad_horiz = torch.zeros((num_channels, height+2*pad_size[0],  pad_size[1]), dtype=torch.float32, device=device)
        #print("pad_horiz.shape = ", pad_horiz.shape)
        op = torch.cat((pad_horiz, op, pad_horiz), axis=2)
        #print("op = ", op.shape) # shape must be (num_channels, height+2*pad_size[0], width+2*pad_size[1])
    return op




# pooling a 2D array
def pool_arr(x, kernel_size,  stride, mode, device): # works fine
    '''
    x will be a 2D array
    kernel_size will be a tuple
    stride will be a tuple
    mode will be either 'max' or 'avg'
    '''
    # let x be of shape (5, 5), kernel be of  shape(2,2), stride be of shape(2, 2)
    # so resultant shape must be (2,2)
    op = []
    for i in range(0, x.shape[0], stride[0]):  
        if i+kernel_size[0]<=x.shape[0]:
            row = []
            for j in range(0, x.shape[1], stride[1]):
                if j+kernel_size[1]<=x.shape[1]:
                    if mode=='max':
                        row.append(torch.max(x[i:i+kernel_size[0], j:j+kernel_size[1]]))
                    else: # mode=='avg'
                        row.append(torch.mean(x[i:i+kernel_size[0], j:j+kernel_size[1]]))
            op.append(torch.stack(row))
    return torch.stack(op) # this will be 2D tensor

# pooling a 3D image
def pool_single_img(x, kernel_size, pad_size, stride ,mode, device): # pool just with respect to one image, works fine
    '''
    kernel_size will be a tuple
    remember this is a parameterless operation to do
    '''
    x = pad_single_image(x, pad_size, device)
    op = torch.stack([pool_arr(x[i], kernel_size, stride, mode, device) for i in range(x.shape[0])])
    return op

# convolution of a 3D image
def conv_single_image(x, params, pad_size, stride, device):
    '''
    remember params is a list of weights and biases alternatively like [w1, b1, w2, b2, ..., w_out_channels, b_out_channels]
    all w_i shape will be (in_channels, _ , _) and b_i will be scalars
    '''
    x = pad_single_image(x, pad_size, device=device)
    out_channels = len(params)//2 # according to the format desired
    in_channels, p, q = params[0].shape # in_channels, _, _
    op = []
    for i in range(out_channels): # output no. of channels 
        channel = []
        for j in range(0, x.shape[1], stride[0]): # x.shape[1] = height
            if j+p<=x.shape[1]:
                row =[]
                for k in range(0, x.shape[2], stride[1]): # x.shape[2] = width
                    if k+q<=x.shape[2]:
                        kernel, bias = params[2*i], params[2*i +1]
                        row.append(torch.sum(torch.multiply(kernel, x[:, j:j+p, k:k+q])) + bias[0])
                #print('row = ', row)
                #print("row.shape = ", torch.stack(row).shape)
                channel.append(torch.stack(row)) # 1D 
        op.append(torch.stack(channel)) # 2D
    return torch.stack(op) # 3D





# padding
def pad_batch(x, pad_size, device): # works fine
    '''
    shape of x will be - (batch_size, num_channles, height, width)
    '''
    op = torch.stack([pad_single_image(x[i], pad_size, device) for i in range(x.shape[0])])
    return op

# pooling
def pool_batch(x, kernel_size, pad_size, stride, mode, device): # works fine
    '''
    shape of x will be - (batch_size, num_channles, height, width)
    '''
    op = torch.stack([pool_single_img(x[i], kernel_size, pad_size, stride, mode, device) for i in range(x.shape[0])])
    return op


# conv forward for a batch 
def conv_batch(x, params, pad_size, stride, device):
    '''
    shape of x will be - (batch_size, num_channles, height, width)
    '''
    op = torch.stack([conv_single_image(x[i], params, pad_size, stride, device) for i in range(x.shape[0])])
    return op



def init_conv_params(in_channels, out_channels, kernel_size, device):
    # xavier initialization for breaking the symmetry and stable training with using 
    height, width = kernel_size
    var = math.sqrt(6/(in_channels + out_channels))
    op = []
    for i in range(out_channels):
        weight = torch.tensor(np.random.uniform(-var, var, (in_channels, height, width)), dtype=torch.float32, device = device, requires_grad=True)   
        bias = torch.tensor(np.random.uniform(-var, var, (1)), dtype=torch.float32, device=device, requires_grad=True)
        op.append(weight)
        op.append(bias)
    return op