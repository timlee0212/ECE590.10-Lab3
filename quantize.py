import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from pruned_layers import *
import torch.nn as nn
import heapq

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            #Eliminate Prunned Values
            ori_weight = m.conv.weight.data.cpu().detach().numpy()
            weight = ori_weight[ori_weight!=0].reshape(-1, 1)

            #Deal With Exception
            #No need to quantize
            if weight.shape[0] <= (2**bits):
                print("Parameter Size %d less than the encoding size %d, skip."%(weight.shape[0], 2**bits))
                cur_centeroids = np.zeros((1, 2**bits)).flatten()
                cur_centeroids[:weight.shape[0]] = weight.flatten()
            else:
                #Initilize centroids with linear method
                _min = np.min(weight)
                _max = np.max(weight)
                #TODO: Implement Different Initialization Functions
                cur_centeroids = np.linspace(_min, _max, num=2**bits).reshape(-1, 1)

                kmeans = KMeans(n_clusters=len(cur_centeroids), init=cur_centeroids, algorithm='full')
                kmeans.fit(weight)

                new_weight = np.zeros_like(ori_weight)
                new_weight[ori_weight!=0] = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

                wdict = m.conv.state_dict()
                wdict['weight'] = torch.tensor(new_weight).to(device)
                m.conv.load_state_dict(wdict)

            cluster_centers.append(cur_centeroids.flatten())
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            #Eliminate Prunned Values
            ori_weight = m.linear.weight.data.cpu().detach().numpy()
            weight = ori_weight[ori_weight!=0].reshape(-1, 1)

            if weight.shape[0] <= (2**bits):
                cur_centeroids = np.zeros((1, 2**bits)).flatten()
                cur_centeroids[:weight.shape[0]] = weight.flatten()

            else:
                #Initilize centroids with linear method
                _min = np.min(weight)
                _max = np.max(weight)
                #TODO: Implement Different Initialization Functions
                cur_centeroids = np.linspace(_min, _max, num=2**bits).reshape(-1, 1)

                kmeans = MiniBatchKMeans(n_clusters=len(cur_centeroids), init=cur_centeroids)
                kmeans.fit(weight)

                new_weight = np.zeros_like(ori_weight)
                new_weight[ori_weight!=0] = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

                wdict = m.linear.state_dict()
                wdict['weight'] = torch.tensor(new_weight).to(device)
                m.linear.load_state_dict(wdict)

            cluster_centers.append(cur_centeroids.flatten())
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

