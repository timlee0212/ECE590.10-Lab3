from vgg16 import VGG16, VGG16_half
from train_util import train, finetune_after_prune, test
from quantize import quantize_whole_model
from huffman_coding import huffman_coding
from summary import summary
import torch
import numpy as np
from prune import prune

import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = VGG16_half()
net = net.to(device)

# Load the best weight paramters
net.load_state_dict(torch.load("net_before_pruning.pt"))
test(net)
#
print("-----Summary before pruning-----")
summary(net)
print("-------------------------------")
#
# ### Pruning & Finetune with pruned connections
# # Test accuracy before fine-tuning
#
prune(net, method='std', q=0.45, s=0.75)
#
# print("-----Summary after pruning-----")
summary(net)
# print("-------------------------------")
#


