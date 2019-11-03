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

print("-----Summary before pruning-----")
summary(net)
print("-------------------------------")

### Pruning & Finetune with pruned connections
# Test accuracy before fine-tuning



sensitive = np.linspace(0, 3, 30).flatten()
percentage = np.linspace(0, 95, 30).flatten()
sparsity = np.zeros((1, 30)).flatten()
acc = np.zeros((1, 30)).flatten()

f = open("result_per.csv", "w")
f.write("s, sparsity, acc\n")

for i in range(30):
    model = copy.deepcopy(net)
    prune(model, method='percentage', q=percentage[i], s=0.45)
    acc[i] = test(model)

    print("-----Summary after pruning-----")
    sparsity[i] = summary(model)
    print("-------------------------------")
    f.write("%f, %f, %f\n"%(percentage[i], sparsity[i], acc[i]))


