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

net.load_state_dict(torch.load("net_after_pruning.pt"))
acc = test(net)

while acc>0.9:
    # Load the best weight paramters
    net.load_state_dict(torch.load("net_after_pruning.pt"))
    test(net)

    # Test accuracy before fine-tuning
    prune(net, method='std', q=45.0, s=1.25)
    test(net)

    finetune_after_prune(net, epochs=50, batch_size=128, lr=0.001, reg=5e-4)
    net.load_state_dict(torch.load("net_after_pruning.pt"))
    acc = test(net)
    spar = summary(net)
    torch.save(net.state_dict(), "net_after_pruning%.2f_%.2f.pt"%(acc, spar))
