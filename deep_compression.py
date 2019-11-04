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
net.load_state_dict(torch.load("net_after_pruning0.91_0.96.pt"))
test(net)
#
print("-----Summary before pruning-----")
summary(net)
print("-------------------------------")

# centers = quantize_whole_model(net, bits=5)
# test(net)
#
# frequency_map, encoding_map = huffman_coding(net, centers)
# np.save("huffman_encoding", encoding_map)
# np.save("huffman_freq", frequency_map)

#
# ### Pruning & Finetune with pruned connections
# # Test accuracy before fine-tuning
#
# prune(net, method='std', q=0.45, s=0.75)
#
# print("-----Summary after pruning-----")
# summary(net)
# print("-------------------------------")
#
# finetune_after_prune(net, lr=0.05, reg=5e-5)

# f = open("result_quant_precise.csv", "w")
# f.write("bit, acc\n")
# bits = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
# acc = np.zeros((1, 16)).flatten()
# for bit in bits:
#     model = copy.deepcopy(net)
#     centers = quantize_whole_model(model, bits=bit)
#     #np.save("codebook_vgg16.npy", centers)
#     acc[bit-1] = test(model)
#     f.write("%d, %f\n"%(bit, acc[bit-1]))



#torch.save(net.state_dict(), "net_after_quantization.pt")



# sensitive = np.linspace(0, 3, 30).flatten()
# percentage = np.linspace(0, 95, 30).flatten()
# sparsity = np.zeros((1, 30)).flatten()
# acc = np.zeros((1, 30)).flatten()
#
# f = open("result_per.csv", "w")
# f.write("s, sparsity, acc\n")
#
# for i in range(30):
#     model = copy.deepcopy(net)
#     prune(model, method='percentage', q=percentage[i], s=0.45)
#     acc[i] = test(model)
#
#     print("-----Summary after pruning-----")
#     sparsity[i] = summary(model)
#     print("-------------------------------")
#     f.write("%f, %f, %f\n"%(percentage[i], sparsity[i], acc[i]))


