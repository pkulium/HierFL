import copy
import torch
from torch import nn

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], torch.tensor(s_num[i]/temp_sample_num, dtype=torch.float))
        w_avg[k] = torch.mul(w_avg[k], torch.tensor(temp_sample_num/total_sample_num, dtype=torch.float))
    return w_avg
