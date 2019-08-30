# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:20:20 2019

@author: yoavb
"""

import torch


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
else:
    print("no usable gpus")
    
a = torch.cuda.FloatTensor([1.])
