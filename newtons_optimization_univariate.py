# Optimization of a univariate function using Newton's method in Pytorch

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision import transforms, utils
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import math
import random

def func(x):
  return 9 - x * (x - 10)

guess = torch.tensor(10., requires_grad = True)

for i in range(10):

  f = func(guess)
  f.backward()
  grad = guess.grad
  next_guess = guess - f / grad
  guess = Variable(next_guess, requires_grad = True)
  print(guess)

print('minima:', guess, 'Value at minima:', func(guess.item()))
