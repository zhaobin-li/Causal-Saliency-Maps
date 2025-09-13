import itertools
import random

import torch

n = 3
lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
# random.shuffle(lst) # shouldn't change leverage values

lst = torch.tensor(lst).float().cuda()

# lst = lst - 0.5 # identical diagonal elements
XtX = lst.T @ lst
print("Eigen XtX", torch.linalg.eig(XtX))
G = torch.linalg.inv(XtX)
print("Eigen G", torch.linalg.eig(G))

H = lst @ G @ lst.T
# print(H)
print("G", G)
print("Lev", torch.diagonal(H))
# print(lst)
print(lst[torch.diagonal(H) == torch.diagonal(H).max()])
