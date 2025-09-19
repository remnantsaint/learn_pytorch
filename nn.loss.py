import torch
from torch.nn import MSELoss

inputs = torch.tensor([1, 3, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))

loss = torch.nn.L1Loss() # 对应每位相减，取绝对值后全部相加
result = loss(inputs, targets)
print(result)

loss_mse = MSELoss()
result = loss_mse(inputs, targets)
print(result)

x = torch.tensor([0.1, 0.2, 0.7])
y = torch.tensor([2]) # 正确的标签索引
x = torch.reshape(x, (1, 3))
loss_cross = torch.nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross) # 越小越好