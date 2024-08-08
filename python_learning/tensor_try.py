
# a = torch.randn(3, 2)
# print(a.stride( ))  # (2, 1)

import torch

a = torch.arange(0, 6)
print('a = {}\n'.format(a))
print('tensor a 存储区的数据内容 ：{}\n'.format(a.storage()))
print('tensor a 相对于存储区数据的偏移量 ：{}\n'.format(a.storage_offset()))

print('*'*20, '\n')

b = a.view(2,3)
print('b = {}\n'.format(b))
print('tensor b 存储区的数据内容 ：{}\n'.format(b.storage()))
print('tensor b 相对于存储区数据的偏移量 ：{}\n'.format(b.storage_offset()))