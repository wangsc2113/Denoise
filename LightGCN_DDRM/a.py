import torch

a = torch.randn((4, 3))
print (a)

index = torch.LongTensor([0, 2])
b = torch.randn((2, 3))
print (b)

a.index_copy_(dim=0, index=index, source=b)
print (a)

print (a[index])