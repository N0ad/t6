import torch
import struct
import torch.nn as nn

def save_bin(path, inn):
    f = open(path, 'wb')
    for item in inn:
        s = str(item) + '\n'
        bt = s.encode()
        f.write(bt) 
    f.close()


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(32*32, 16*16)
        self.fc2 = nn.Linear(16*16, 4*4)
        self.fc3 = nn.Linear(4*4,1)

    def forward(self, x):

        sigmoid = nn.Sigmoid()

        x = sigmoid(self.fc1(x))

        x = sigmoid(self.fc2(x))

        x = sigmoid(self.fc3(x))

        return x

inp = torch.rand(32*32)
net = Net()
save_bin('./inp.bin', inp.detach().numpy())
res = net(inp)
print(res)

num, count = 1, 1
for j in net.parameters():
    if (count % 2) == 1:
        save_bin(f'./w{num}.bin', j.detach().numpy().flatten())
    else:
        save_bin(f'./b{num}.bin', j.detach().numpy().flatten())
        num+=1
    count+=1
