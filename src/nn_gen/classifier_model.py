import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def backprop(self, data, loss_fn, optimizer):
        self.train()
        optimizer.zero_grad()
        outputs = self.forward(data.x_train)
        loss = loss_fn(outputs, data.y_train.squeeze().long())
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == data.y_train.squeeze().long()).sum().item() / data.y_train.size(0)
        return loss.item(), accuracy
