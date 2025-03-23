import torch
import torch.nn as nn
import torch.nn.functional as F

class HandGestureNet(nn.Module):
    def __init__(self, num_classes):
        super(HandGestureNet, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3)
        return self.fc3(x)  # raw logits (softmax in loss)
