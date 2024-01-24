import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Example usage
input_size = 100  # Example input size
hidden_size = 50  # Example hidden layer size
output_size = 10  # Example output size
dropout_rate = 0.5  # Dropout rate (0.5 is 50%)

model = MLP(input_size, hidden_size, output_size, dropout_rate)
print(model)  # Print model summary