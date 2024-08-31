import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取数据
df = pd.read_csv('./ML04california.csv', header=0, index_col=0)
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

# 数据标准化
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# 将数据转换为PyTorch张量
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y.values, dtype=torch.float32).view(-1, 1)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y.values, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 实例化模型并移动到设备
model = MLP(input_size=train_x.shape[1]).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
