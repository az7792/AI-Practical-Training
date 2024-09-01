import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取数据
df = pd.read_csv('./ML04california.csv', header=0, index_col=0)
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
torch.set_grad_enabled(True)

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
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# 实例化模型并移动到设备
model = MLP(input_size=train_x.shape[1]).to(device)

# 读取模型
model_load_path = './mlp_model.pth'
try:
    model.load_state_dict(torch.load(model_load_path))
    model.to(device)
    print(f'Model loaded from {model_load_path}')
except FileNotFoundError:
    print(f'No model found at {model_load_path}, training from scratch.')

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
losses = []
num_epochs = 300
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
    losses.append(running_loss/len(train_loader))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 将损失保存到losses.csv
loss_df = pd.DataFrame(losses)
loss_df.to_csv('losses.csv', mode='a', index=False, header=not os.path.exists('losses.csv'))

# 保存模型
model_save_path = './mlp_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')


# 测试模型
model.eval()
with torch.no_grad():
    train_preds = []
    train_targets = []
    test_preds = []
    test_targets = []

    # 训练集预测
    for batch_x, batch_y in DataLoader(train_dataset, batch_size=32, shuffle=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        train_preds.append(outputs.cpu().numpy())
        train_targets.append(batch_y.cpu().numpy())

    # 测试集预测
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        test_preds.append(outputs.cpu().numpy())
        test_targets.append(batch_y.cpu().numpy())

    train_preds = np.concatenate(train_preds, axis=0)
    train_targets = np.concatenate(train_targets, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    # 计算MSE和R2得分
    train_mse = mean_squared_error(train_targets, train_preds)
    train_r2 = r2_score(train_targets, train_preds)
    test_mse = mean_squared_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)

    print(f'Training MSE: {train_mse:.4f}')
    print(f'Training R2: {train_r2:.4f}')
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test R2: {test_r2:.4f}')