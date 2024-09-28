import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from val import test_single_model
import pandas as pd

# 设置随机种子以确保可重复性
seed = 42
torch.manual_seed(seed)

# 优先使用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# 加载数据集
train_dataset = datasets.ImageFolder(root='train1', transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root='val', transform=data_transforms['val'])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 获取类别名称
class_names = train_dataset.classes

# 尝试加载已保存的模型
model_save_path = 'best_model.pth'
if os.path.exists(model_save_path):
    model = torch.load(model_save_path, map_location=device)
else:
    # 加载预训练的 ResNet 模型
    model = models.resnet50(weights='DEFAULT')
    # 替换最后一层全连接层以适应分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 50
best_acc = 0.0

train_val_df = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs-1}')    
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    #训练集指标
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    #验证集指标
    results = test_single_model(model=model,data_loader=val_loader,device=device)
    val_acc,val_loss = results['accuracy'],results['loss']
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # 保存训练记录
    train_val_df.loc[len(train_val_df)] = [epoch_loss,epoch_acc.cpu().item(),val_loss,val_acc]

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model, model_save_path)

# 保存训练记录
train_val_df.to_csv('train_val_data.csv', index=False, mode='a', header=False)

print(f'Best Acc: {best_acc:.4f}')