import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 优先使用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(root='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载模型
model1_save_path = 'best_model.pth'  # resnet50
if os.path.exists(model1_save_path):
    model1 = torch.load(model1_save_path, map_location=device)
else:
    raise FileNotFoundError(f"加载模型失败: 未找到文件 '{model1_save_path}'.")

model2_save_path = 'modelSave/vgg16_voc2005.pth'  # vgg16
if os.path.exists(model2_save_path):
    model2 = torch.load(model2_save_path, map_location=device)
else:
    raise FileNotFoundError(f"加载模型失败: 未找到文件 '{model2_save_path}'.")

#XGB训练集(resnet每个类别概率,vgg每个类别概率,实际值)
df = pd.DataFrame(columns=['c0', 'c1', 'c2', 'd0', 'd1', 'd2', 'v0', 'v1', 'v2'])

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 获取模型的输出
        output1 = torch.softmax(model1(images), dim=1).cpu().numpy()
        output2 = torch.softmax(model2(images), dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        # 逐批次将数据写入DataFrame
        new_rows = []
        for i in range(len(labels)):
            new_rows.append({
                'c0': output1[i, 0],
                'c1': output1[i, 1],
                'c2': output1[i, 2],
                'd0': output2[i, 0],
                'd1': output2[i, 1],
                'd2': output2[i, 2],
                'v0': 1 if labels[i] == 0 else 0,
                'v1': 1 if labels[i] == 1 else 0,
                'v2': 1 if labels[i] == 2 else 0
            })

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# 拆分训练集和验证集
X = df[['c0', 'c1', 'c2', 'd0', 'd1', 'd2']].values
y = df[['v0', 'v1', 'v2']].values.argmax(axis=1)  # 转换为单一类标签

#20的验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 XGBoost 分类器，使用 GPU 训练
model = xgb.XGBClassifier(
    objective='multi:softmax',  # 多分类任务
    num_class=3,  # 类别数
    eval_metric='mlogloss',  # 多分类log损失
    use_label_encoder=False,
    tree_method='gpu_hist'
)

# 训练模型
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# 预测验证集
y_pred = model.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 保存模型
model.save_model('xgboost_model.json')