import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets



criterion = nn.CrossEntropyLoss()
# 测试单个模型的函数
def test_single_model(model, data_loader, device):
    """
    测试单个模型,计算准确率、精度、召回率、F1分数和损失值。
    """
    model.eval()  # 切换到评估模式
    y_pred = []
    y_true = []
    total_loss = 0.0

    with torch.no_grad():  # 在评估模式下禁用梯度计算以提高效率
        # 遍历测试集的图片
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)  # 模型预测
            loss = criterion(output, labels)  # 计算损失
            total_loss += loss.item()  # 累积损失值

            predicted_labels = torch.argmax(output, dim=1).cpu().numpy()  # 获取预测类别
            y_pred.extend(predicted_labels)
            y_true.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    avg_loss = total_loss / len(data_loader)  # 计算平均损失

    # 返回结果字典
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss  # 增加损失值
    }

    return results