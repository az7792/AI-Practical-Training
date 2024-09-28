import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from val import test_single_model


def train_model(model, criterion, optimizer, dataloaders, image_datasets, device, num_epochs=25):
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # 设置模型为训练模式

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders:
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

        epoch_loss = running_loss / len(image_datasets)
        epoch_acc = running_corrects.double() / len(image_datasets)

        losses.append(epoch_loss)
        accuracies.append(epoch_acc.item())

        # print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        results = test_single_model(model=model, data_loader=val_loader, device=device)
        val_acc = results['accuracy']
        val_loss = results['loss']
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        print(f'Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f};  Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

    return model, losses, accuracies, val_losses, val_accuracies


if __name__ == '__main__':
    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # 数据集路径
    data_dir = './train'
    val_dir = './val'

    # 加载数据集，并应用预处理
    image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    dataloaders = DataLoader(image_datasets, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 加载预训练的VGG16模型
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 修改模型的最后一层，全连接层输出改为3个类别
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 3)

    # 选择损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 将模型移动到GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练模型
    model, losses, accuracies, val_losses, val_accuracies = train_model(model, criterion, optimizer, dataloaders,
                                                                        image_datasets, device, num_epochs=50)

    # 保存模型参数
    torch.save(model, 'modelSave/vgg16_voc2005.pth')

    # 将损失和准确率写入txt文件
    with open('modelSave/training_log_val.txt', 'w') as f:
        for epoch, (loss, acc) in enumerate(zip(losses, accuracies)):
            f.write(f'Train Epoch {epoch + 1}: Train Loss = {loss:.4f}, Train Accuracy = {acc:.4f}, '
                    f'Val Loss = {val_losses[epoch]:.4f}, Val accuracies = {val_accuracies[epoch]}\n')

    # 绘制损失和准确率的折线图
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Train Loss')
    plt.plot(range(1, len(losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), accuracies, label='Train Accuracy')
    plt.plot(range(1, len(losses) + 1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('modelSave/training_curve.png')
    #plt.show()
