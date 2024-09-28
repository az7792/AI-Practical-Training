import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision import datasets
import xgboost as xgb

# 优先使用cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
model1_save_path = 'best_model.pth' #resnet50
if os.path.exists(model1_save_path):
    model1 = torch.load(model1_save_path, map_location=device)
else:
    raise FileNotFoundError(f"加载模型失败: 未找到文件 '{model1_save_path}'.")

model2_save_path = 'modelSave/vgg16_voc2005.pth' #vgg16
if os.path.exists(model2_save_path):
    model2 = torch.load(model2_save_path, map_location=device)
else:
    raise FileNotFoundError(f"加载模型失败: 未找到文件 '{model2_save_path}'.")

xgb_model = xgb.XGBClassifier() #XGB
xgb_model.load_model('xgboost_model.json')


def test_models(model1, model2, data_loader, device, Mtype=2):
    model1.eval()
    model2.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            output1 = torch.softmax(model1(images), dim=1)
            output2 = torch.softmax(model2(images), dim=1)
            if Mtype == 0:
                predicted_labels = torch.argmax(output1, dim=1).cpu().numpy()
            elif Mtype == 1:
                predicted_labels = torch.argmax(output2, dim=1).cpu().numpy()
            else:
                features = torch.cat((output1, output2), dim=1).cpu().numpy()
                predicted_labels = xgb_model.predict(features)

            y_pred.extend(predicted_labels)
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

print(f'ResNet50 : {test_models(model1,model2,test_loader,device,Mtype = 0)*100 :.4f}%') #resnet50
print(f'VGG16    : {test_models(model1,model2,test_loader,device,Mtype = 1)*100 :.4f}%') #VGG16
print(f'XGB混合  : {test_models(model1,model2,test_loader,device,Mtype = 2)*100 :.4f}%') #XGB混合