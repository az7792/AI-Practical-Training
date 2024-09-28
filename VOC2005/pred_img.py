import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import xgboost as xgb
import argparse

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='模型预测工具')
    parser.add_argument('image_path', type=str, help='要预测的图片路径')
    parser.add_argument('--Mtype', type=int, default=2, choices=[0, 1, 2], help='模型类型: 0-ResNet50, 1-VGG16, 2-XGBoost混合')
    return parser.parse_args()

class_names = ['bicycle','car','motorcycle']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

xgb_model = xgb.XGBClassifier()  # XGB
xgb_model.load_model('xgboost_model.json')

# 定义预测函数
def predict_single_image(model1, model2, image, device, Mtype=2):
    model1.eval()
    model2.eval()

    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # 添加 batch 维度

        output1 = torch.softmax(model1(image), dim=1)
        output2 = torch.softmax(model2(image), dim=1)
        
        if Mtype == 0:
            predicted_labels = torch.argmax(output1, dim=1).cpu().item()
        elif Mtype == 1:
            predicted_labels = torch.argmax(output2, dim=1).cpu().item()
        else:
            features = torch.cat((output1, output2), dim=1).cpu().numpy()
            predicted_labels = xgb_model.predict(features)[0]

    return predicted_labels

def main():
    args = parse_args()

    # 加载图片
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"未找到图片文件 '{args.image_path}'")
    
    image = Image.open(args.image_path).convert('RGB')
    image = transform(image)

    # 预测类别
    if args.Mtype == 0:
        label = predict_single_image(model1, model2, image, device, Mtype=0)
        print(f'预测类别 (ResNet50): {class_names[label]}')
    elif args.Mtype == 1:
        label = predict_single_image(model1, model2, image, device, Mtype=1)
        print(f'预测类别 (VGG16): {class_names[label]}')
    else:
        label = predict_single_image(model1, model2, image, device, Mtype=2)
        print(f'预测类别 (XGBoost混合): {class_names[label]}')

if __name__ == '__main__':
    main()
