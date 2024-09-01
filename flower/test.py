import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image

flower_name = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 加载预训练的 ResNet 模型
model = models.resnet18(weights='DEFAULT')

# 替换最后一层全连接层以适应鲜花分类
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)

model = model.to(device)

# 尝试加载已保存的模型参数
model_save_path = 'best_model.pth'
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print('Model parameters loaded from previous training.')
else:
    print('No saved model found. Exiting.')
    exit()

# 测试图像目录
test_img_dir = 'test_img'

# 加载测试图像
image_files = [f for f in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, f))]

model.eval()
with torch.no_grad():
    for image_file in image_files:
        img_path = os.path.join(test_img_dir, image_file)
        image = Image.open(img_path).convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0).to(device)  # 添加 batch dimension

        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        print(f'{image_file}: {flower_name[preds.item()]}')

