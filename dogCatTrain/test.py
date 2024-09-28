import torch
import os
import random
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 加载模型
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model = model.to(device)

model_save_path = 'best_model.pth'
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print('Model loaded successfully.')
else:
    raise ValueError(f'Model file not found at {model_save_path}')

model.eval()

# 图片目录
test_dir = 'test1'
image_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
random_images = random.sample(image_files, 20)

# 类别名称
class_names = ['cat', 'dog']

# 设置画布大小
fig, axs = plt.subplots(4, 5, figsize=(20, 8))
axs = axs.flatten()

# 预测和显示结果
for i, image_name in enumerate(random_images):
    img_path = os.path.join(test_dir, image_name)
    image = Image.open(img_path).convert("RGB")
    image_tensor = data_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        cat_prob, dog_prob = probs[0]  # 获取猫和狗的概率

    # 显示图片和预测结果
    axs[i].imshow(image)
    axs[i].set_title(f'Cat: {cat_prob.item()*100:.2f}%, Dog: {dog_prob.item()*100:.2f}%')
    axs[i].axis('off')

# 保存组合图片
plt.tight_layout()
save_path = 'pred_combined.png'
plt.savefig(save_path)
plt.close()
