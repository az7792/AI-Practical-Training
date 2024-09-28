import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('train_val_data.csv')

# 绘制训练损失和验证损失
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df.index, df['train_loss'], label='train_loss', marker='o')
plt.plot(df.index, df['val_loss'], label='val_loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('train_loss and val_loss')
plt.legend()

# 绘制训练准确率和验证准确率
plt.subplot(1, 2, 2)
plt.plot(df.index, df['train_acc'], label='train_acc', marker='o')
plt.plot(df.index, df['val_acc'], label='val_acc', marker='o')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.title('train_acc and val_acc')
plt.legend()

# 调整图表布局
plt.tight_layout()

# 保存图表为PNG文件
plt.savefig('training_results.png')

# 关闭图表
plt.close()
