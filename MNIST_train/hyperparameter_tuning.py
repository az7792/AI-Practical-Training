import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from xgboost import XGBClassifier

def load_mnist(path, labelfile, datafile):
    labels_path = os.path.join(path, labelfile)
    images_path = os.path.join(path, datafile)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

# 读训练集(60000张图片)
features0, labels0 = load_mnist("./MNIST", "train-labels-idx1-ubyte", "train-images-idx3-ubyte")
features = features0[0:60000, :]  # 仅使用前10000张图片
labels = labels0[0:60000]

# 读测试集(10000张图片)
testfeatures0, testlabels0 = load_mnist("./MNIST", "t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte")
test_features = testfeatures0[0:10000, :]  # 仅使用前10000张图片
test_labels = testlabels0[0:10000]

# 模型及超参数
model = XGBClassifier(random_state=42,tree_method = 'hist',device = 'cuda')
params = {
            "n_estimators": [150],
            "learning_rate": [0.1],
            "max_depth": [11],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        }

# 超参数搜索
grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy', verbose=3,return_train_score=True)
grid_search.fit(features, labels)

# 记录数据
records = []
for i, param in enumerate(grid_search.cv_results_['params']):
    record = {
        'model': 'XGBClassifier',
        'n_estimators': param['n_estimators'],
        'learning_rate': param['learning_rate'],
        'max_depth': param['max_depth'],
        'subsample': param['subsample'],
        'colsample_bytree': param['colsample_bytree'],
        'train_acc': grid_search.cv_results_['mean_train_score'][i],
        'val_acc': grid_search.cv_results_['mean_test_score'][i]
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv('MNIST训练记录.csv', mode='a', index=False, header=not os.path.exists('MNIST训练记录.csv'))

#最终模型及超参数的对应得分
pred_lable = grid_search.best_estimator_.predict(features)
print(f'acc: {accuracy_score(labels, pred_lable)}')
print(f'confusionmatrix:\n{confusion_matrix(labels, pred_lable)}')
print(f'classficationreport:\n{classification_report(labels, pred_lable, target_names=[str(i) for i in range(10)])}')
