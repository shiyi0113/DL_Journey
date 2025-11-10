'''
糖尿病患者数据集 diabetes_prediction_dataset
'''
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import multiprocessing
from utils.visualization import plot_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Using device:{device}')

# 数据准备
df = pd.read_csv(os.path.join(SCRIPT_DIR, os.pardir, 'data/diabetes_prediction_dataset.csv'))
# 对分类列进行独热编码
gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
smoking_history_dummies = pd.get_dummies(df['smoking_history'],prefix='smoking_history')
df = pd.concat([df.drop(['gender', 'smoking_history'],axis=1), gender_dummies, smoking_history_dummies], axis=1)

arr = df.values
arr =arr.astype(np.float32)
ts = torch.tensor(arr).to(device)

train_size = int(ts.size(0)*0.7)
test_size = int(ts.size(0)*0.3)
ts = ts[torch.randperm(ts.size(0)),:]
train_Data = ts[:train_size,:]
test_Data = ts[train_size:,:]

# 模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(15,32),nn.ReLU(),
            nn.Linear(32,8),nn.ReLU(),
            nn.Linear(8,4),nn.ReLU(),
            nn.Linear(4,1)
        )
    def forward(self,x):
        y = self.net(x)
        return y
model = DNN().to(device)

# 损失函数+优化器
loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)

def train_model():
    epochs = 1000
    train_losses = []
    train_accuracies = []
    X = train_Data[:,:-1]
    Y = train_Data[:,-1].reshape((-1,1))
    model.train()
    for epoch in range(epochs):
        Pred = model(X)
        loss = loss_fn(Pred,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Pred[Pred>0.5]=1
        Pred[Pred<=0.5]=0
        corrent = torch.sum(Pred==Y)
        total = Y.size(0)
        accuracies = corrent.float()/total
        train_losses.append(loss.item())
        train_accuracies.append(accuracies.item())

    torch.save(model.state_dict(),os.path.join(MODEL_DIR,'dnn_dpd.pth'))
    return train_losses, train_accuracies

def test_model():
    model = DNN().to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR, 'dnn_dpd.pth'))
    model.load_state_dict(state_dict)
    X = test_Data[:,:-1]
    Y = test_Data[:,-1].reshape((-1,1))
    model.eval()
    with torch.no_grad():
        Pred = model(X)
        Pred[Pred>0.5]=1
        Pred[Pred<=0.5]=0
        corrent = torch.sum(Pred==Y)
        total = Y.size(0)
    print(f'测试准确率：{corrent.float()/total}')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    losses, accuracies = train_model()
    plot_metrics(losses, accuracies)
    test_model()