'''
批处理
糖尿病患者数据集 diabetes_prediction_dataset
'''
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import multiprocessing
from tqdm import tqdm
from utils.visualization import plot_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 数据准备
class TensorDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = torch.tensor(X_data.values.astype(np.float32))
        self.Y = torch.tensor(Y_data.values.astype(np.float32)).reshape((-1, 1))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len
def prepare_data(filepath):
    df = pd.read_csv(filepath)
    # 独热编码
    categorical_cols = ['gender', 'smoking_history']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df.drop(columns=['diabetes'])
    Y = df['diabetes']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.3, 
        random_state=42, 
        stratify=Y  
    ) 
    
    # 特征缩放
    continuous_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()

    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = prepare_data(
    os.path.join(SCRIPT_DIR, os.pardir, 'data/diabetes_prediction_dataset.csv')
)
train_Data = TensorDataset(X_train, Y_train)
test_Data = TensorDataset(X_test, Y_test)

# 数据加载器
train_Loader = DataLoader(
    dataset=train_Data,
    shuffle=True,
    batch_size=256
)
test_Loader = DataLoader(
    dataset=test_Data,
    shuffle=False,
    batch_size=256
)
# 模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13,32),nn.ReLU(),
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
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

def train_model():
    epochs = 50
    train_losses = []
    train_accuracies = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        total = 0
        corrent = 0
        for i,(X,Y) in enumerate(tqdm(train_Loader,desc=f"Epoch{epoch+1}/{epochs}",ncols=100,dynamic_ncols=True),1):
            X,Y = X.to(device),Y.to(device)
            Pred = model(X)
            loss = loss_fn(Pred,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Pred[Pred>0.0] = 1.0
            Pred[Pred<=0.0] = 0.0
            running_loss += loss
            corrent += torch.sum(Pred==Y)
            total += Y.size(0)
        
        accuracies = corrent/total
        train_losses.append(running_loss.item())
        train_accuracies.append(accuracies.item())

    torch.save(model.state_dict(),os.path.join(MODEL_DIR,'dnn_batch.pth'))
    return train_losses, train_accuracies

def test_model():
    model = DNN().to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR, 'dnn_batch.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        corrent = 0
        total = 0
        for X,Y in test_Loader:
            X,Y = X.to(device),Y.to(device)
            Pred = model(X)
            Pred[Pred>0.0]=1.0
            Pred[Pred<=0.0]=0.0
            corrent += torch.sum(Pred==Y)
            total += Y.size(0)
    print(f'测试准确率：{corrent/total}')

if __name__ == '__main__':
    print(f'Using device:{device}')
    multiprocessing.set_start_method('spawn')
    losses, accuracies = train_model()
    plot_metrics(losses, accuracies)
    test_model()