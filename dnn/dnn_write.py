"""
手写数字识别 - DNN
"""
import os
import torch
import torch.nn as nn
import multiprocessing
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from utils.visualization import plot_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307,0.3081)  # 手写数字识别数据集转换为标准正态分布
])

train_Data = datasets.MNIST(
    root=os.path.join(SCRIPT_DIR, os.pardir, 'data'),  # 下载数据集的路径
    train=True,                   # 下载训练集
    download=True,                # 如果本地没有就下载
    transform=transform           # 转换参数
)

test_Data = datasets.MNIST(
    root=os.path.join(SCRIPT_DIR, os.pardir, 'data'),  # 下载数据集的路径
    train=False,                  # 下载测试集
    download=True,                # 如果本地没有就下载
    transform=transform           # 转换参数
)
# 数据加载器
train_loader = DataLoader(train_Data,shuffle=True,batch_size=256)
test_loader = DataLoader(test_Data,shuffle=False,batch_size=256)

# 模型 样本输入是28*28 输出是10
class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),nn.ReLU(),
            nn.Linear(512,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,10),
        )
    def forward(self,x):
        y = self.net(x)
        return y
model = DNN().to(device)

# 损失函数 + 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
    momentum=0.5
)

def train_model():
    epochs = 50
    train_losses = []
    train_accuracies = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        corrent = 0
        total = 0
        for i,(X,Y) in enumerate(tqdm(train_loader,desc=f"Epoch{epoch+1}/{epochs}",ncols=100,dynamic_ncols=True),1):
            X,Y = X.to(device),Y.to(device)
            Pred = model(X)
            loss = loss_fn(Pred,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Pred_index = torch.argmax(Pred.data,dim=1)
            corrent += torch.sum(Pred_index == Y)
            total+=Y.size(0)
            running_loss+=loss
        accuracies = corrent/total
        train_losses.append(running_loss.item())
        train_accuracies.append(accuracies.item())
    torch.save(model.state_dict(),os.path.join(MODEL_DIR,'dnn_write.pth'))    
    return train_losses,train_accuracies

def test_model():
    model = DNN().to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR,'dnn_write.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        corrent = 0
        total = 0
        for X,Y in test_loader:
            X,Y = X.to(device),Y.to(device)
            Pred = model(X)
            Pred_index = torch.argmax(Pred.data,dim=1)
            corrent += torch.sum(Pred_index == Y)
            total += Y.size(0)
    print(f"测试准确率：{corrent/total}")

if __name__ == '__main__':
    print(f'Using device:{device}')
    multiprocessing.set_start_method('spawn')
    losses, accuracies = train_model()
    plot_metrics(losses, accuracies)
    test_model()