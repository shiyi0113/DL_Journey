"""
手写数字识别
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
print(f'Using device:{device}')

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

# 模型 样本输入是1*28*28 输出是10
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16,120,kernel_size=5),nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120,84),nn.Tanh(),
            nn.Linear(84,10)
        )

    def forward(self,x):
        y = self.net(x)
        return y
model = CNN().to(device)

# 查看网络结构
X=torch.rand(size=(1,1,28,28)) # 输入尺寸(1,28,28)
for layer in CNN().net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

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
    best_state_dict = model.state_dict()
    Pref_accuracies = 0
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
        if accuracies>Pref_accuracies:
            best_state_dict = model.state_dict().copy()
            Pref_accuracies = accuracies
    torch.save(best_state_dict,os.path.join(MODEL_DIR,'cnn_LeNet5.pth'))    
    return train_losses,train_accuracies

def test_model():
    model = CNN().to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR,'cnn_LeNet5.pth'))
    model.load_state_dict(state_dict)

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
    multiprocessing.set_start_method('spawn')
    losses, accuracies = train_model()
    plot_metrics(losses, accuracies)
    test_model()