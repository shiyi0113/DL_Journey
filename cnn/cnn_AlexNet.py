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
from utils.visualization import plot_metrics,visualize_images

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 数据准备
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
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
train_loader = DataLoader(train_Data,shuffle=True,batch_size=128,num_workers=2,pin_memory=True)
test_loader = DataLoader(test_Data,shuffle=False,batch_size=128,num_workers=2,pin_memory=True)

# 模型 样本输入是1*227*227 输出是10

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 定义卷积层和池化层
        self.features = nn.Sequential(
            BasicConv2d(1,64,kernel_size = 11,stride = 4,padding = 2), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64,192,kernel_size = 5,padding = 2),  
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(192,384,kernel_size = 3,padding = 1),
            BasicConv2d(384,256,kernel_size = 3,padding = 1),
            BasicConv2d(256,256,kernel_size = 3,padding = 1),
            nn.MaxPool2d(kernel_size=3, stride=2)  
        )
        # 定义展平层
        self.flatten = nn.Flatten()
        # 定义分类层
        self.classifier = nn.Sequential(
            nn.Dropout(),  
            nn.Linear(256 * 6 * 6, 4096),  nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 10) 
        )

    def forward(self, x):
        x = self.features(x)       # 卷积层
        x = self.flatten(x)        # 展平层
        x = self.classifier(x)     # 全连接层
        return x
model = AlexNet().to(device)

# 损失函数 + 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.001,
)

def train_model():
    epochs = 5
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
    torch.save(best_state_dict,os.path.join(MODEL_DIR,'cnn_AlexNet.pth'))    
    return train_losses,train_accuracies

def test_model():
    model = AlexNet().to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR,'cnn_AlexNet.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    predictions = []  # 用于保存预测结果
    images = []       # 用于保存输入图像
    labels = []       # 用于保存实际标签

    with torch.no_grad():
        total = 0
        corrent = 0
        for X, Y in test_loader:
            X,Y = X.to(device),Y.to(device)  
            Pred = model(X)
            Pred_index = torch.argmax(Pred.data,dim=1)
            total += Y.size(0)
            corrent += torch.sum(Pred_index == Y)
            predictions.extend(Pred_index.cpu().numpy())  # 保存预测结果到 CPU 上
            images.extend(X.cpu().numpy())     # 保存输入图像到 CPU 上
            labels.extend(Y.cpu().numpy())     # 保存真实标签到 CPU 上
    print(f"测试准确率：{corrent/total}")
    visualize_images(images[:6], labels[:6], predictions[:6])

if __name__ == '__main__':
    print(f'Using device:{device}')
    multiprocessing.set_start_method('spawn')
    losses, accuracies = train_model()
    plot_metrics(losses, accuracies)
    test_model()