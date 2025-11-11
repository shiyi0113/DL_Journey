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
import torch.nn.functional as F
from utils.visualization import plot_metrics,visualize_images

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 数据准备
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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

# 模型 
# --- 基础的卷积块 ---
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y
# --- Inception块 ---
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # Branch 1: 1x1 卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # Branch 2: 1x1 卷积 (降维) -> 3x3 卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1) 
        )
        # Branch 3: 1x1 卷积 (降维) -> 5x5 卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2) #
        )
        # Branch 4: 3x3 Max Pooling -> 1x1 卷积 (降维/投影)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, 1)

# --- 辅助分类器类 (仅在训练时使用) ---
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.aux = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            BasicConv2d(in_channels, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        y = self.aux(x)
        return y

# --- 主模型结构 ---
# 输入数据为1x224x224的手写数字识别数据集，输出为10
class GoogLeNet(nn.Module):  
    def __init__(self, num_classes=10, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits # 分类器启动开关
        # --- 起始阶段 ---
        self.stem = nn.Sequential(
            BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # --- Inception Modules 堆叠 (Body) ---
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)   # Output: 256 x 28 x 28
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64) # Output: 480 x 28 x 28
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)  # Output: 512 x 14 x 14
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64) # Output: 512 x 14 x 14 
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64) # Output: 512 x 14 x 14
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64) # Output: 528 x 14 x 14
        if aux_logits:
            self.aux2 = InceptionAux(528, num_classes)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128) # Output: 832 x 14 x 14
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128) # Output: 832 x 7 x 7
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128) # Output: 1024 x 7 x 7

        # --- 顶层分类器 ---
        self.head = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        # Stem
        x = self.stem(x)
        # Inception 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        # Inception 4 + Aux Classifier
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.aux_logits else None # 训练时使用辅助分类器 1
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.aux_logits else None # 训练时使用辅助分类器 2
        x = self.inception4e(x)
        x = self.maxpool4(x)
        # Inception 5
        x = self.inception5a(x)
        x = self.inception5b(x)
        # Head
        x = self.head(x)

        # 训练时返回主输出和两个辅助输出,推理时只返回主输出
        if self.aux_logits:
            return x, aux2, aux1
        
        return x


model = GoogLeNet(aux_logits=True).to(device)

# 损失函数 + 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.005,
)
def train_model():
    epochs = 2
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
            Pred,aux2,aux1 = model(X)
            loss_main = loss_fn(Pred,Y)
            loss_aux1 = loss_fn(aux1,Y)
            loss_aux2 = loss_fn(aux2,Y)
            loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2

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
    torch.save(best_state_dict,os.path.join(MODEL_DIR,'cnn_GoogLeNet.pth'))    
    return train_losses,train_accuracies

def test_model():
    model = GoogLeNet(aux_logits=True).to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR,'cnn_GoogLeNet.pth'))
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
            Pred,_,_= model(X)
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