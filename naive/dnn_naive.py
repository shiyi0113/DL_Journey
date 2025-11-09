'''三分类问题'''
import os
import torch
import torch.nn as nn
import multiprocessing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, os.pardir, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Using device:{device}')

# 数据准备
x1 = torch.rand(10000,1)
x2 = torch.rand(10000,1)
x3 = torch.rand(10000,1)

y1 = ((x1+x2+x3)<1).float()
y2 = (((x1+x2+x3)>=1)&((x1+x2+x3)<2)).float()
y3 = (x1+x2+x3>=2).float()

Data = torch.cat([x1,x2,x3,y1,y2,y3],axis=1)
Data = Data.to(device)

train_size = int(len(Data)*0.7)
test_size = int(len(Data)*0.3)
Data = Data[torch.randperm(Data.size(0)),:] 
train_Data = Data[:train_size,:]
test_Data = Data[train_size:,:]

# 模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(3,5),nn.ReLU(),
            nn.Linear(5,5),nn.ReLU(),
            nn.Linear(5,5),nn.ReLU(),
            nn.Linear(5,3)
        )
    def forward(self,x):
        y = self.net(x)
        return y
model = DNN().to(device)

# 损失函数 + 优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

def train_model():
    '''训练'''
    epochs = 1000
    X = train_Data[:,:3]
    Y = train_Data[:,-3:]
    for epoch in range(epochs):
        model.train()
        Pred = model(X)
        loss = loss_fn(Pred,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'dnn_naive.pth'))

def test_model():
    '''测试'''
    model = DNN().to(device)
    state_dict = torch.load(os.path.join(MODEL_DIR, 'dnn_naive.pth'))
    model.load_state_dict(state_dict)
    model.eval()
    X = test_Data[:,:3]
    Y = test_Data[:,-3:]
    with torch.no_grad():
        Pred = model(X)
        Pred_idx = torch.argmax(Pred, dim=1)
        Pred_oh = torch.zeros_like(Pred)
        Pred_oh.scatter_(1, Pred_idx.unsqueeze(1), 1)
        correct = torch.sum((Pred_oh == Y).all(1))
        total = Y.size(0)
        print(f"测试准确率：{correct/total}")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    train_model()
    test_model()
