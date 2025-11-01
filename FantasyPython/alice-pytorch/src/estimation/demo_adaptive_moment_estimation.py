import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成虚拟回归数据集
X, y = make_regression(n_samples=2000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转为tensor格式
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 标准化数据
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

# 构建神经网络模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 200
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 计算测试集上的损失
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

# 绘制图像
plt.figure(figsize=(14, 7))

# 图1：训练和测试损失随epoch的变化
plt.subplot(1, 2, 1)
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, train_losses, label='Training Loss', color='blue', linewidth=3)
plt.plot(epochs_range, test_losses, label='Testing Loss', color='red', linestyle='--', linewidth=3)
plt.fill_between(epochs_range, train_losses, test_losses, color='#FF1493', alpha=0.1)
plt.title('Training vs Testing Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)

# 图2：预测值与真实值的对比
plt.subplot(1, 2, 2)
with torch.no_grad():
    y_pred = model(X_test).numpy()
y_test_np = y_test.numpy()
plt.scatter(y_test_np, y_pred, label='Predicted vs Actual', color='green', s=40, alpha=0.7)
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], color='orange', linestyle='--', linewidth=3)
plt.title('Predicted vs Actual Values', fontsize=14)
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()