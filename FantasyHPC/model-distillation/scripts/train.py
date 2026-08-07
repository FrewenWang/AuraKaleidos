
from pathlib import Path
import sys

from torch import optim, nn
from torch.utils.data import DataLoader
from torchaudio import datasets
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import TeacherModel

#  数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#  加载  MNIST  数据集
data_dir = PROJECT_ROOT / "data"
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


def train_teacher(model, train_loader, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch   [{epoch + 1}/{epochs}],   Loss:   {total_loss / len(train_loader):.4f}")


#   初始化并训练教师模型
teacher_model = TeacherModel()
train_teacher(teacher_model, train_loader)
