import torch.optim as optim
from torch.utils.data import DataLoader
from AlicePyTorch.face_rect_train.FaceDataset import FaceDataset
from AlicePyTorch.face_rect_train.YOLOv8MobileNet import YOLOv8MobileNet
import torch
import torch.nn as nn
import torchvision
from albumentations.pytorch import ToTensorV2
import albumentations as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv8MobileNet(num_classes=1).to(device)


def yolo_loss(pred, target):
    pred_boxes = pred[..., :4]
    pred_conf = pred[..., 4]

    target_boxes = target[..., :4]
    target_conf = target[..., 4]

    box_loss = nn.MSELoss()(pred_boxes, target_boxes)
    conf_loss = nn.BCEWithLogitsLoss()(pred_conf, target_conf)

    loss = box_loss + conf_loss
    return loss


def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 3, 1).contiguous()
            outputs = outputs.view(outputs.size(0), -1, 5)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs} Loss: {epoch_loss:.4f}')
    return model


transform = A.Compose([
    A.Resize(416, 416),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

model = YOLOv8MobileNet(num_classes=1).to(device)
criterion = yolo_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = FaceDataset(annotations_file='path/to/train_annotations.csv', img_dir='path/to/train_images',
                            transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

trained_model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=25)
