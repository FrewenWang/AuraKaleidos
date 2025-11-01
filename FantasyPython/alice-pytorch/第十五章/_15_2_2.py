import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()
        self.cnn1=nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.fc1=nn.Sequential(
            nn.Linear(8 * 120 * 120,500),
            nn.ReLU(inplace=True),
            nn.Linear(500,500),
            nn.ReLU(inplace=True),
            nn.Linear(500,128)
        )

    def forward_once(self, x):
        x = torch.unsqueeze(x,dim=1)
        out = self.cnn1(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out


    def forward(self, input1,input2):
        output1=self.forward_once(input1)
        output2=self.forward_once(input2)
        return output1,output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def forward(self, output1, output2, label):
        label = label.view(label.size()[0], )
        euclidean_distance = self.pdist(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

