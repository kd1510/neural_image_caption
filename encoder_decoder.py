import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b0


class Encoder(nn.Module):
    def __init__(self, D):
        super().__init__()

        self.convnet = efficientnet_b0(pretrained=True)
        self.convnet.eval()
        self.convnet.classifier[1] = nn.Linear(1280, D)
        
        for param in self.convnet.parameters():
            param.requires_grad = False
        self.convnet.classifier[1].requires_grad = True

        self.params = self.convnet.classifier[1].parameters()

    def forward(self, img):
        x = self.convnet(img)
        return x


class Decoder(nn.Module):
    def __init__(self, V, D, H):
        super().__init__()
        self.V = V
        self.D = D

        self.embed = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, H, batch_first=False)
        self.h1 = nn.Linear(H, V)

    def forward(self, x):
        if x.shape[1] == self.D:
            emb = x
        else:
            emb = self.embed(x)
        
        x, _ = self.lstm(emb)
        x = self.h1(x)
        return x