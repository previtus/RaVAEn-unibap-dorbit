# import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset

class LilModel(torch.nn.Module):
    def __init__(self, input_size = 128, output_size=1):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)
        self.criterion = nn.BCELoss()


    def forward(self, x):
        return torch.sigmoid(self.l1(x))
        return torch.relu(self.l1(x))

    def training_step(self, batch, batch_idx):
        # used only in pl
        x, y = batch

        # print("train step report:", x.shape, y.shape)

        y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        #optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=0.02)

class LilDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        # print(x.shape, y.shape)
        # print(x, y)

        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)

        return x, y

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
