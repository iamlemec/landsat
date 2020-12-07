from os import path
from glob import glob
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.models import resnet

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert()

def gen_path(i):
    tag = f'{i:07d}'
    sub = tag[:4]
    return f'{sub}/{tag}'

class ImageDataset(data.Dataset):
    def __init__(self, sources, metadata, transform=None, ext='jpg'):
        self.ids = metadata.index.tolist()
        self.data = torch.tensor(metadata.to_numpy(), dtype=torch.float)

        self.sources = sources
        self.paths = [f'{gen_path(i)}.{ext}' for i in self.ids]

        self.transform = transform

    def __getitem__(self, idx):
        path, data = self.paths[idx], self.data[idx]
        imgs = torch.cat([
            self.transform(pil_loader(f'{src}/{path}')) for src in self.sources
        ])
        return imgs, data
    
    def __len__(self):
        return len(self.ids)

def make_resnet(nchan=1, nclass=1):
    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=nclass)
    model.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def loss_function(y, yh, scale=100):
    return scale*F.mse_loss(yh, y)

def train(model, loader, optim, epoch):
    log_interval = len(loader.dataset) // loader.batch_size // 5
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(loader):
        img, stat = data[0].to('cuda'), data[1].to('cuda')
        optim.zero_grad()
        pred = model(img).squeeze(1)
        loss = loss_function(pred, stat)
        loss.backward()
        train_loss += loss.item()
        optim.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{:06d}/{} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(stat), len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item() / len(stat)))

def test(model, loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            img, stat = data[0].to('cuda'), data[1].to('cuda')
            pred = model(img).squeeze(1)
            test_loss += loss_function(pred, stat).item()
    avg_loss = test_loss / len(loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f}')

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        stat_list = []
        pred_list = []
        for batch_idx, data in enumerate(loader):
            img, stat = data[0].to('cuda'), data[1].to('cuda')
            pred = model(img).squeeze(1)
            stat_list.append(stat)
            pred_list.append(pred)
        stat_list = torch.cat(stat_list)
        pred_list = torch.cat(pred_list)
    return stat_list, pred_list
