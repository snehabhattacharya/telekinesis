import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import EEGDataset, my_collate_two


num_epochs = 100
batch_size = 10
learning_rate = 1e-3


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 64))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
"""
fpath = "/home/snehabhattac/ubicompdata/pre_processed/"
dset = EEGDataset(fpath)
loader = DataLoader(dset,num_workers=2, batch_size=100, collate_fn=my_collate_two)

model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    
    for data, target in loader:
                
        data_ = Variable(data.float()).cuda()
        output = model(data_)
        loss = criterion(output, data_)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print loss.data[0]
    # ===================log========================
    # print 'epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data[0])
    print epoch, loss.cpu().data[0]
torch.save(model.state_dict(), "vae.pt")
"""
