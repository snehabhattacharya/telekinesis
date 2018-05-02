import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from dataloader import EEGDataset, my_collate, my_collate_two
from torch.utils.data import DataLoader
from autoencoder import Autoencoder

class CNN_net(nn.Module):
    def __init__(self, 
                 in_chans,
                 n_classes,
                 batch_size,
                 input_time_length,
                 kernel_size,
                 #final_conv_length,
                 n_filters_time=16,
                 n_filters_spat=10,
                 
                 filter_time_length=10,
                 pool_time_length=3,
                 pool_time_stride=3,
                 n_filters_2=50,
                 filter_length_2=10,
                 n_filters_3=100,
                 filter_length_3=10,
                 n_filters_4=200,
                 filter_length_4=10,
                 #first_nonlin=elu,
                 first_pool_mode='max',
                 #first_pool_nonlin=identity,
                 #later_nonlin=elu,
                 later_pool_mode='max',
                 #later_pool_nonlin=identity,
                 drop_prob=0.5,
                 double_time_convs=False,
                 split_first_layer=True,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 stride_before_pool=False):
        super(CNN_net, self).__init__()
        self.in_chans = in_chans
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.n_filters_time = n_filters_time
        self.input_time_length = input_time_length
        self.n_filters_spat = n_filters_spat
        self.n_classes = n_classes
        self.time_conv = nn.Sequential(
            nn.Conv2d(self.in_chans,self.n_filters_time, kernel_size=2,
            stride=2),
            #nn.ReLU(),
            nn.MaxPool2d(2))
        self.spat_conv = nn.Sequential(
            nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                     (1, 1),
                     stride=(1, 1),),
            nn.BatchNorm2d(self.n_filters_spat),
            nn.ReLU())

        self.fc = nn.Linear(self.n_filters_time, 
                       self.n_classes)

    def forward(self, x):
        x = x.float()
        N = x.size()[0]
        x = x[:19680]
        #x = x.view()
        #x = torch.to_numpy()
        x = x.view(1230, 64,4,4)
        #print x.size(), "new"
        x = self.time_conv(x)
        #print x.size(), "after time Conv2d"
        # x = self.spat_conv(x)
        x = x.view(x.size(0),-1)
        
        x = self.fc(x)
        #x = F.softmax
        return x
# torch.optim.Adam
model = CNN_net(in_chans=64, n_classes=3, input_time_length=1, batch_size=1, kernel_size=2).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()
fpath = "/home/snehabhattac/ubicompdata/pre_processed/"
trng = "/home/snehabhattac/ubicompdata/test/test/"
dset = EEGDataset(fpath)
tset = EEGDataset(trng)
loader = DataLoader(dset,num_workers=2, batch_size=2, collate_fn=my_collate_two)
loader_test = DataLoader(tset,num_workers=2, batch_size=2, collate_fn=my_collate_two)
model_2 = Autoencoder().cuda()
model_2.load_state_dict(torch.load("vae.pt"))
model_2.eval()
print model_2
# for data, target in loader:
    
num_epochs = 100
# criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(
# #     model.parameters(), lr=learning_rate, weight_decay=1e-5)
for epoch in range(num_epochs):
    for data, target in loader:
        data = Variable(data).cuda().float()
        data = model_2.forward(data)
        #target = Variable(target)
        output = model(data)
        _, preds = torch.max(output.data,1)
        #print output.size(), "output", preds.size(), "preds"
        target = target[:19680]
        new_target = torch.zeros(1230)
        for i in range(0,len(target),16):
            if not i > 1230:
                new_target[i] = target[i]
        # print new_target.size(), "new_target"
        # print preds.size()
        loss = criterion(output, Variable(new_target).cuda().long())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = 0
    for data, target in loader_test:
        data = Variable(data).cuda().float()
        data = model_2(data)
        output = model(data)
        _, preds = torch.max(output.data,1)
        target = target[:19680]
        new_target = torch.zeros(1230)
        for i in range(0,len(target),16):
            if not i > 1230:
                new_target[i] = target[i]
        accuracy += torch.sum(Variable(new_target) == preds.cpu().float())
        #print loss
    # ===================log========================
    print'epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data[0])
    print accuracy



    # def optimse(self,X,y,lr, num_batches):
    #     optimizer = optim.Adam(self.parameters(),lr=lr)
    #     criterion = nn.CrossEntropyLoss()
    #     # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #     X =X
    #     for epoch in range(1):
    #         for n in range(num_batches):
    #             X_batch = None
    #             y_batch = None
    #             indices = np.random.choice(np.arange(X.shape[0]), 200)
    #             X_batch = X[indices]
    #             y_batch = y[indices]
    #             print len(X_batch), "size"
    #             optimizer.zero_grad()
    #             output = self.forward(X_batch)
    #             y_batch = Variable(torch.from_numpy(y_batch)).long()
    #             loss = criterion(output, y_batch)
    #             optimizer.step()
    #             # print loss
    
