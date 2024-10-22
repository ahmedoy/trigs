import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Taken from: https://github.com/UMBCvision/Universal-Litmus-Patterns/blob/master/CIFAR-10/utils/stn.py
'''

class STN(nn.Module):
    def __init__(self,nofchannels=3):
        super(STN,self).__init__()         
        self.nofchannels=nofchannels
        self.localization= nn.Sequential(nn.Conv2d(3,16,kernel_size=7),
                           nn.MaxPool2d(2,stride=2),
                           nn.ReLU(True), 
                           nn.Conv2d(16,32,kernel_size=5),
                           nn.MaxPool2d(2,stride=2),
                           nn.ReLU(True),
                           nn.Conv2d(32,64,kernel_size=3),
                           nn.MaxPool2d(2,stride=2),
                           nn.ReLU(True))
        
        self.fc_loc=  nn.Sequential(nn.Linear(64,128),
                      nn.ReLU(True),
                      nn.Linear(128,64),
                      nn.ReLU(True),
                      nn.Linear(64,6))
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1,0,0,0,1,0]))

        
    def forward(self,x):
        xs=self.localization(x)
        xs=xs.view(-1,64)
        theta=self.fc_loc(xs)
        theta=theta.view(-1,2,3)
        grid=F.affine_grid(theta,x.size())
        x=F.grid_sample(x,grid)
        return x


'''
Adapted from: https://github.com/UMBCvision/Universal-Litmus-Patterns/blob/master/CIFAR-10/utils/model.py
Modified init_num_filters from 32 to 64
Modified inter_fc_dim from 128 to 384
Removed STN usage. Have to check whether the models in the dataset use it.
'''

class VGG_MOD(nn.Module):
    """ MNIST Encoder from Original Paper's Keras based Implementation.
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=64, lrelu_slope=0.2, inter_fc_dim=384, num_classes=10,nofchannels=3,use_stn=True):
        super(VGG_MOD, self).__init__()
        self.use_stn=use_stn
        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.nofclasses_ = num_classes
        if self.use_stn:
            self.stn = STN()

        self.features = nn.Sequential(
            nn.Conv2d(nofchannels, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=5,padding=2),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),

            nn.MaxPool2d(2,2),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.init_num_filters_ *4*4, self.inter_fc_dim_),
            nn.BatchNorm1d(self.inter_fc_dim_),
            nn.ReLU(True),
            nn.Dropout(p=.2),

            nn.Linear(self.inter_fc_dim_, int(self.inter_fc_dim_/2)),
            nn.BatchNorm1d(int(self.inter_fc_dim_/2)),
            nn.ReLU(True),
            nn.Dropout(p=.2),

            nn.Linear(int(self.inter_fc_dim_/2), self.nofclasses_)
        )


    def forward(self, x):
        if self.use_stn:
            x = self.stn(x)
        x = self.features(x)
        x = x.reshape(-1, self.init_num_filters_ *4*4)
        x = self.fc(x)
        return x