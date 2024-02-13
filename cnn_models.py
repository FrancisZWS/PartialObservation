from pickle import encode_long
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda")
'''
modified to fit dataset size
'''
NUM_CLASSES = 10
# freqpts = 64 #freq pts per band


class decouple_cnn(nn.Module):
# Feb-Mar 2023 version, in-progress
# need to provide: 
# cfg: neurons per layer, needs to be given, [40, 8nch,8nch,8nch]
# ty_chs=True/False for 10/20 channels, 
# nch is the num of channels fo be detected for this DNN
# pay attention to ty_chs and related pooling config
    def __init__(self, nch=5, cfg=None, ty_chs=False, init_weights=True):
        # super(decouple_cnn, self).__init__()
        super().__init__()
        self.nch = nch
        if cfg is None:
            cfg = [40, 8*nch, 8*nch, 8*nch]
        self.cfg = cfg
        self.ty_chs = ty_chs
        self.ks = 4 #final conv layer output kernal size 2x2=4
        self.features = self.make_layers(cfg) # The number of groups: dataset( # of classes )
        self.fc = nn.Linear(self.ks*cfg[-1], nch)#------------------------------------ 
        if init_weights:
            self._initialize_weights_mod()

    def make_layers(self, cfg): #make layers for normal_cnn
        nch = self.nch
        layers = [ 
                nn.Conv2d(1, cfg[0], kernel_size=3, padding = 1 ),
                    nn.BatchNorm2d(cfg[0]),
                    nn.ReLU(inplace=True),
                nn.Conv2d(cfg[0], cfg[1], kernel_size=3, padding = 1), # default: decouple from layer 2
                    nn.BatchNorm2d(cfg[1]),
                    nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(4,1) ),
                nn.Conv2d(cfg[1], cfg[2], kernel_size=3, groups=self.nch, padding = 1), #if decouple from layer3, cancel "group" here
                    nn.BatchNorm2d(cfg[2]),
                    nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2,1) ),
                nn.Conv2d(cfg[2], cfg[3], kernel_size=3, groups=self.nch, padding = 1),
                    nn.BatchNorm2d(cfg[3]),
                    nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=(4,5) ),
                    ]

        if self.ty_chs: #if 20_channel input dim, change one Pooling layer
            layers[-5] = nn.MaxPool2d(kernel_size=(2,2) ) #W batchnorm
            # layers[-4] = nn.MaxPool2d(kernel_size=(2,2) ) # W/O batchnorm

        return nn.Sequential(*layers)


    def forward(self, x): ## 
        x = self.features(x)
        x = x.view(x.size(0), -1) # 128 batch size 
        #print('batch size', x.size())
        Lx = x.size(1)// self.nch ##  
        #x.size(1) ?? corrected
        y = torch.empty(x.size(0), self.nch).to(device)#  # empy or zero?
        
        for i in range(self.nch):
            xin = torch.zeros_like(x)
            xin[:, Lx*i : Lx*(i+1)] = x[:, Lx*i : Lx*(i+1)] # Mask input data cols
            y_p = self.fc( xin ) # y_p is a list of logits 
            y[:, i] = 1*y_p[:, i] # set cols of output          
        #print('len y list',len(y) )
        return y

    def _initialize_weights_mod(self):  # May not be used
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # if conv layer
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d): #if batchnorm layer
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): #if linear layer
                m.weight.data.zero_()
                
                Lx = m.weight.data.size(1)//self.nch                
                for i in range(self.nch):
                    # m.weight.data[i][i*Lx : (i+1)*Lx].normal_(0, 0.01) #previous method
                    m.weight.data[i][i*Lx: (i+1)*Lx].fill_(0.5) # same method as normal_cnn
                #m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()   

class decouple_cnn_mod(nn.Module):
# Feb-Mar 2023 version, use ModuleList for FC 
# need to provide: 
# cfg: neurons per layer, needs to be given, [40, 8nch,8nch,8nch]
# ty_chs=True/False for 10/20 channels, 
# nch is the num of channels fo be detected for this DNN
# Need different Merge/Split function
    def __init__(self, nch=5, cfg=None, ty_chs=False, init_weights=True):
        super().__init__()
        self.nch = nch
        if cfg is None:
            cfg = [40, 8*nch, 8*nch, 8*nch]
        self.cfg = cfg
        self.ty_chs = ty_chs
        self.ks = 4 #final conv layer output kernal size 2x2=4
        self.features = self.make_layers(cfg) # The number of groups: dataset( # of classes )
        self.fc = nn.ModuleList([nn.Linear(int(self.ks*cfg[-1]//nch),1) for f in range(nch)])
        if init_weights:
            self._initialize_weights_mod()

    def make_layers(self, cfg): #make layers for normal_cnn
        nch = self.nch
        layers = [ 
                nn.Conv2d(1, cfg[0], kernel_size=3, padding = 1 ),
                    nn.BatchNorm2d(cfg[0]),
                    nn.ReLU(inplace=True),
                nn.Conv2d(cfg[0], cfg[1], kernel_size=3, padding = 1), # default: decouple from layer 2
                    nn.BatchNorm2d(cfg[1]),
                    nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(4,1) ),
                nn.Conv2d(cfg[1], cfg[2], kernel_size=3, groups=self.nch, padding = 1), #if decouple from layer3, cancel "group" here
                    nn.BatchNorm2d(cfg[2]),
                    nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2,1) ),
                nn.Conv2d(cfg[2], cfg[3], kernel_size=3, groups=self.nch, padding = 1),
                    nn.BatchNorm2d(cfg[3]),
                    nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=(4,5) ),
                    ]

        if self.ty_chs: #if 20_channel input dim, change one Pooling layer
            layers[-5] = nn.MaxPool2d(kernel_size=(2,2) ) #W batchnorm
            # layers[-4] = nn.MaxPool2d(kernel_size=(2,2) ) # W/O batchnorm
        return nn.Sequential(*layers)

    def forward(self, x): ## 
        x = self.features(x)
        x = x.view(x.size(0), -1) # 128 batch size 
        Lx = int(x.size(1)/self.nch) ## 52 length of each critical pth output, x.size(1) ????????
        y = torch.empty(x.size(0), self.nch).to(device)#  # empy or zero?
        for i in range(self.nch):
            xin = x[:, Lx*i:Lx*(i+1)]
            # print('Lx: ', Lx)
            yi=self.fc[i](xin)
            y[:,i] = torch.squeeze(yi)
        return y

    def _initialize_weights_mod(self):  # May not be used
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # if conv layer
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d): #if batchnorm layer
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): #if linear layer
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()


class standalone_cnn(nn.Module):
#Feb-Mar 2023 version, in progress, try hard-code non-square pooling
# need to provide: 
# cfg: neurons per layer, 
# ty_chs=True/False for 10/20 channels, 
# nch is the num of channels fo be detected for this DNN

# For FL applications, set nch to maximum, set cfg and ty_ch accordingly  
    def __init__(self, nch=5, cfg=None, ty_chs=False, init_weights=True):
        # super(normal_cnn, self).__init__()
        super().__init__()
        self.nch = nch # num of classes
        if cfg is None:
            cfg = [40, 8*nch, 8*nch, 8*nch]
        self.ty_chs = ty_chs # it is 20-channel input or not
        self.features = self.make_layers(cfg) # The number of groups: dataset( # of classes )
        self.fc = nn.Linear(4*cfg[-1], nch) # get 2x2 after final layer

        if init_weights:
            self._initialize_weights_mod()

    def make_layers(self, cfg): #make layers for normal_cnn
        nch = self.nch
        layers = [ # 10_channel version by default
                nn.Conv2d(1, cfg[0], kernel_size=3, padding = 1 ),
                    nn.BatchNorm2d(cfg[0]),
                    nn.ReLU(inplace=True),
                nn.Conv2d(cfg[0], cfg[1], kernel_size=3, padding = 1),
                    nn.BatchNorm2d(cfg[1]),
                    nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(4,1) ),
                nn.Conv2d(cfg[1], cfg[2], kernel_size=3, padding = 1),
                    nn.BatchNorm2d(cfg[2]),
                    nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2,1) ),
                nn.Conv2d(cfg[2], cfg[3], kernel_size=3, padding = 1),
                    nn.BatchNorm2d(cfg[3]),
                    nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=(4,5) ),
                    ]

        if self.ty_chs : #if 20_channel input dim, change one Pooling layer
            layers[-5] = nn.MaxPool2d(kernel_size=(2,2) )

        return nn.Sequential(*layers)


    def forward(self, x): ## 
        x = self.features(x)
        x = x.view(x.size(0), -1) # 128 batch size 
        #print('batch size', x.size())
        y = self.fc(x)
        return y
                
    def _initialize_weights_mod(self):  # May not be used
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # if conv layer
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d): #if batchnorm layer
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): #if linear layer
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()



class DeepSense(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSense, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(32*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x

class DeepSenseHalf(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSenseHalf, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(16, 16, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(16*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x


class DeepSenseQuarter(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSenseQuarter, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(4, 8, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(8, 8, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(8*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x

class DeepSenseEighth(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSenseEighth, self).__init__()
        self.conv1 = nn.Conv1d(1, 2, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(2, 2, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(2, 4, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(4, 4, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(4*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x

