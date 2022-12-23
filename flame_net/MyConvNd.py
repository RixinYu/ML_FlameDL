
import torch
import torch.nn as nn


#------------------------------
#------------------------------------------------------------------------------
class MyConvNd(nn.Module):  # keep strid ==1
    def __init__(self, nDIM, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular',
                 bias=True, bRelu=True, bNorm=False, type='Conv'):
        super(MyConvNd, self).__init__()
        self.nDIM = nDIM

        self.type = type
        if bNorm == True:    bias = False  # when batchnorm is on, bias becomes redundent

        if 'r' == type.casefold()[0]:  # e.g 'Residual' , 'Residual1', 'Resid3'
            numRepeat = int(type[-1]) if type[-1].isdigit() else 1  # the repeat time is given by the last character (digit) of the given type
            self.net = ResidualBlockNd(nDIM, numRepeat, in_channels, out_channels, kernel_size, stride, padding,
                                       padding_mode, bias, bRelu, bNorm)
        else:

            layers = []
            # ----------------------------------------------------------------------
            if 'c' in type.casefold()[0]: # standard CNN
                # default parameter setting for learning flame stability
                if kernel_size == 1:
                    padding = 0
                elif kernel_size == 3:
                    padding = 1
                    padding_mode = 'circular'
                layers.append(nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                              padding_mode=padding_mode, bias=bias))
            elif 't' == type.casefold()[0]:
                layers.append(
                    nn_ConvTransposeNd(nDIM)(in_channels, out_channels, kernel_size, stride=stride, bias=bias))
            elif 'i' in type.casefold()[0]:
                layers.append(InceptionND_v3(nDIM, in_channels, out_channels))
            else:
                raise ValueError(type + ' is not found: MyConv1d')
            # ----------------------------------------------------------------------

            if bRelu == True:     layers.append(nn.ReLU())
            if bNorm == True:     layers.append(nn_BatchNormNd(nDIM)(out_channels))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#------------------------------


def nn_ConvNd(nDIM):
    if nDIM==1:
        return nn.Conv1d
    elif nDIM==2:
        return nn.Conv2d
    else:
        raise ValueError('nn_ConvNd: nDIM='+str(nDIM) )
def nn_ConvTransposeNd(nDIM):
    if nDIM==1:
        return nn.ConvTranspose1d
    elif nDIM==2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('nn_ConvTransposeNd: nDIM='+str(nDIM) )
def nn_MaxPoolNd(nDIM):
    if nDIM==1:
        return nn.MaxPool1d
    elif nDIM==2:
        return nn.MaxPool2d
    else:
        raise ValueError('nn_MaxPoolNd: nDIM='+str(nDIM) )
def nn_AvgPoolNd(nDIM):
    if nDIM == 1:
        return nn.AvgPool1d
    elif nDIM == 2:
        return nn.AvgPool2d
    else:
        raise ValueError('nn_AvgPoolNd: nDIM=' + str(nDIM))
def nn_BatchNormNd(nDIM):
    if nDIM == 1:
        return nn.BatchNorm1d
    elif nDIM == 2:
        return nn.BatchNorm2d
    else:
        raise ValueError('nn_BatchNormNd: nDIM=' + str(nDIM))


# -------------------
class InceptionND_v3(nn.Module):
    def __init__(self, nDIM, in_fts, out_fts):
        super(InceptionND_v3, self).__init__()
        self.nDIM = nDIM

        # nn_ConvNd = nn.Conv1d if nDIM==1 else nn.Conv2d
        if type(out_fts) is not list:
            out_fts = [out_fts // 4, out_fts // 4, out_fts // 4, out_fts // 4]
        ###################################
        ### 1x1 conv + 3x3  conv + 3x3 conv
        ###################################
        self.branch1 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[0], kernel_size=1, stride=1),
            nn_ConvNd(nDIM)(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular'),
            nn_ConvNd(nDIM)(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular')
        )
        ###################################
        ### 1x1 conv  + 3x3 conv
        ###################################
        self.branch2 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[1], kernel_size=1, stride=1),
            nn_ConvNd(nDIM)(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular'),
        )
        ###################################
        ###  3x3 MAX POOL  +  1x1 CONV
        ###################################
        self.branch3 = nn.Sequential(
            nn_MaxPoolNd(nDIM)(kernel_size=3, stride=1, padding=1),
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[2], kernel_size=1, stride=1)
        )
        ###################################
        ###  1x1 CONV
        ###################################
        self.branch4 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[3], kernel_size=1, stride=1)
        )

    def forward(self, input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        x = torch.cat([o1, o2, o3, o4], dim=-1 - self.nDIM)
        return x


# ---------------------------
class ResidualBlockNd(nn.Module):
    def __init__(self, nDIM, numRepeat, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 padding_mode='circular', bias=True, bRelu=True, bNorm=False):
        super(ResidualBlockNd, self).__init__()
        self.nDIM = nDIM
        self.numRepeat = numRepeat

        self.bRelu = bRelu
        self.bNorm = bNorm

        layers = []
        layers.append(
            nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode,
                            bias=bias))
        if bRelu == True:     layers.append(nn.ReLU())
        if bNorm == True:     layers.append(nn_BatchNormNd(nDIM)(out_channels))
        self.cnn1 = nn.Sequential(*layers)

        # self.cnn1 =nn.Sequential(
        #    nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding, padding_mode=padding_mode, bias=bias),
        #    nn.ReLU(),
        #    nn.BatchNorm1d(out_channels),
        # )
        # self.cnn1.apply(init_weights)

        layers = []
        layers.append(
            nn_ConvNd(nDIM)(out_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode,
                            bias=bias))
        if bNorm == True:     layers.append(nn_BatchNormNd(nDIM)(out_channels))
        self.cnn2 = nn.Sequential(*layers)
        # self.cnn2 = nn.Sequential(
        #    nn.Conv1d(out_channels,out_channels,kernel_size,     1,padding,padding_mode=padding_mode, bias=bias),
        #    nn.BatchNorm1d(out_channels)
        # )

        # if stride != 1 or in_channels != out_channels:
        #    self.shortcut = nn.Sequential(
        #        nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=stride,bias=bias),
        #        #nn.BatchNorm1d(out_channels)
        #    )

        if stride != 1 or in_channels != out_channels:

            layers = []
            layers.append(nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                          padding_mode=padding_mode, bias=bias))
            if bNorm == True:     layers.append(nn_BatchNormNd(nDIM)(out_channels))
            self.shortcut = nn.Sequential(*layers)

            # self.shortcut = nn.Sequential(
            #        nn.Conv1d(in_channels,out_channels,kernel_size=3,padding=1, padding_mode=padding_mode, stride=stride,bias=bias),
            #        nn.BatchNorm1d(out_channels)
            #      )

        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):

        for dummy in range(self.numRepeat):
            residual = x
            x = self.cnn1(x)
            x = self.cnn2(x)
            x += self.shortcut(residual)
            if self.bRelu:
                x = nn.ReLU()(x)

        return x

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

#----------------------------------------

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) #, gain=nn.init.calculate_gain('relu'))
        #m.weight.data.fill_(1.0)
        #print('init weights:', m)
    elif type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #print('init weights:', m)
