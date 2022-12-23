


#-----------------------------------------------------
#import torch.nn.functional as F
#from  torch.autograd.functional import vjp
#import operator
#from functools import reduce
#from functools import partial
#from utilities3 import *
#-----------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from .MyConvNd import MyConvNd, nn_MaxPoolNd, nn_AvgPoolNd




class ConvPDE_Nd(nn.Module):
    def __init__(self, nDIM, N,
                 in_channel=1, out_channel=1,
                 en1_channels =[ 8,16,32,64,[64] ], #  None, [,,,,]
                 de1_channels = None,
                 method_nonlinear='all', # 'all', 'none', 'de'(i.e. decoder)
                 method_types_conv ='conv_all',
                 #method_OP='', #this key is removed: ( #'none', 'OP')
                 method_skip='full',#  'off', #   (decaperated:  'width2', 'width4' )
                 bUpSampleOrConvTranspose='upsample',
                 method_pool='max', # 'ave'
                 #method_conv='',       #this key is removed!!!! # 'more' # no allowed( 'share')
                 num_PDEParameters=0 ,
                 method_BatchNorm=False
                 #bExternalConstraint = False
                 #,yB_1DNormalization =None
                 ):
        super(ConvPDE_Nd, self).__init__()

        #self.bExternalConstraint = bExternalConstraint

        self.nDIM = nDIM
        self.N = N
        self.in_channel = in_channel
        if out_channel is None:
            out_channel = in_channel

        self.out_channel = out_channel
        self.num_PDEParameters=num_PDEParameters

        if 'off' in method_skip.casefold(): #
            self.bSkipConnect = False
        else:
            self.bSkipConnect = True
            if 'n' in method_skip.casefold()[0]:
                self.n_skipconnect =  int(  method_skip.casefold()[-1] )


        #--------------------
        #if en1_channels is  None:  # default [1, 1, 1, 1, 1]*in_channel
        #    nLen = int( np.log2(N) ) - 2
        #    en1_channels = (np.ones(nLen)*in_channel).astype(int)  #  np.array( [ inout_channel  for idx in range( nLen ) ]   )



        #if type( en1_channels[0] ) is list:
        #elif type( en1_channels[0] ) is int:
        #    self.en_channels   = np.concatenate( [  [in_channel], en1_channels ] ) #   np.insert(en1_channels, 0, indata_channel)    # [ indata_channel ] + en1_channels
        #    self.de_channels   = np.concatenate( [ en1_channels[::-1], [out_channel]          ] )         # en1_channels[::-1] + [ 1]


        # The following convert [3,[5,8],7] to [[3],[5,8],[7]]
        for idx, c in enumerate( en1_channels):
            try: # check if it is a list
                c[0]
            except:
                en1_channels[idx] = [c]

        if de1_channels is None:
            de1_channels =  en1_channels[::-1]  + [[out_channel]]  # (i) skip the last then reverse, later get the last element
            for idx, c_l in enumerate(de1_channels):
                de1_channels[idx] = [ c_l[-1] ]



        #self.en_channels   =  [ [in_channel] ]     + [ [ [ n_t[0] ]  for n_t in n_t__n_t ] for n_t__n_t in  en1_channels  ]
        #self.de_channels   =   [ [ [ n_t[0] ]  for n_t in n_t__n_t ] for n_t__n_t in  de1_channels  ]    +  [ [out_channel]  ]
        #self.en_types   =    [ [ [ n_t[1] ]  for n_t in n_t__n_t ] for n_t__n_t in  en1_channels  ]
        #self.de_types   =    [ [ [ n_t[1] ]  for n_t in n_t__n_t ] for n_t__n_t in  de1_channels  ]

        #---------------
        en_channels   =  en1_channels[:] #make a copy
        for l, _ in enumerate(en_channels ):
            first_channel = in_channel if l==0 else en_channels[l-1][-1]
            en_channels[l] = [first_channel] + en_channels[l]
        self.en_channels  = en_channels
        #---------------

        de_channels = de1_channels[:] #make a copy
        for l, _ in enumerate(de_channels):
            first_channel = en_channels[-1][-1] if l==0 else de_channels[l-1][-1]
            if l>= 1:
                if hasattr(self,'n_skipconnect'):
                    if l <= (len(de_channels)-1) - self.n_skipconnect :
                        first_channel = first_channel * 2
                else:
                    first_channel = first_channel*2
            de_channels[l] = [first_channel] + de_channels[l]
        self.de_channels  = de_channels

        print( 'ConvPDE_Nd: en_channels = ', self.en_channels )
        print( 'ConvPDE_Nd: de_channels = ', self.de_channels )

        #self.en_Nmesh_list =  N//( 2**( np.array( [i for i in range(len(self.en_channels))]) ) )  # make sure the last is >= 4
        #self.de_Nmesh_list = self.en_Nmesh_list[::-1]
        # --------------------
        #self.yB_1DNormalization = None
        #if yB_1DNormalization is not None:
        #    self.yB_1DNormalization = torch.tensor( yB_1DNormalization, dtype=torch.float )


        #------------------------
        self.method_types_conv = method_types_conv
        #Len_en1 = len(en1_channels)
        #if  'conv_all' in method_types_conv.casefold():
        #    self.en_types   = ['Conv' for _ in range(Len_en1)  ]
        #    self.de_types   = ['Conv' for _ in range(Len_en1) ]
        #elif 'res' in method_types_conv.casefold():
        #    self.en_types   = ['Conv'] + ['Residual' for _ in range(Len_en1-1)  ]
        #    self.de_types   = ['Conv'] + ['Residual' for _ in range(Len_en1-1)  ]
        #elif 'inception_most' in method_types_conv.casefold():
        #    self.en_types   = ['Inception' for _ in range(Len_en1  ) ]
        #    self.de_types   = ['Inception' for _ in range(Len_en1-1) ]  + ['Conv']
        #elif 'inception_less' in method_types_conv.casefold():
        #    self.en_types   = ['Conv' for _ in  range(Len_en1) ] ; self.en_types[-1]='Inception';  self.en_types[2] ='Inception'
        #   self.de_types   = ['Conv' for _ in  range(Len_en1) ] ; self.de_types[1] ='Inception';  self.de_types[-2]='Inception'
        #else:
        #    raise ValueError( method_types_conv + ', wrong method_types_conv')
        #------------------------

        type_char =  method_types_conv[0]
        en_types = [ [type_char for _ in range(len(c_l)) ] for c_l in en1_channels]
        de_types = [ [type_char for _ in range(len(c_l)) ] for c_l in de1_channels]

        if 'inception_most' in method_types_conv.casefold():
            de_types[-1][-1] = 'c'
        elif 'inception_less' in method_types_conv.casefold():
            en_types = [['c' for _ in range(len(c_l))] for c_l in en1_channels]
            de_types = [['c' for _ in range(len(c_l))] for c_l in de1_channels]
            en_types[ 1][-1] = 'i'
            en_types[ 2][-1] = 'i'
            en_types[ 3][-1] = 'i'

            de_types[-2][-1] = 'i'
        #else:
        #    raise ValueError( method_types_conv + ', wrong method_types_conv')

        self.en_types =en_types
        self.de_types =de_types
        print('ConvPDE_Nd: en_types = ', en_types)
        print('ConvPDE_Nd: de_types = ', de_types)

        #--------------------------------
        #self.method_conv = method_conv
        self.method_pool = method_pool
        if 'max' in method_pool.casefold():
            PoolNd = nn_MaxPoolNd(nDIM)(kernel_size=2, stride=2)
        else: #'ave'
            PoolNd = nn_AvgPoolNd(nDIM)(kernel_size=2, stride=2)


        bRelu_skip = False

        self.method_skip = method_skip
        #if 'width' in method_skip.casefold():  #'width2' or 'width8'
        #    self.skip_channels_width = int( method_skip[-1] )
        #    #self.skip_in_channels = self.en_channels[0:-2]
        #    #self.skip_out_channels = (np.ones( len(self.en_channels-1) )*self.skip_channels_width).astype(int)
        #    self.Skip_conv = nn.ModuleList()
        #    for l in range( len(self.en_channels)-2 ) :
        #        self.Skip_conv.append(
        #            nn.Sequential(
        #                MyConvNd( nDIM, self.en_channels[l][-1], self.skip_channels_width, kernel_size=3, type='Conv', bRelu=bRelu_skip, bNorm=method_BatchNorm),
        #                PoolNd
        #            )
        #        )

        self.bRelu_Conv = True  #if 'all' in method_nonlinear.casefold() else False

        #self.method_OP= method_OP
        #if 'op' in method_OP.casefold():
        #    self.OP_conv =  nn.ModuleList()
        #    for l in range( len(self.en_channels)-2 ):
        #        in_OP_channel = self.skip_channels_width  if hasattr(self,'Skip_conv') else self.en_channels[l+1]
        #        middle_channel = 8  #self.en_channels[l+1]
        #        self.OP_conv.append(
        #            nn.Sequential(
        #                MyConvNd(nDIM, in_OP_channel ,middle_channel, kernel_size=3, type='Conv', bRelu=self.bRelu_Conv, bNorm=method_BatchNorm),
        #                #MyConv1d(  middle_channel, middle_channel, kernel_size=3, type='Conv', bRelu=False, bNorm=False),
        #                MyConvNd(nDIM, middle_channel, in_OP_channel , kernel_size=3, type='Conv', bRelu=self.bRelu_Conv, bNorm=method_BatchNorm),
        #            )
        #        )

        self.bUpSampleOrConvTranspose = bUpSampleOrConvTranspose

        #-------------------------------------------------------------------------------
        #bEnable_share_conv = False
        #if 'share' in method_conv.casefold() :
        #    bEnable_share_conv = True
        #self.bEnable_share_conv = bEnable_share_conv
        #--------------------------------------------------------------------------------
        #if bEnable_share_conv == True:
        #    self.shared_conv__nInOut = 16
        #    self.share_conv__en = MyConvNd(nDIM, self.shared_conv__nInOut, self.shared_conv__nInOut, kernel_size=3, type='Conv',bRelu=True, bNorm=method_BatchNorm)
        #    self.share_conv__de = MyConvNd(nDIM, self.shared_conv__nInOut, self.shared_conv__nInOut, kernel_size=3, type='Conv',bRelu=True, bNorm=method_BatchNorm)

        #    self.bRelu_Conv0__en = False
        #    self.bRelu_Conv1__en = True if 'all' in method_nonlinear.casefold() else False
        #    self.bRelu_Conv0__de = False
        #    self.bRelu_Conv1__de = True if ('all' in method_nonlinear.casefold() or 'de' in method_nonlinear.casefold() ) else False

        #    self.en_conv0 = nn.ModuleList()
        #    self.en_conv1 = nn.ModuleList()
        #    for l in range( len(self.en_channels)-1 ):
        #        self.en_conv0.append(  MyConvNd(nDIM, self.en_channels[l],self.shared_conv__nInOut,kernel_size=3, type='Conv', bRelu=self.bRelu_Conv0__en, bNorm=method_BatchNorm)               )
        #        self.en_conv1.append(
        #                nn.Sequential(
        #                   MyConvNd(nDIM, self.shared_conv__nInOut,self.en_channels[l+1],kernel_size=3, type='Conv', bRelu=self.bRelu_Conv1__en, bNorm=method_BatchNorm),
        #                   PoolNd
        #                )
        #            )
        #else: # bEnable_share_conv == False:

        self.bRelu_Conv0__en = True if 'all' in method_nonlinear.casefold() else False
        self.bRelu_Conv0__de = True if ('all' in method_nonlinear.casefold() or 'de' in method_nonlinear.casefold() ) else False

        self.en_conv0= nn.ModuleList()
        for l , channels_list in enumerate(self.en_channels) :
            #if type( en1_channels[0] ) is list:
            #channels_list = [ self.en_channels[l][-1] ] + self.en_channels[l+1]
            #self.en_conv0.append(
            #     nn.Sequential(
            #         *[ MyConvNd(nDIM, channels_list[idx],channels_list[idx+1],kernel_size=3, type=self.en_types[l][idx], bRelu=self.bRelu_Conv0__en, bNorm=method_BatchNorm) for idx in range(len(channels_list)-1) ] ,
            #         PoolNd
            #     )
            #)
            layers =[]
            if l >0:
                layers.append( PoolNd )
            #channels_list = [ self.en_channels[l][-1] ] + self.en_channels[l+1]
            for idx in range( len(channels_list)-1):
                layers.append( MyConvNd(nDIM, channels_list[idx],channels_list[idx+1],kernel_size=3, type=self.en_types[l][idx], bRelu=self.bRelu_Conv0__en, bNorm=method_BatchNorm)  )

            self.en_conv0.append( nn.Sequential( *layers ) )


            #elif type( en1_channels[0] ) is int:
            #    self.en_conv0.append(
            #         nn.Sequential(
            #             MyConvNd(nDIM, self.en_channels[l],self.en_channels[l+1],kernel_size=3, type=self.en_types[l], bRelu=self.bRelu_Conv0__en, bNorm=method_BatchNorm),
            #             PoolNd
            #         )
            #    )


        self.de_conv0 = nn.ModuleList()
        #if bEnable_share_conv==True:
        #    self.de_conv1 = nn.ModuleList()

        #for l in range( len(self.de_channels) ):
        for l, channels_list in enumerate( self.de_channels):
            #in_channel = self.de_channels[l][0]
            #if self.bSkipConnect== True and l>0:
            #    if hasattr(self,'Skip_conv'):
            #        in_channel = self.de_channels[l][-1] + self.skip_channels_width
            #    else:
            #        in_channel *= 2  #self.de_channels[l][-1]


            #out_channel_1stlevel = self.de_channels[l+1]
            #--------------------------
            #channels_list =  [in_channel]  + self.de_channels[l+1]
            #--------------------------

            #if bEnable_share_conv == True:
            #    out_channel_1stlevel = self.shared_conv__nInOut

            #--------------------------
            #bRelu_set0 = False     if l == len(self.de_channels)-2 else self.bRelu_Conv0__de

            #if bEnable_share_conv == True:
            #    bRelu_set1 = False if l == len(self.de_channels)-2 else self.bRelu_Conv1__de


            layers = []
            if l ==0:
                layers.append( PoolNd )

            if self.bUpSampleOrConvTranspose =='upsample':
                #layers.append( MyConvNd(nDIM, in_channel, out_channel_1stlevel ,kernel_size=3,      type=self.de_types[l], bRelu=bRelu_set0, bNorm=method_BatchNorm)  )
                for idx in range( len(channels_list)-1 ) :
                    bRelu_set0 = False if l==len(self.de_channels)-1 and idx==len(channels_list)-2 else self.bRelu_Conv0__de
                    layers.append( MyConvNd(nDIM, channels_list[idx] , channels_list[idx+1], kernel_size=3,  type=self.de_types[l][idx], bRelu=bRelu_set0, bNorm=method_BatchNorm)  )
                if l< len(self.de_channels)-1:
                    layers.append(nn.Upsample(scale_factor=2))
            else:
                #layers.append( MyConvNd(nDIM, in_channel, out_channel_1stlevel,kernel_size=2,stride = 2, type='Transpose', bRelu=bRelu_set0, bNorm=method_BatchNorm)  )
                #bRelu_set0 = False if l == len(self.de_channels) - 2 and 0==len(channels_list)-2 else self.bRelu_Conv0__de
                for idx in range( len(channels_list)-2 ) :
                    bRelu_set0 = self.bRelu_Conv0__de
                    layers.append( MyConvNd(nDIM, channels_list[idx] , channels_list[idx+1], kernel_size=3,  type=self.de_types[l][idx], bRelu=bRelu_set0, bNorm=method_BatchNorm)  )

                if l == len(self.de_channels)-1:
                    bRelu_set0 = False
                    type_set = self.de_types[l][len(channels_list)-2]
                    layers.append(   MyConvNd(nDIM, channels_list[-2], channels_list[-1], kernel_size=3,  type=type_set,   bRelu=bRelu_set0, bNorm=method_BatchNorm))
                elif l< len(self.de_channels)-1:
                    bRelu_set0 = self.bRelu_Conv0__de
                    type_set = 'Transpose'
                    layers.append(MyConvNd(nDIM, channels_list[-2], channels_list[-1], kernel_size=2, stride=2, type=type_set, bRelu=bRelu_set0, bNorm=method_BatchNorm))
            #if bEnable_share_conv == False:

            #if 'more' in self.method_conv:
            #    layers.append(  MyConvNd(nDIM, out_channel_1stlevel , self.de_channels[l+1],kernel_size=3, type=self.en_types[l], bRelu=bRelu_set0, bNorm=method_BatchNorm)   )

            #else: #if bEnable_share_conv == True:
            #    self.de_conv1.append( MyConvNd(nDIM, out_channel_1stlevel, self.de_channels[l+1], kernel_size=3, type='Conv',   bRelu=bRelu_set1, bNorm=method_BatchNorm))

            self.de_conv0.append( nn.Sequential( *layers ) )


    def encoder(self,x):
        x_all = []
        #for l in range( len(self.en_channels)-1 ):
        for l, _ in enumerate(self.en_conv0):
            x_en = self.en_conv0[l](x)
            #x_en = self.share_conv_at_level(x_en, l, 'en')

            #if l == len(self.en_channels)-2 and  hasattr(self,'fc_throat') :
            #    x = self.fc_throat(x)

            #if hasattr(self, 'Skip_conv') and l< len(self.en_channels)-1 :
            #    x_skip = self.Skip_conv[l](x)
            #    x_all.append(x_skip)
            #else:

            if hasattr(self, 'n_skipconnect'):
                if l >= self.n_skipconnect:
                    x_all.append(x_en)
            else:
                x_all.append(x_en)

            x = x_en

        return x_all

    def decoder(self, x_all ):

        #if hasattr(self,'OP_conv'):
        #    xx_all = []
        #    for l, x in enumerate( x_all ) :
        #        if l<= len(x_all)-2:
        #            xx_all.append( x + self.OP_conv[l]( x)  )
        #        else:
        #            xx_all.append( x  )
        #    x_all = xx_all

        x_all = x_all[::-1]  #reverse
        x = x_all[0]

        #for l in range( len(self.de_channels)-1 ):

        for l, _ in enumerate(self.de_conv0):
            if self.bSkipConnect==True and l>=1 and  l-1<len(x_all) :
                #print('l', l , x.shape, x_all[l].shape )
                x = self.de_conv0[l]( torch.cat( (x, x_all[l-1]), dim=-1-self.nDIM )   )
                #x = self.share_conv_at_level(x,l,'de')
            else:
                x = self.de_conv0[l]( x )
                #x = self.share_conv_at_level(x,l,'de')
        return x

#    def share_conv_at_level(self,x,l, en_or_de ):
#        if self.bEnable_share_conv == True:
#            if en_or_de == 'de':
#                for i in range(3):
#                    x = x + self.share_conv__de(x)
#                x = self.de_conv1[l](x)
#            elif en_or_de == 'en':
#                for i in range(3):
#                    x = x + self.share_conv__en(x)
#                x = self.en_conv1[l](x)
#        return x

    def forward(self,x):

        #if self.yB_1DNormalization is not None:
        #    x = 2*( x - self.yB_1DNormalization[0] )/( self.yB_1DNormalization[1] - self.yB_1DNormalization[0] ) - 1

        x = x.permute(0, 2, 1) if self.nDIM == 1 else x.permute(0, 3, 1, 2)

        x_all = self.encoder(   x  )
        x     = self.decoder(x_all)

        x = x.permute(0, 2, 1) if self.nDIM == 1 else x.permute(0, 2, 3, 1)


        #if self.yB_1DNormalization is not None:
        #    x = (x+1)/2* (self.yB_1DNormalization[1] - self.yB_1DNormalization[0]) +  self.yB_1DNormalization[0]

        #if self.bExternalConstraint ==True:
        #    if self.nDIM ==1:
        #        x =  x - torch.mean(x , dim=-2,keepdim=True)

        return x
