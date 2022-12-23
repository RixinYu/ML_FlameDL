
#-----------------------------------------------------
#from utilities3 import *
import operator
from functools import reduce
#from torch.autograd import Variable
################################################################
# require torch.__version__ == 1.9.0
################################################################
#-----------------------------------------------------


import torch
import torch.nn as nn
from functools import partial

#Complex multiplication
def compl_mulNd(a, b, op_einsum= None):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    if op_einsum is None:
        if a.ndim == 4:  # compl_mul1d
            op_einsum="bix,iox->box"
        elif a.ndim == 5: # compl_mul2d
            op_einsum="bixy,ioxy->boxy"
        else: 
            raise ValueError('wrong a.ndim')

    op = partial(torch.einsum, op_einsum )
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

def compl_mulNd_nuMulti(a, b, s, op_einsum=None):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    if op_einsum is None:
        if a.ndim == 4:  # compl_mul1d
            op_einsum="bix,iox,bx->box"
        elif a.ndim == 5: # compl_mul2d
            op_einsum="bixy,ioxy,bxy->boxy" 
        else: 
            raise ValueError('wrong a.ndim')

    op3 = partial(torch.einsum, op_einsum )
    return torch.stack([
        op3(a[..., 0], b[..., 0], s ) - op3(a[..., 1], b[..., 1], s),
        op3(a[..., 1], b[..., 0], s ) + op3(a[..., 0], b[..., 1], s)
    ], dim=-1)
    
class SpectralConvNd_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes_fourier ):    # , mode_operation = None):
        super(SpectralConvNd_fast, self).__init__()
        
        if type(modes_fourier) == int:
            self.nDIM = 1
        else:
            assert len(modes_fourier)==2 
            self.nDIM = 2

        """
        Nd Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        #self.myparams ={
        #    'in_channels':in_channels,
        #    'out_channels':out_channels,
        #    'mode_fourier':modes_fourier,
        #    'mode_operation':mode_operation
        #    }

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_fourier = modes_fourier #Number of Fourier modes to multiply, at most floor(N/2) + 1
        
        scale = (1 / (in_channels * out_channels))
    

        if self.nDIM == 2:
            self.weights_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, 2 ) )
            self.weights_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, 2 ) )
            #for d,_ in enumerate( self.modes_fourier )
            #    self.weights[d] = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, 2 ) )
        elif self.nDIM==1:
            self.weights = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_fourier, 2 ) )



    def forward(self, x ,fourierweight_scaling):

        batchsize = x.shape[0]
        if self.nDIM == 2:
            x_ft   = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),  x.size(-1)//2 + 1, 2, device=x.device)
        elif self.nDIM==1:
            x_ft   = torch.fft.rfft (x, norm="ortho")
            out_ft = torch.zeros(batchsize, self.out_channels,               x.size(-1)//2 + 1, 2, device=x.device)
        
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        if self.nDIM==2:
            if type( fourierweight_scaling ) is torch.Tensor:
                if fourierweight_scaling.shape[-1] != self.modes_fourier[-1] or fourierweight_scaling.shape[-2] != self.modes_fourier[-2]  :
                    raise ValueError('wrong weights scaling (SpectralConvNd_fast)') 
                out_ft[:,:, :self.modes_fourier[0], :self.modes_fourier[1] ] = compl_mulNd_nuMulti( torch.view_as_real(x_ft)[:, :, :self.modes_fourier[0], :self.modes_fourier[1] ], self.weights_dim0, fourierweight_scaling)
                out_ft[:,:,-self.modes_fourier[0]:, :self.modes_fourier[1] ] = compl_mulNd_nuMulti( torch.view_as_real(x_ft)[:, :,-self.modes_fourier[0]:, :self.modes_fourier[1] ], self.weights_dim1, fourierweight_scaling)
            else: 
                out_ft[:,:, :self.modes_fourier[0], :self.modes_fourier[1]] = compl_mulNd( torch.view_as_real(x_ft)[:, :, :self.modes_fourier[0], :self.modes_fourier[1] ], self.weights_dim0 ) 
                out_ft[:,:, -self.modes_fourier[0]:, :self.modes_fourier[1]] = compl_mulNd( torch.view_as_real(x_ft)[:, :,-self.modes_fourier[0]:, :self.modes_fourier[1] ], self.weights_dim1 ) 

        elif self.nDIM==1:

            if type( fourierweight_scaling ) is torch.Tensor:

                if fourierweight_scaling.shape[-1] != self.modes_fourier:
                    raise ValueError('wrong weights scaling (SpectralConvNd_fast)')

                out_ft[:,:, :self.modes_fourier] = compl_mulNd_nuMulti( torch.view_as_real(x_ft)[:, :, :self.modes_fourier], self.weights, fourierweight_scaling)

            else: 
                out_ft[:,:, :self.modes_fourier] = compl_mulNd( torch.view_as_real(x_ft)[:, :, :self.modes_fourier], self.weights ) 

        ###########################################################

        #Return to physical space
        if self.nDIM ==2: 
            x= torch.fft.irfft2( torch.view_as_complex(out_ft), (x.size(-2), x.size(-1)), norm='ortho' )
        elif self.nDIM==1:
            x= torch.fft.irfft( torch.view_as_complex(out_ft),                x.size(-1), norm='ortho' )

        return x

class Fc_PDEpara(nn.Module):
    def __init__(self, num_PDEParameters, modes_fourier): 
        super(Fc_PDEpara, self).__init__()

        assert num_PDEParameters>= 1

        self.num_PDEParameters  = num_PDEParameters
        self.modes_fourier = modes_fourier
    
        if type(modes_fourier)==int: 
            self.nDIM_PDE=1
        else: 
            assert len(modes_fourier) ==2
            self.nDIM_PDE =2

        if self.num_PDEParameters>=1: # learning  PDEs with multi parameters
            self.block0= nn.Sequential( 
                nn.Linear( self.num_PDEParameters, 10),     nn.Sigmoid(),
                nn.Linear( 10, 10),                         nn.Sigmoid()
            ) 
            if self.nDIM_PDE==1 : 
                self.blockout= nn.Sequential(    nn.Linear(10,  self.modes_fourier) ,  nn.Sigmoid())
            elif self.nDIM_PDE==2:
                self.blockout_dim0= nn.Sequential(    nn.Linear(10,  self.modes_fourier[0]),  nn.Sigmoid() )
                self.blockout_dim1= nn.Sequential(    nn.Linear(10,  self.modes_fourier[1]),  nn.Sigmoid() )

            #self.list_layers_F_PDEPara = [ [self.num_PDEParameters, 10], [10, 10], [10, self.modes_fourier[0]] ] 
            #self.F_PDEPara = nn.ModuleList()
            #for idx, layer_struct in enumerate(self.list_layers_F_PDEPara) :
            #    self.F_PDEPara.append( nn.Linear( *layer_struct ) )

    def forward(self,x):
        
        batch, last = x.shape
        assert last == self.num_PDEParameters
        x = self.block0(x)

        if self.nDIM_PDE ==1: 
            x = self.blockout(x) 
        elif self.nDIM_PDE ==2: 
            a = self.blockout_dim0(x).unsqueeze(-1).repeat(1, 1,self.modes_fourier[1] ) 
            b = self.blockout_dim1(x).unsqueeze(-2).repeat(1, self.modes_fourier[0],1 )
            x= a*b

        x = 0.4 + x*0.6 # the output value lies between 0.4 and 1.0
            
        return x


class FourierOp_Nd(nn.Module):
    def __init__(self, nDIM, modes_fourier, width,
                 method_TimeAdv='simple',
                 T_in=1, depth=4,
                 num_PDEParameters=0,
                 data_channel=1,
                 method_Attention=0,
                 method_WeightSharing=0,
                 method_SkipConnection=0,
                 method_BatchNorm=0,
                 brelu_last=1):
        super(FourierOp_Nd, self).__init__()
        
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 1 location (u(t-10, x), ..., u(t-1, x),  x)
        input shape: (batchsize, x=64, y=64, c=11)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.nDIM = nDIM
        if nDIM ==2:
            assert len(modes_fourier)== nDIM

        self.modes_fourier     = modes_fourier
        self.width             = width
        self.depth             = depth
        self.num_PDEParameters = num_PDEParameters
        self.T_in                = T_in
        self.data_channel      = data_channel
        self.method_TimeAdv    = method_TimeAdv
        self.method_Attention  = method_Attention
        self.method_WeightSharing = method_WeightSharing
        self.method_SkipConnection = method_SkipConnection
        self.method_BatchNorm     = method_BatchNorm
        self.brelu_last           = brelu_last
        #
        if self.method_Attention == 1:
            # Attension  mechansim, similar to transformer in language modelling
            self.conv_U = SpectralConvNd_fast(self.width, self.width, self.modes_fourier)  
            self.conv_V = SpectralConvNd_fast(self.width, self.width, self.modes_fourier)  
            if self.nDIM == 2: 
                self.w_U    = nn.Conv2d(self.width, self.width, 1)
                self.w_V    = nn.Conv2d(self.width, self.width, 1)
            elif self.nDIM == 1: 
                self.w_U    = nn.Conv1d(self.width, self.width, 1)
                self.w_V    = nn.Conv1d(self.width, self.width, 1)

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        if self.method_WeightSharing >= 1: 
            for l in range(self.method_WeightSharing):
                self.conv.append( SpectralConvNd_fast(self.width, self.width, self.modes_fourier) )   
                if self.nDIM == 2: 
                    self.w.append( nn.Conv2d(self.width, self.width, 1)  )   
                elif self.nDIM==1: 
                    self.w.append( nn.Conv1d(self.width, self.width, 1)  )   
        else: 
            for l in range(self.depth):
                self.conv.append( SpectralConvNd_fast(self.width, self.width, self.modes_fourier)  )   
                if self.nDIM == 2: 
                    self.w.append( nn.Conv2d(self.width, self.width, 1)  )   
                elif self.nDIM==1: 
                    self.w.append( nn.Conv1d(self.width, self.width, 1)  )   

        #self.conv0 = SpectralConv1d_fast(self.width, self.width, self.modes)
        #self.conv3 = SpectralConv1d_fast(self.width, self.width, self.modes)

        #self.w0 = nn.Conv1d(self.width, self.width, 1)
        #self.w3 = nn.Conv1d(self.width, self.width, 1)
 


        if self.method_BatchNorm==1:
            self.bn = nn.ModuleList()
            if self.method_WeightSharing >= 1: 
                for l in range(self.method_WeightSharing):
                    if self.nDIM == 2:
                        self.bn.append( torch.nn.BatchNorm2d(self.width)  )
                    elif self.nDIM == 1:
                        self.bn.append( torch.nn.BatchNorm1d(self.width)  )
            else:
                for l in range(self.depth):
                    if self.nDIM == 2:
                        self.bn.append( torch.nn.BatchNorm2d(self.width)  )
                    elif self.nDIM==1: 
                        self.bn.append( torch.nn.BatchNorm1d(self.width)  )


        #self.bn0 = torch.nn.BatchNorm1d(self.width)
        #self.bn3 = torch.nn.BatchNorm1d(self.width)
        ################################################################


        if self.num_PDEParameters>=1: # learning  PDEs with multi parameters
            self.fc_PDEPara = Fc_PDEpara(self.num_PDEParameters,self.modes_fourier)

            #self.list_layers_F_PDEPara = [ [self.num_PDEParameters, 10], [10, 10], [10, self.modes_fourier] ] 
            #self.F_PDEPara = nn.ModuleList()
            #for idx, layer_struct in enumerate(self.list_layers_F_PDEPara) :
            #    self.F_PDEPara.append( nn.Linear( *layer_struct ) )

        ############################################################    

        if self.method_TimeAdv.casefold() == 'simple': # simple time advancement

            self.fc_in = nn.Linear( T_in*self.data_channel, self.width)
            # input channel is T_in: the solution of the previous T_in timesteps # + 1 location (u(t-10, x), ..., u(t-1, x),  x)            
            self.fc_out0 = nn.Linear(self.width, 128)
            self.fc_out1 = nn.Linear(128, 1*self.data_channel )

        elif self.method_TimeAdv.casefold() == 'gru': # GRU

            self.fc0_in_GRU = nn.Linear( 2*self.data_channel, self.width, bias=False)
            self.fc1_in_GRU = nn.Linear( 2*self.data_channel, self.width, bias=False)

            if self.nDIM == 2: 
                self.w__xh_z_GRU = nn.Conv2d( self.width, self.width, 1)
                self.w__xh_r_GRU = nn.Conv2d( self.width, self.width, 1)
            elif self.nDIM==1: 
                self.w__xh_z_GRU = nn.Conv1d( self.width, self.width, 1)
                self.w__xh_r_GRU = nn.Conv1d( self.width, self.width, 1)

            #self.conv__xrh_H = nn.ModuleList()
            #self.w__xrh_H    = nn.ModuleList()
            #for l in range(self.depth):
            #    self.conv__xrh_H.append( SpectralConv1d_fast(self.width, self.width, self.modes_fourier,mode_operation) )
            #    self.w__xrh_H.append( nn.Conv1d(self.width, self.width, 1)  )

            self.fc_R_GRU = nn.Linear(self.width, 1*self.data_channel,bias=False)
            self.fc_Z_GRU = nn.Linear(self.width, 1*self.data_channel,bias=False)

            self.fc_H0_GRU = nn.Linear(self.width, 128)
            self.fc_H1_GRU = nn.Linear(128, 1*self.data_channel)

        return


    def update_fourierweight_scaling(self, PDEparas, device ):
       #assert batchsize == PDEparas.shape[0]
        if self.num_PDEParameters >=1:
            #PDEPara_va      = torch.reshape( PDEparas, (-1,1,1)  ).expand( -1, self.modes_fourier,-1 ).to( x.device )
            #k_mode          = torch.arange ( self.modes_fourier ).reshape(1,-1,1).expand( batchsize, -1,-1 ).to( x.device )
            #fourierweight_scaling  = torch.cat( (PDEPara_va ,k_mode) ,  dim = -1  )
            ##---------------------------------
            #for idx, fc in enumerate(self.F_PDEPara):
            #    fourierweight_scaling = fc( fourierweight_scaling ) 
            #    #if idx < len(self.list_layers_F_PDEPara)-1:
            #    fourierweight_scaling = torch.sigmoid(fourierweight_scaling )
            ## Eventually, fourierweight_scaling.shape == batch, 1
            #fourierweight_scaling  = 0.5+ fourierweight_scaling/2  # the output value lies between 0.5 and 1.0
            ##----------------------------------
            if PDEparas.ndim==1 and  self.num_PDEParameters==1: 
                PDEparas =  PDEparas.unsqueeze(-1)
            fourierweight_scaling = self.fc_PDEPara ( PDEparas  ).to(device)
            #---------------------------------
            #for idx, fc in enumerate(self.F_PDEPara):
            #    fourierweight_scaling = fc( fourierweight_scaling ) 
            #    #if idx < len(self.list_layers_F_PDEPara)-1:
            #    fourierweight_scaling = torch.sigmoid(fourierweight_scaling )
            ## Eventually, fourierweight_scaling.shape == batch, 1
            #fourierweight_scaling  = 0.4+ fourierweight_scaling*.6  # the output value lies between 0.5 and 1.0
            #----------------------------------
            #fourierweight_scaling = fourierweight_scaling.to(device)
        else:
            fourierweight_scaling = 1.0
        
        return fourierweight_scaling
        

    def depth_advance_fixed_width(self,x,fourierweight_scaling):
        if self.method_Attention==1:
            if self.method_SkipConnection== 1:
                U = x+ torch.relu( self.conv_U(x,fourierweight_scaling)  + self.w_U(x) )
                V = x+ torch.relu( self.conv_V(x,fourierweight_scaling)  + self.w_V(x) )
            else:
                U =    torch.relu( self.conv_U(x,fourierweight_scaling)  + self.w_U(x) )
                V =    torch.relu( self.conv_V(x,fourierweight_scaling)  + self.w_V(x) )

            # Attension  mechansim, similar to transformer in language modelling
            H =  torch.sigmoid( self.conv[0](x,fourierweight_scaling)  + self.w[0](x) )
            for l in range(1, self.depth-1):
                l_actual = 0 if self.method_WeightSharing >=1 else l
                Z = torch.sigmoid( self.conv[l_actual](H,fourierweight_scaling)  + self.w[l_actual](H) )
                H =(1-Z)* U + Z*V
            
            # special for the last layer
            l = self.depth -1 
            l_actual = self.method_WeightSharing-1 if self.method_WeightSharing >=1 else l

            if self.method_SkipConnection == 1:
                x = H+ torch.relu( self.conv[l_actual](H,fourierweight_scaling)  + self.w[l_actual](H)  )
            else:
                x =    torch.relu( self.conv[l_actual](H,fourierweight_scaling)  + self.w[l_actual](H)  )
    
        elif self.method_Attention==0: 
            
            for l in range(self.depth):
                #l_actual = l
                #if  self.method_WeightSharing == 1  or (l<self.depth-1  and  self.method_WeightSharing == 2) :
                #    l_actual = 0 
                #if l==self.depth-1 and self.method_WeightSharing == 2 :
                #    l_actual = 1
                l_actual = max( 0,  l-(self.depth-1)+(self.method_WeightSharing-1)  )  if self.method_WeightSharing>0 else l

                x_12 = self.conv[l_actual](x,fourierweight_scaling)+ self.w[l_actual](x)
                if self.method_BatchNorm == 1: # apply batch normlization before nonlinear func
                    x_12 = self.bn[l_actual](x_12)

                #if not (l==self.depth-1 and self.method_WeightSharing ==2) :
                if l<self.depth-1  or  ( l==self.depth-1 and self.brelu_last==1) :
                    x_12 =  torch.relu(x_12)

                x = x + x_12 if self.method_SkipConnection==1 else x_12


            #l = self.depth-1
            #x = self.conv[l_spectral](x,fourierweight_scaling)  + self.w[l_w](x) 


            #x1 = self.conv0(x)
            #x2 = self.w0(x)  # x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            #x = self.bn0(x1 + x2)
            #x= x1+x2
            #x = F.relu(x)

            #x1 = self.conv1(x)
            #x2 = self.w1(x) # x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            ##x= x1+x2
            #x = self.bn1(x1 + x2)
            #x = F.relu(x)

            #x1 = self.conv2(x)
            #x2 = self.w2(x)  #.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            ##x = self.bn2(x1 + x2)
            #x= x1+x2
            #x = F.relu(x)

            #x1 = self.conv3(x)
            #x2 = self.w3(x) #.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            ##x = self.bn3(x1 + x2)
            #x= x1+x2        
        return x



    def forward(self,x,PDEparas=None, init_states=None ):
        if self.method_TimeAdv.casefold()=='simple':

            return self.forward_simple(x,PDEparas)

        elif self.method_TimeAdv.casefold()=='gru':

            return self.forward_gru(x,PDEparas, init_states)


    def forward_simple(self, x, PDEparas ):
        #batchsize = x.shape[0]

        fourierweight_scaling = self.update_fourierweight_scaling(PDEparas,x.device )
        #
        if self.nDIM == 1:
            x = self.fc_in(x).permute(0, 2, 1)
        elif self.nDIM==2:
            x = self.fc_in(x).permute(0, 3, 1, 2)

        #
        x=self.depth_advance_fixed_width(x,fourierweight_scaling)
        #

        if self.nDIM == 1:
            x = x.permute(0, 2,1 )
        elif self.nDIM==2:
            x = x.permute(0, 2, 3, 1 )
        
        x=  torch.relu( self.fc_out0(x) )  
        x = self.fc_out1(x)
 
        return x




    def initHidden_gru(self, batch_size, hidden_size, device):
        return torch.zeros(batch_size, hidden_size, self.data_channel ).to(device)
     
    def forward_gru(self, x , PDEparas, init_states=None ):
        fourierweight_scaling = self.update_fourierweight_scaling(PDEparas, x.device)

        #
        bs, S_sz, seq_sz = x.size()
        hidden_seq = []
        
        if init_states is None:
            #h_t = torch.zeros(bs, 1, hidden_size).to(x.device)
            h_t = self.inithidden_gru(bs, S_sz,x.device)
        else:
            h_t = init_states

        for t in range(seq_sz):
            x_t = x[:,  :,   t*self.data_channel:(t+1)*self.data_channel ] # [batch_size, 512, 1]
            # [batch_size, 512, 2] --fc--> [batch_size, 512, 32] --perm-->[batch_size, 32, 512]
            xh_t = self.fc0_in_GRU( torch.cat( [x_t, h_t],dim=-1)).permute(0, 2, 1)

            # [batch_size, 32, 512 ] --Conv+W-->[batch_size, 32, 512 ] --perm-->[batch_size, 512, 32 ] --fc-->[batch_size, 512,1 ]
            #z_t =torch.sigmoid( self.fc_Z__width_to_1(  (self.conv__xh_z_0(xh_t) + self.w__xh_z_0(xh_t)).permute(0,2,1)    ) )
            #r_t =torch.sigmoid( self.fc_R__width_to_1(  (self.conv__xh_r_0(xh_t) + self.w__xh_r_0(xh_t)).permute(0,2,1)    ) )
            #z_t =0.001*torch.tanh( self.fc_Z__width_to_1(  ( self.w__xh_z_0(xh_t)).permute(0,2,1)    ) )
            #r_t =0.001*torch.tanh( self.fc_R__width_to_1(  ( self.w__xh_r_0(xh_t)).permute(0,2,1)    ) )
            
            z_t =torch.sigmoid( self.fc_Z_GRU(  ( self.w__xh_z_GRU(xh_t)).permute(0,2,1)    ) )
            r_t =torch.sigmoid( self.fc_R_GRU(  ( self.w__xh_r_GRU(xh_t)).permute(0,2,1)    ) )

            # [batch_size, 512, 2] --fc--> [batch_size, 512, 32] --perm-->[batch_size, 32, 512]
            xrh_t = self.fc1_in_GRU( torch.cat( [x_t,  r_t*h_t  ], dim=-1) ).permute(0, 2, 1)

            # [batch_size, 32, 512 ] --Conv+W-->[batch_size, 32, 512 ]
            #xrh_t = self.conv__xrh_H_0(xrh_t) + self.w__xrh_H_0(xrh_t)
            #xrh_t = torch.relu( xrh_t)
            #xrh_t = self.conv__xrh_H_1(xrh_t) + self.w__xrh_H_1(xrh_t)
            #xrh_t = torch.relu( xrh_t)
            #xrh_t = self.conv__xrh_H_2(xrh_t) + self.w__xrh_H_2(xrh_t)
            #xrh_t = torch.relu( xrh_t)
            #xrh_t = self.conv__xrh_H_3(xrh_t) + self.w__xrh_H_3(xrh_t)
            
            #for l in range(self.depth):
            #    xrh_t = torch.relu( self.conv__xrh_H[l]( xrh_t ) + self.w__xrh_H[l](xrh_t)  )
            
            xrh_t = self.depth_advance_fixed_width(xrh_t,fourierweight_scaling)

            #xrh_t = torch.tanh( xrh_t)
            #H_t =   torch.tanh(  self.fc_H__width_to_1(  xrh_t.permute(0,2,1)  ) )

            # [batch_size, 32, 512 ] --perm-->[batch_size, 512, 32 ] -fc -> [batch_size, 512, 128 ] --fc-->[batch_size, 512,1 ]
            H_t = self.fc_H1_GRU(  self.fc_H0_GRU( xrh_t.permute(0,2,1) ) )

            h_t =  z_t* h_t + (1-z_t) * H_t
            #hidden_seq.append(h_t.unsqueeze(0))
            hidden_seq.append(h_t)


        hidden_seq = torch.cat(hidden_seq, dim=-1)
        #hidden_seq = hidden_seq.contiguous()
        #hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, h_t


    #def count_learnable_params(self):
    #    c = 0
    #    for p in self.parameters():
    #        c += reduce(operator.mul, list(p.size()))
    #    return c







#class Net1d(nn.Module):
#    
#    def __init__(self, modes_fourier, width, T_in=1,depth=4, params={'num_PDEParameters':0, 'data_channel':1, 'method_Attention':0,'method_WeightSharing':'','method_SkipConnection':0}    ):
#        super(Net1d, self).__init__()
#
#        """
#        A wrapper function
#        """
#        self.bIs_RNN = False
#        self.conv1 = SimpleBlock1d(modes_fourier, width,T_in=T_in,depth=depth, **params )
#    def forward(self, x, PDEpara =1.0):
#        y = self.conv1(x, PDEpara)
#
#        return y





