
#import h5py
#import sklearn.metrics
#from scipy.ndimage import gaussian_filter



import torch
import numpy as np
#import scipy.io
#import torch.nn as nn

import operator
from functools import reduce

from timeit import default_timer
###########################
import time
import pickle

from .libSiva import libSiva, CSolverSiva
from .libData import libData
from .libcfdData import libcfdData

from .DeepONet_1d import DeepONet_1d
from .FourierOp_Nd import FourierOp_Nd
from .ConvPDE_Nd import ConvPDE_Nd

#################################################
#
# lib Utilities
#
#################################################
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Cdata_sys:
    def __init__(self,sysname,syspara,method_default_siva_data_gen=1):
        assert type(syspara)==list
        if 'siva' in sysname.casefold():
            self.sysname = 'siva'
            self.list_nu =  syspara
            self.method_default_siva_data_gen=method_default_siva_data_gen
        elif 'cfd' in sysname.casefold():
            self.sysname    = 'cfd'
            self.list_cfdfilename = syspara
        else:
            raise ValueError('wrong Cdata_sys type')


    def get_num_PDEParameters(self):
        if 'siva' in self.sysname:
            if len(self.list_nu)==1:
                return 0
            else:
                return 1
        elif 'cfd' in self.sysname:
            if len( self.list_cfdfilename)==1:
                return 0
            else:
                return 2


# print('count_learnable_params=', str( count_learnable_params(model) ) )
def count_learnable_params(model):
    c = 0
    for p in model.parameters():
        c += reduce(operator.mul, list(p.size()))
    return c


class lib_Model:
    @staticmethod
    def set_default_params( data_sys, nDIM ):

        assert type(data_sys) == Cdata_sys

        params = {'data_channel': 1,
                  'method_TimeAdv':'simple',
                  'fourier:modes_fourier':32,
                  'fourier:width':20,
                  'fourier:depth':4,
                  'fourier:method_Attention': 0,
                  'fourier:method_WeightSharing': 1,
                  'fourier:method_SkipConnection': 1,
                  'fourier:brelu_last': 1,
                  'fourier:method_BatchNorm': 0,
                  'onet:type_branch':'conv',
                  'onet:P': 30,
                  'onet:fc_layers_branch':[100,100,100,100],
                  'onet:fc_layers_trunk':[100,100,100,100],
                  'onet:trunk_featurepair': 1,
                  'onet:type_trunk': 'simple',
                  'onet:method_nonlinear_act':'tanh',
                  'onet:method_skipconnection':False,
                  'conv:en1_channels':[ [16],[32,32],[64,64],[128],[128],[64],[32]]  ,    # en1_channels=[2,2,2,2,2],[1,1,1,1,1],[4,4,4,4,4],[8,8,8,8,8],[8,16,32,64,64]
                  'conv:de1_channels': None,
                  'conv:out_channel':1,
                  'conv:method_nonlinear':'all',
                  'conv:method_types_conv':'conv_all',
                  #'conv:method_OP':'',
                  'conv:method_skip':'full',
                  'conv:bUpSampleOrConvTranspose':'upsample',
                  'conv:method_pool':'Max',
                  #'conv:method_conv':'',
                  'conv:method_BatchNorm':False
                  #'bExternalConstraint':False
                  #,'yB_1DNormalization':None
        }

        #---------------------------------------
        params['T_in'] = 1
        params['T_out'] =20
        params['T_d_out'] =1
        params['num_PDEParameters'] = data_sys.get_num_PDEParameters()

        params['nDIM']=nDIM

        #------------------------------------
        params['data:yB_estimate']=np.array([-0.7, 1.3]) * np.pi
        params['data:AspectRatio_set'] = 1
        params['data:dir_save_training_data']= './data/'
        params['data:nStep'] = 1
        params['data:nStepSkip'] = 1
        params['Nx'] = 128

        #---------------------------------
        params['train:data_norm_rms'] = 1
        params['train:checkpoint_dir'] = './checkpoints'
        params['train:batch_size'] = 2000
        params['train:learning_rate'] = 0.0025
        params['train:scheduler_step'] = 100
        params['train:scheduler_gamma'] = 0.5
        params['train:epochs'] = 1000
        params['train:epochs_per_save'] = 100
        #if nDIM == 1:
            #params['yB_1DNormalization'] =  np.array([-0.7,1.3])*np.pi

        #el
        if nDIM == 2:
            params['fourier:modes_fourier'] = [32,32]

        return params



    @staticmethod
    def update_dependent_params( data_sys, params ):

        if params['nDIM']==1:
            if 'siva' in data_sys.sysname:
                if data_sys.list_nu == [0.02]:
                    params['train:batch_size'] = 1000
                else:
                    params['train:batch_size'] = 500


        if 'cfd' in data_sys.sysname:
            params['train:batch_size'] = 50

        return params


    @staticmethod
    def build_model(model_name_detail,params):
        if 'fourier' in model_name_detail.casefold():
            model = FourierOp_Nd(params['nDIM'],
                                 params['fourier:modes_fourier'],
                                 params['fourier:width'],
                                 params['method_TimeAdv'],
                                 params['T_in'],
                                 params['fourier:depth'],
                                 params['num_PDEParameters'],
                                 params['data_channel'],
                                 params['fourier:method_Attention'],
                                 params['fourier:method_WeightSharing'],
                                 params['fourier:method_SkipConnection'],
                                 params['fourier:method_BatchNorm'],
                                 params['fourier:brelu_last']    ).cuda()
        elif 'onet' in model_name_detail.casefold():
            assert params['nDIM']==1
            model = DeepONet_1d(params['Nx'],
                                params['onet:type_branch'],
                                params['data_channel'],
                                params['onet:P'],
                                params['onet:trunk_featurepair'],
                                params['onet:type_trunk'],
                                params['num_PDEParameters'],
                                params['onet:method_nonlinear_act'],
                                params['onet:method_skipconnection'],
                                params['T_in']-1,
                                params['train:data_norm_rms'],
                                params['onet:fc_layers_branch'],
                                params['onet:fc_layers_trunk']
                                #,params['yB_1DNormalization']
                                ).cuda()


        elif 'conv' in model_name_detail.casefold():

            model = ConvPDE_Nd(params['nDIM'],params['Nx'],
                               params['data_channel'],
                               params['conv:out_channel'],
                               params['conv:en1_channels'],
                               params['conv:de1_channels'],
                               params['conv:method_nonlinear'],
                               params['conv:method_types_conv'],
                               #params['conv:method_OP'],
                               params['conv:method_skip'],
                               params['conv:bUpSampleOrConvTranspose'],
                               params['conv:method_pool'],
                               #params['conv:method_conv'],
                               params['num_PDEParameters'],
                               params['conv:method_BatchNorm']
                               #params['bExternalConstraint']
                               #,params['yB_1DNormalization']
                               ).cuda()

        print('count_learnable_params=', str( count_learnable_params(model) ) )
        return model

    @staticmethod
    def get_model_name_detail(model_name, data_sys, params):
        nDIM = params['nDIM']
        if 'onet' in model_name.casefold():
            model_name_detail= 'ONet'
        elif 'conv' in model_name.casefold():
            model_name_detail = 'Conv'
        elif 'fourier' in model_name.casefold():
            model_name_detail = 'Fourier'

        if nDIM == 1:
            model_name_detail += '_'
        elif nDIM ==2:
            model_name_detail += '2D_'

        if 'fourier' in model_name.casefold():
            if nDIM ==1:
               model_name_detail  += 'm'+ str( params['fourier:modes_fourier'] )+'w'+str( params['fourier:width'])
            elif nDIM ==2:
               model_name_detail  += 'm' + str(params['fourier:modes_fourier'][0]) + '_' + str(params['fourier:modes_fourier'][1]) + 'w' + str(params['fourier:width'])

        if 'siva' in data_sys.sysname.casefold():
            #if len(nu_Siva)>0:
            nu_str_ = 'nu'
            for nu in data_sys.list_nu:
               nu_str_ += '{:g}'.format(nu)[2:] + '_'
            model_name_detail += nu_str_[:-1]

        elif 'cfd' in data_sys.sysname.casefold():   #len( para_cfdNS) > 0:
            cfdstr_ = 'cfd'
            for filename in data_sys.list_cfdfilename:
               cfdstr_ += filename + '_'
            model_name_detail += cfdstr_[:-1]


        if params['method_TimeAdv']=='gru':    model_name_detail +=  '_gru'
        if params['T_in'] >=2:                 model_name_detail +=  '_Tin' + str(params['T_in'])
        if params['data:nStep'] >=2:           model_name_detail +=  '_nStep' + str(params['data:nStep'])
        if params['num_PDEParameters']>=1:     model_name_detail +=  '_nPara'+ str(params['num_PDEParameters'])
        if params['data_channel']>=2:          model_name_detail += '_dchan'+ str(params['data_channel'])

        if 'fourier' in model_name.casefold():
            if params['fourier:method_Attention']==1:      model_name_detail +=  '_att'
            if params['fourier:method_WeightSharing']==1:  model_name_detail +=  '_share'
            if params['fourier:method_WeightSharing']==2:  model_name_detail +=  '_share2'
            if params['fourier:method_SkipConnection']==1: model_name_detail +=  '_skip'
            if params['fourier:method_BatchNorm']==1:      model_name_detail +=  '_bn'
            if params['fourier:brelu_last']==0:            model_name_detail += '_noLastRelu'

        elif 'conv' in model_name.casefold():

            if params['conv:method_skip'] != 'full':                 model_name_detail += '_skip'+ params['conv:method_skip']
            if  params['conv:en1_channels'] != [ [16],[32,32],[64,64],[128],[128],[64],[32] ] :   # [[16],[32],[64],[64],[64]]:
                mystr = 'e'
                for li in params['conv:en1_channels'] :
                    mystr += '_'
                    for l in li:
                        mystr += str( int(np.log2(l))  )
                model_name_detail += mystr
            if  params['conv:de1_channels'] is not None:   # [[16],[32],[64],[64],[64]]:
                mystr = 'd'
                for li in params['conv:de1_channels'] :
                    mystr += '_'
                    for l in li:
                        mystr += str( int(np.log2(l))  )
                model_name_detail += mystr

            if params['conv:method_types_conv'] != 'conv_all': model_name_detail += ( '_' + params['conv:method_types_conv'] )
            if params['conv:method_nonlinear'] != 'all':       model_name_detail += ('_NonLinear' + params['conv:method_nonlinear'])
            if params['conv:method_BatchNorm']==True:          model_name_detail += '_bn'
        elif 'onet' in model_name.casefold():
            model_name_detail += '_branch' + params['onet:type_branch']
            #if params['onet:type_branch'] !='conv':
            #    model_name_detail += '_branchfc'
            if params['onet:fc_layers_branch'] != [100,100,100,100]:
                model_name_detail +='Br'+''.join( [str(e)+'_' for e in params['onet:fc_layers_branch']  ]  )
            if params['onet:fc_layers_trunk'] != [100,100,100,100]:
                model_name_detail +='Tr'+''.join( [str(e)+'_' for e in params['onet:fc_layers_trunk']  ]  )

            if params['onet:P'] != 30:
                model_name_detail += '_P'+str(params['onet:P'])

            if params['onet:trunk_featurepair'] !=1:
                model_name_detail += '_feature' + str(params['onet:trunk_featurepair'])

            if params['onet:type_trunk'] != 'simple':
                model_name_detail += '_trunkfancy'

            if params['onet:method_skipconnection'] :
                model_name_detail += '_skipconn'
            if params['train:data_norm_rms'] != 1 :
                model_name_detail += '_Norm'

        #if params['bExternalConstraint'] == True:   model_name_detail += '_EC'
        #if params['T_out'] > 1:
        model_name_detail += '_o' + str(params['T_out'])

        print(model_name_detail)

        return model_name_detail


class lib_DataGen:
    @staticmethod
    def print_help():
        print('----- params for DataGen -----')
        print('nDIM,T_in,T_out,Nx,nStep,nStepSkip,data_channel, data_sys.sysname, data_sys.list_nu, data_sys.list_cfdfilename')
        print('------------------------------')

    @staticmethod
    def DataGen(data_sys,params) :

        lib_DataGen.print_help()
        t1 = default_timer()

        if 'siva' in data_sys.sysname:
            sequence_disp, sequence_disp_test, sequence_nu,sequence_nu_test = \
               lib_DataGen.DataGen_siva( data_sys.list_nu, params['T_in'],params['T_out']*params['T_d_out'],nDIM=params['nDIM'],Nx=params['Nx'],
                                         yB_estimate=params['data:yB_estimate'],AspectRatio_set=params['data:AspectRatio_set'],
                                         nStep=params['data:nStep'],nStepSkip=params['data:nStepSkip'],dir_save_training_data=params['data:dir_save_training_data'],
                                         method_default_siva_data_gen=data_sys.method_default_siva_data_gen)

        elif 'cfd' in  data_sys.sysname:
            sequence_disp, sequence_disp_test, sequence_nu,sequence_nu_test = \
              lib_DataGen.DataGen_cfd( params['T_in'],params['T_out'],nDIM=params['nDIM'],Nx=params['Nx'],
                                       yB_estimate=params['data:yB_estimate'],AspectRatio_set=params['data:AspectRatio_set'],
                                       data_channel=params['data_channel'],
                                       nStep=params['data:nStep'],nStepSkip=params['data:nStepSkip'],
                                       list_picklefilename=data_sys.list_cfdfilename)


        if params['nDIM']==2:
            sequence_disp       = np.tanh(sequence_disp)
            sequence_disp_test  = np.tanh(sequence_disp_test)
            print('np.tanh is applied')

        train_disp, test_disp, train_PDEpara,test_PDEpara = \
            lib_DataGen.np_array_To_torch_tensor(sequence_disp, sequence_disp_test,sequence_nu,sequence_nu_test,data_sys,params)

        t2 = default_timer()
        print('preprocessing finished, time used:', t2 - t1)

        return train_disp, test_disp, train_PDEpara, test_PDEpara



    @staticmethod
    def np_array_To_torch_tensor(sequence_disp, sequence_disp_test,sequence_nu,sequence_nu_test,data_sys,params):

        print('sequence_disp.shape, sequence_disp_test.shape,sequence_nu.shape,sequence_nu_test.shape' )
        print( sequence_disp.shape, sequence_disp_test.shape,sequence_nu.shape,sequence_nu_test.shape)

        nDIM =  params['nDIM']
        data_channel = params['data_channel']
        if  nDIM==1 and ('cfd' in data_sys.sysname) :
            sequence_disp       = np.moveaxis(sequence_disp,      1, -2)
            sequence_disp_test  = np.moveaxis(sequence_disp_test, 1, -2)
            #(2965, 2048, 11, 3)
            s = sequence_disp.shape
            train_disp = torch.tensor(sequence_disp.reshape(s[0], s[1], s[2] * s[3]), dtype=torch.float)
            train_PDEpara = torch.tensor(sequence_nu, dtype=torch.float)

            s = sequence_disp_test.shape
            test_disp = torch.tensor(sequence_disp_test.reshape(s[0], s[1], s[2] * s[3]), dtype=torch.float)
            test_PDEpara = torch.tensor(sequence_nu_test, dtype=torch.float)
        else:
            sequence_disp       = np.moveaxis(sequence_disp,      1, -1)
            sequence_disp_test  = np.moveaxis(sequence_disp_test, 1, -1)
            #(20000, 128, 21) in 1D ,  or , (20000, 128, 128, 21)  in 2D
            train_disp = torch.repeat_interleave( torch.tensor(sequence_disp,dtype=torch.float), data_channel, dim=-1 )
            train_PDEpara = torch.tensor(sequence_nu, dtype=torch.float)
            test_disp = torch.repeat_interleave(torch.tensor(sequence_disp_test, dtype=torch.float), data_channel, dim=-1)
            test_PDEpara = torch.tensor(sequence_nu_test, dtype=torch.float)


        print('train_disp.shape, test_disp.shape, train_PDEpara.shape,test_PDEpara.shape')
        print(train_disp.shape, test_disp.shape, train_PDEpara.shape, test_PDEpara.shape)

        return train_disp, test_disp, train_PDEpara, test_PDEpara


    @staticmethod
    def DataGen_siva(nu_Siva, T_in,T_out, nDIM=1, Nx=128,
                      yB_estimate=np.array([-0.7, 1.3])*np.pi, AspectRatio_set=1,
                      nStep=1, nStepSkip=1,
                      dir_save_training_data = './data/',method_default_siva_data_gen=1):

        if nu_Siva not in [ [0.07, 0.125, 0.4, 0.7], [0.07], [0.125], [0.4], [0.7], [0.05] , [0.02], [0.01] ]:
            raise ValueError('DataLoad_Siva, nu_Siva did not found')

        #dir_save_training_data = './data/'

        SivaEq = CSolverSiva(nu_Siva, method_default_siva_data_gen)

        Ny, yB = libSiva.get2D_Ny_yB_from_estimate(Nx, yB_estimate, AspectRatio_set=AspectRatio_set)

        if nDIM==1:
            name_xsol= 'dsol'
        elif nDIM==2:
            name_xsol= 'ylevel'
            print( '2D: Ny_actual=', Ny, 'yB=', yB)

        list_xsol, list_nu           = SivaEq.generate_or_load_DEFAULT_xsol_list('train', dir_save_training_data,
                                                                                 name_xsol=name_xsol, Nx=Nx, yB_estimate=yB,AspectRatio_set=AspectRatio_set)
        list_xsol_test, list_nu_test = SivaEq.generate_or_load_DEFAULT_xsol_list('test' , dir_save_training_data,
                                                                                 name_xsol=name_xsol, Nx=Nx, yB_estimate=yB,AspectRatio_set=AspectRatio_set)
        #print('SivaEq.generate_or_load_DEFAULT_xsol_list')

        #if params['method_TimeAdv'] == 'simple':
        # sequence_disp = libData.Reorg_list_dsol( list_dsol, T_out, T_in )
        sequence_disp, sequence_nu           = libData.Reorg_list_xsol(list_xsol, list_nu,           T_out, T_in, nStep, nStepSkip, name_xsol=name_xsol)
        #print('libData.Reorg_list_xsol')
        sequence_disp_test, sequence_nu_test = libData.Reorg_list_xsol(list_xsol_test, list_nu_test, T_out, T_in, nStep, nStepSkip, name_xsol=name_xsol)
        #print('libData.Reorg_list_xsol')

        #else:  # params['method_TimeAdv'] == 'gru':
        #    #sequence_disp, sequence_nu = libData.Reorg_list_dsol(list_dsol, list_nu, seq_length, T_in)
        #    raise ValueError('Not implemented')

        return sequence_disp, sequence_disp_test, sequence_nu,sequence_nu_test



    def DataGen_cfd( T_in,T_out,
                     nDIM, Nx=128, yB_estimate = np.array([-0.5, 2])*np.pi,AspectRatio_set=1,
                     data_channel=1,
                     nStep=1,nStepSkip=1,
                     cfd_data_dir='./Data_PRE_LaminarFlame/', # '/cephyr/NOBACKUP/groups/ml_flame/siva_fourier_torch19/Data_PRE_LaminarFlame/',
                     list_picklefilename=None ):

        #yB_estimate = np.array([-1, 2.2]) * np.pi
        if list_picklefilename is None:
            #list_picklefilename = ['L512_rho5.pkl','L512_rho8.pkl','L512_rho10.pkl']
            list_picklefilename = ['L512_rho8.pkl']


        if nDIM==1:
            varname = 'y_simple' if data_channel==1 else 'y3'

            list_y, list_p = libcfdData.load_PREdata(list_picklefilename, cfd_data_dir, Nx_target=Nx,varname=varname)

        elif nDIM ==2:
            Ny, yB = libSiva.get2D_Ny_yB_from_estimate(Nx, yB_estimate,AspectRatio_set=AspectRatio_set)

            list_y, list_p = libcfdData.load_2DPREdata(list_picklefilename, cfd_data_dir, Nx, yB, AspectRatio_set=AspectRatio_set)

        #sequence_disp, sequence_nu = libData.Reorg_list_xsol(list_y, list_p, T_out, T_in, nStep, nStepSkip,name_xsol=name_xsol)
        sequence_disp, sequence_nu = libcfdData.Reorg_list_y(list_y, list_p, T_out, T_in, nStep, nStepSkip)


        sequence_disp_test = np.copy(sequence_disp[-1:])
        sequence_nu_test = np.copy(sequence_nu[-1:])

        return sequence_disp, sequence_disp_test,sequence_nu,sequence_nu_test






#----------------------





#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)



class lib_ModelTrain:


    @staticmethod
    def Train(train_disp, test_disp,train_PDEpara,test_PDEpara,
              model,model_name_detail,device,
              params ):

        print('batch_size=', params['train:batch_size'])
        #-------------
        nDIM         = params['nDIM']
        data_channel = params['data_channel']
        T_in         = params['T_in']
        T_out        = params['T_out']
        T_d_out        = params['T_d_out']
        #-------------

        ntrain = train_disp.shape[0]
        ntest = test_disp.shape[0]
        print('ntrain=', ntrain, ' ,ntest=', ntest)


        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_disp[:, ...], train_PDEpara, ),
                                                   batch_size=params['train:batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_disp[:, ...], test_PDEpara, ),
                                                  batch_size=params['train:batch_size'], shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['train:learning_rate'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['train:scheduler_step'], gamma=params['train:scheduler_gamma'])

        myloss = LpLoss(size_average=False)

        ################
        if T_in ==1:
            if nDIM==1:
                train_l2_indentity = myloss(train_disp[..., :T_in*data_channel].repeat(1, 1, T_out // T_in), train_disp[..., T_in*data_channel:])
                test_l2_indentity  = myloss(test_disp[..., :T_in*data_channel].repeat(1, 1, T_out // T_in), test_disp[..., T_in*data_channel:])
            elif nDIM==2:
                train_l2_indentity = myloss(train_disp[..., :T_in*data_channel].repeat(1, 1, 1, T_out // T_in), train_disp[..., T_in*data_channel:])
                test_l2_indentity  = myloss(test_disp[..., :T_in*data_channel].repeat(1, 1,  1, T_out // T_in), test_disp[..., T_in*data_channel:])
        else:
            train_l2_indentity =ntrain
            test_l2_indentity =ntest
        print('train_l2_indentity=', train_l2_indentity / ntrain, ',  test_l2_indentity=', test_l2_indentity / ntest)

        filename_Saved_Model =  params['train:checkpoint_dir'] + '/' + model_name_detail
        # model = torch.load(filename_Saved_Model,map_location=torch.device(run_device))
        ######################################
        # fig, axs = plt.subplots(2,2,figsize=(12, 9))
        #######################################
        list_output_info = []
        for ep in range( params['train:epochs']):
            model.train()

            t1 = default_timer()

            train_l2_step = 0
            train_l2_full = 0

            test_l2_step = 0
            test_l2_full = 0


            # params['method_TimeAdv'].casefold() == 'simple':
            ################
            for  train_a, train_p in train_loader:
                assert train_a.shape[-1] == (T_in + T_out*T_d_out)*data_channel, "train_a.shape[-1]==(T_in+T_out*T_d_out)*data_channel"
                train_a = train_a.to(device)  # train_a.shape[-1]== (T_in+T_out)*data_channel
                train_p = train_p.to(device)  # train_p.shape[-1]== (T_in+T_out)*data_channel
                current_batch_size = train_a.shape[0]

                for idx in range(T_d_out):
                    T_0 = idx*T_out*data_channel

                    x  = train_a[...,T_0                  :T_0+ T_in*data_channel]  # x.shape[-1]== T_in*data_channel
                    yy = train_a[...,T_0+T_in*data_channel:T_0+(T_in+T_out)*data_channel]  # yy.shape[-1]== T_out*data_channel
                    p  = train_p
                    loss = 0
                    for t in range(T_out):
                        y = yy[..., t*data_channel:(t+1)*data_channel]  # y.shape[-1]== 1*data_channel

                        if model.num_PDEParameters ==0:
                            im = model(x)
                        else:
                            im = model(x,p)

                        # im = model(x,gridx, p)                                 #im.shape[-1]== 1*data_channel
                        #im = model(x, p)
                        # if NNmodel=='Conv':
                        #    im = model(x)                                 #im.shape[-1]== 1*data_channel
                        # elif NNmodel == 'ONet':
                        #    im = model(x,gridx)                                 #im.shape[-1]== 1*data_channel

                        # print (t, end=" ")

                        loss += myloss(im.reshape(current_batch_size, -1), y.reshape(current_batch_size, -1))

                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)

                        x = torch.cat((x[..., 1 * data_channel:], im), dim=-1)  # x.shape[-1]== [(T_in-1)+1]*data_channel

                    train_l2_step += loss.item()
                    l2_full = myloss(pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1))

                    # l2_part1 = myloss(pred[...,0::Nray].reshape(current_batch_size, -1), yy[...,0::Nray].reshape(current_batch_size, -1))
                    # l2_part2 = myloss(pred[...,1::Nray].reshape(current_batch_size, -1), yy[...,1::Nray].reshape(current_batch_size, -1))
                    # l2_part3 = myloss(pred[...,2::Nray].reshape(current_batch_size, -1), yy[...,2::Nray].reshape(current_batch_size, -1))
                    # l2_part3 = myloss( (pred[...,1::Nray]*pred[...,2::Nray]).reshape(current_batch_size, -1), (yy[...,1::Nray]*yy[...,2::Nray]).reshape(current_batch_size, -1))
                    # l2_part4 = myloss( (pred[...,1::Nray]*pred[...,3::Nray]).reshape(current_batch_size, -1), (yy[...,1::Nray]*yy[...,3::Nray]).reshape(current_batch_size, -1))
                    # l2_full = l2_part1 + 1* l2_part2 + 1*l2_part3 #+ l2_part4
                    # l2_full = l2_part1

                    train_l2_full += l2_full.item()

                    optimizer.zero_grad()
                    # loss.backward()
                    l2_full.backward()
                    optimizer.step()

                    print('', end='.')
                    #t1_e = default_timer()
                    #print( t1_e-t1, '[s]', end = '\r')

            # validation test
            model.eval()
            with torch.no_grad():
                for test_a, test_p in test_loader:
                    test_a = test_a.to(device)
                    test_p = test_p.to(device)
                    current_batch_size = test_a.shape[0]

                    for idx in range(T_d_out):
                        T_0 = idx * T_out * data_channel

                        x = test_a[..., T_0                  :T_0+T_in * data_channel]
                        yy = test_a[...,T_0+T_in*data_channel:T_0 + ( T_in + T_out) * data_channel]
                        p = test_p
                        loss = 0

                        for t in range(T_out):
                            y = yy[..., t * data_channel:(t + 1) * data_channel]

                            if model.num_PDEParameters == 0:
                                im = model(x)
                            else:
                                im = model(x, p)
                            # if NNmodel == 'Conv':
                            #    im = model(x) #p)
                            # elif NNmodel == 'ONet':
                            #    im = model(x, gridx) #p)

                            loss += myloss(im.reshape(current_batch_size, -1), y.reshape(current_batch_size, -1))

                            if t == 0:
                                pred = im
                            else:
                                pred = torch.cat((pred, im), -1)

                            x = torch.cat((x[..., 1 * data_channel:], im), dim=-1)

                        test_l2_step += loss.item()
                        test_l2_full += myloss(pred.reshape(current_batch_size, -1),
                                               yy.reshape(current_batch_size, -1)).item()

                    ################

            t2 = default_timer()
            scheduler.step()
            print('')

            # -----------------------
            if ep == 0:
                err_normalizer_ep0 = (train_l2_full / ntrain)
                output_dict = {0: 'ep', 1: 't[s]', 2: 'trainErr_norm', 3: 'testErr_norm', 4: 'train_l2', 5: 'test_l2',
                               6: 'train_l2_step', 7: 'test_l2_step', 8:'train_rel_ind', 9:'test_rel_ind'}
                for key, value in output_dict.items():
                    print(value, end=' ')
                print('')

            output_info = (ep,
                           t2 - t1,
                           (train_l2_full / ntrain) / err_normalizer_ep0,
                           (test_l2_full / ntest) / err_normalizer_ep0,
                           train_l2_full / ntrain,
                           test_l2_full / ntest,
                           train_l2_step / ntrain / T_out,
                           test_l2_step / ntest / T_out,
                           train_l2_full / train_l2_indentity,
                           test_l2_full / test_l2_indentity
                           )
            list_output_info.append(output_info)
            print('%d, %4.2f, %4.3f, %4.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f' % output_info)
            # ---------------------------
            output_dict['list_output_info'] = list_output_info
            # -----------------------
            save_train_log(filename_Saved_Model, output_dict)

            #if ep % 50 == 0:
            #    model.eval()
            #    with torch.no_grad():
            #        update_plot(train_a, 'NNmodel', current_batch_size, gridx, model)

            if ep %  params['train:epochs_per_save']  == 0 and ep > 0:
                filename_Saved_Model_ep = filename_Saved_Model+'_ep' + str(ep)
                print(filename_Saved_Model_ep)
                torch.save(model, filename_Saved_Model_ep)


            if True or output_info[1] > 30: # more than 30 sec
                if  np.argmin( np.array(list_output_info)[:,2]  ) == ep :
                    torch.save(model, filename_Saved_Model + '_best' )

        # ---------------------------
        print(filename_Saved_Model)
        torch.save(model, filename_Saved_Model)
        #---------

        # retreived_list_output_info = pickle.load(open('trainlog.dump', 'rb'))

def save_train_log(filename_Saved_Model,output_dict):
    open_file = open(filename_Saved_Model + 'trainlog.pkl', 'wb')
    pickle.dump(output_dict, open_file)
    open_file.close()
