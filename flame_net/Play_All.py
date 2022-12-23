
import numpy
import torch
import torch.nn as nn



import matplotlib.pyplot as plt

from lib_uti import count_learnable_params
from libSiva import *

from uti_plot import * 

import paramiko
def Load___remote_or_local_model(  local_checkpoint_dir ,   filename_Saved_Model ):
    LocalFull_filename_Saved_Model = local_checkpoint_dir + filename_Saved_Model
    if not os.path.isfile(LocalFull_filename_Saved_Model):
        ssh_client=paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname='alvis1.c3se.chalmers.se',username='rixin',password='ycnPxAhn%t0')
        ftp_client=ssh_client.open_sftp()
        remote_checkpoints_folder ='/cephyr/users/rixin/Alvis/WorkProjects/ml_flame/siva_fourier_torch19/checkpoints'
        #remote_checkpoints_folder ='/cephyr/NOBACKUP/groups/ml_flame/siva_fourier_torch19/checkpoints'
        ftp_client.get( remote_checkpoints_folder+filename_Saved_Model ,  LocalFull_filename_Saved_Model)
        ftp_client.close()
        ssh_client.close()
#%matplotlib

np.random.seed(seed=4513)
############################3
#SivaEq = CSolverSiva('nuMulti_0noise')
SivaEq = CSolverSiva( [0.07, 0.125,0.4,0.7] )
#SivaEq = CSolverSiva( [0.05] )

N = SivaEq.N

#model_name_detail,bRNN,T_in = 'Fourier_nuMulti_0noise_', False,1
#model_name_detail,bRNN,T_in = 'Fourier_nu07_125_4_7', False,1

model_name_detail,bRNN,T_in = 'Fourier_nu07_125_4_7_att_share_skip', False,1
modes,width = 64,15
filename_Saved_Model = '\\' + model_name_detail  +  '_m' + str(modes) + '_w' + str(width) + '_ep800'



#model_name_detail='Fourier_m64w15nu07_125_4_7_nPara1_skip'
#model_name_detail = 'Fourier_m32w20nu07_125_4_7_nPara1_att_share_skip'
#model_name_detail = 'Fourier_m64w15nu05_att_share_skip'
#filename_Saved_Model = '\\' + model_name_detail

#model_name_detail = 'Fourier_m32w20nu05_att_share_skip'
#model_name_detail = 'Fourier_m64w15nu05_att_share_skip'

#model_name_detail = 'Fourier_m32w20nu07_att_share_skip'
#model_name_detail = 'Fourier_m32w20nu07'
#modes,width = 64,15
#modes,width = 65,15
#modes,width = 30,20
#num_PDEParameters = 1

# ----- 2D 2D 2D 2D 2D 2D 2D 2D 2D --------
#model_name_detail = 'Fourier2D_m32_32w20nu07_share_skip'

#model_name_detail = 'Fourier2D_m15_15w10nu07_share_skip_ep700'

#model_name_detail = 'Fourier2D_m32_32w10cfdtestNoJump_share_skip_ep200'
#model_name_detail = 'Fourier2D_m32_32w10cfdtest_share_skip_ep800'

#model_name_detail = 'Fourier2D_m32_32w10nu07_share_skip_ep300'



#model_name_detail = 'Fourier_m32w20cfdtest_nPara1_dchan3_att_share_skip_ep800'
#model_name_detail = 'Fourier_m32w20cfdtest_nPara1_att_share_skip_ep800'

bRNN=False
T_in=1



#filename_Saved_Model = '\\' + model_name_detail  +  '_m' + str(modes) + '_w' + str(width) + '_ep900'


#filename_Saved_Model = '/' + model_name_detail  +  '_m' + str(modes) + '_w' + str(width) + '_ep900'

#if num_PDEParameters>=1:
#    filename_Saved_Model = filename_Saved_Model + '_nPDEPara'+str(num_PDEParameters)
#filename_Saved_Model  = filename_Saved_Model  +  '_ep300'
#filename_Saved_Model  = filename_Saved_Model  +  '_ep700'
print(filename_Saved_Model)

data_channel = 1
nu_check = 0.07

bEnable_Realtimeplot = True
################################
#%matplotlib

#torch.manual_seed(0)
#np.random.seed(0)


local_checkpoint_dir = F"d:\\Work\\00_MLproj\\siva_fourier_torch19\\downloaded_checkpoints"
LocalFull_filename_Saved_Model =  local_checkpoint_dir + filename_Saved_Model

Load___remote_or_local_model(  local_checkpoint_dir ,   filename_Saved_Model )

##################


#from My_Class import *
model = torch.load( LocalFull_filename_Saved_Model ,map_location='cpu')

#print('count_learnable_params =' + str( model.count_learnable_params() ))
count_learnable_params(model)

device = torch.device('cpu')

#ani = libPlot.anim_singleCmp_model_PRE(2048, model, init_str='rand_FFT', nu=10., numTotalTimeStep=1550,  ylim=(-2,5) )
#ani

#ani= libPlot.anim_realtimeCmp_model_analytical(SivaEq,model,nu_check=nu_check, nrows=1,ncols=1)


#ani= libPlot.animlevel2D_realtimeCmp_model_analytical(SivaEq,model,nStep=1,nu_check=nu_check,  nReset=500, nrows=1,ncols=1,method_levelset='ylevel')



#ani= libPlot.animlevel2D_realtimeCmp_model_analytical(SivaEq,model,nStep=1,nu_check=nu_check,  nReset=500, nrows=1,ncols=1,method_levelset='ylevel',bTrainedByTanh=True)
#ani.save('Siva2D.gif', writer='imagemagick', fps=10)


Nx, Ny, yB = 512, 819, np.array( [-1, 2.1953125])*np.pi
##############
#y = np.linspace(yB[0],yB[1],Ny); dy = (yB[1]-yB[0])/Ny
#yleveltanh_ref = np.tanh(-y/dy/(Nx/20) )
yleveltanh_ref=0

#nStep=1 ; nu_check=nu_check  ; nReset=10 ; nrows=1 ; ncols=1 ; method_levelset='ylevel';  bTrainedByTanh=True ; yRef0= yleveltanh_ref

##############
#ani=libPlot.animlevel2D_realtime_model(model, Nx, Ny, yB, nStep=1, nSkipStep_plot=1, nReset=1000, nrows=1, ncols=1, method_levelset='ylevel', bTrainedByTanh=True, yRef0= yleveltanh_ref )
#ani= libPlot.animlevel2D_realtimeCmp_model_analytical(SivaEq,model,nStep=20,nu_check=nu_check,  nReset=30, nrows=1,ncols=1,method_levelset='ylevel',bTrainedByTanh=False)
################

#ani= libPlot.anim_realtimeCmp_model_analytical(SivaEq,model,nu_check=nu_check, nrows=2,ncols=2)
#
ani=libPlot.anim_singleCmp_model_analytical(SivaEq,model,nu=nu_check,numTotalTimeStep=1000,init_str='rand_simple')
#

#ani= libPlot.anim_realtimeCmp_model_analytical(SivaEq,model,nu_check=nu_check, nrows=2,ncols=2)

#model.bEnableResid = False
#model.conv1.bEnalbeAttention =False
#model.conv1.depth_SpectralConv =6


#from matplotlib.animation import FuncAnimation, PillowWriter

#ani = animation.FuncAnimation(fig, update_2animation_flame, loop_length-1, fargs=(truth_plot, pred,pred_nohidden), interval=10,blit=False)
#f = "seem_nu4.gif"
#writergif  = animation.PillowWriter(fps=10)
#ani.save(f, writer=writergif )

#libPlot.Take_4RandInit_plot_Asequence(SivaEq, model, nu=0.07, numTotalTimeStep=1000)




