import torch

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import Linear as Lin, ReLU, Sigmoid, ConstantPad1d
from torch.utils.data import Dataset, DataLoader

class DriverDatasetCase(Dataset):
    def __init__(self,latent=False):
            self.M = M
            self.dMdt = dMdt
            self.M1 = M1
            self.dM1dt = dM1dt
            self.M2 = M2
            self.dM2dt = dM2dt      
            self.bin0 = bin0
            self.bin1coal = bin1coal
            self.bin1condevap = bin1condevap
            self.binmag = binmag
            self.maskall = maskall
    def __len__(self):
        return int(self.M.shape[0])
    def __getitem__(self,idx):

        idx0 = idx

        M = self.M[idx0,:]
        dMdt = self.dMdt[idx0,:]
        bin0 = self.bin0[idx0,:,:]
        bin1coal = self.bin1coal[idx0,:,:]
        bin1condevap = self.bin1condevap[idx0,:]
        binmag = self.binmag[idx0,:]
        
        M1 = self.M1[idx0,:]
        dM1dt = self.dM1dt[idx0,:]
        M2 = self.M2[idx0,:]
        dM2dt = self.dM2dt[idx0,:]        
        
        return M,dMdt,bin0,bin1coal,bin1condevap,binmag,M1,dM1dt,M2,dM2dt
class DriverDatasets(Dataset):
    def __init__(self,purpose,latent=False):

        if purpose == "train":
            self.M = M_train
            self.dMdt = dMdt_train
            self.M1 = M1_train
            self.dM1dt = dM1dt_train
            self.M2 = M2_train
            self.dM2dt = dM2dt_train         
            self.bin0 = bin0_train
            self.bin1coal = bin1coal_train
            self.bin1condevap = bin1condevap_train
            self.binmag = binmag_train
        elif purpose == "val":
            self.M = M_val
            self.dMdt = dMdt_val
            self.M1 = M1_val
            self.dM1dt = dM1dt_val
            self.M2 = M2_val
            self.dM2dt = dM2dt_val  
            
            self.bin0 = bin0_val
            self.bin1coal = bin1coal_val
            self.bin1condevap = bin1condevap_val
            self.binmag = binmag_val
        else:
            self.M = M_test
            self.dMdt = dMdt_test
            
            self.M1 = M1_test
            self.dM1dt = dM1dt_test
            self.M2 = M2_test
            self.dM2dt = dM2dt_test  
 
            self.bin0 = bin0_test
            self.bin1coal = bin1coal_test
            self.bin1condevap = bin1condevap_test
            self.binmag = binmag_test
    def __len__(self):
        return int(self.M.shape[0])
    def __getitem__(self,idx):

        idx0 = idx

        M = self.M[idx0,:]
        dMdt = self.dMdt[idx0,:]
        bin0 = self.bin0[idx0,:,:]
        bin1coal = self.bin1coal[idx0,:,:]
        bin1condevap = self.bin1condevap[idx0,:]
        binmag = self.binmag[idx0,:]
        
        M1 = self.M1[idx0,:]
        dM1dt = self.dM1dt[idx0,:]
        M2 = self.M2[idx0,:]
        dM2dt = self.dM2dt[idx0,:]        
        
        return M,dMdt,bin0,bin1coal,bin1condevap,binmag,M1,dM1dt,M2,dM2dt

class CNNEncoderAE(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=35,n_latent=10):
        super(CNNEncoderAE, self).__init__()
        self.n_bins = n_bins
        self.conv1 = Conv1d(in_channels=2,out_channels=4,kernel_size=4,stride=2,padding=1)
        self.activation1 = ReLU()
        self.conv2 = Conv1d(in_channels=4,out_channels=8,kernel_size=4,stride=2,padding=1)
        self.activation2 = ReLU()
        self.conv3 = Conv1d(in_channels=8,out_channels=4,kernel_size=4,stride=2,padding=1)
        self.activation3 = ReLU()
        self.lin = Lin(16,n_latent)
        
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        
    def forward(self,x):
        n_bins = self.n_bins
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = x.view(-1,16)
        
        return self.lin(x)

class CNNDecoder(torch.nn.Module):
    def __init__(self,n_channels=4,n_bins=35,n_latent=10,n_hidden=50):
        super(CNNDecoder, self).__init__()

        self.n_latent = n_latent
        self.n_channels = n_channels

        self.n_bins = n_bins
        self.lin = Lin(n_latent,16)
        self.conv1 = ConvTranspose1d(in_channels=n_channels,out_channels=n_channels*2,kernel_size=4,stride=2,padding=1)
        self.activation1 = ReLU()
        self.constantpad1d1 = ConstantPad1d((1,0),0)
        self.conv2 = ConvTranspose1d(in_channels=n_channels*2,out_channels=n_channels,kernel_size=4,stride=2,padding=1)
        self.activation2 = ReLU()
        self.constantpad1d2 = ConstantPad1d((1,0),0)
        self.conv3 = ConvTranspose1d(in_channels=n_channels,out_channels=2,kernel_size=4,stride=2,padding=1)
        self.activation3 = ReLU()
        self.lin2 = Lin(n_bins,n_bins)
        self.activation4 = Sigmoid()
        

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.normal_(self.lin2.weight,mean=0.0,std=0.01)       
        
    def forward(self,x):
        n_bins = self.n_bins
        inp = x
        bs = x.size(0)
        x = self.lin(inp)

        x = x.reshape(-1,4,4)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.constantpad1d1(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.constantpad1d2(x)
        x = self.lin2(x)
        
        x = self.activation4(x) 

        return x
    
class MicroAutoEncoder(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=35,n_latent=10):
        super(MicroAutoEncoder, self).__init__()

        self.encoder = CNNEncoderAE(n_channels=n_channels,n_bins=n_bins,n_latent=n_latent)
        self.decoder = CNNDecoder(n_channels=n_channels*2,n_bins=n_bins,n_latent=n_latent)

    def forward(self,x):

        bs = x.shape[0]

        latent = self.encoder(x)

        reconstruction = self.decoder(latent) 

        return reconstruction,latent