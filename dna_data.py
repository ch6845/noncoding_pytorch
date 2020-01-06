import scipy.io
import torch
from torch.utils.data import Dataset

class DeepSEA_Dataset(Dataset):
    """DeepSEA DNA sequence dataset."""
    def __init__(self, filepath, key_X,key_Y):
        """
        Args:
            filepath (string): Path to the csv file with annotations.
            key_X (string): Directory with all the images.
            key_Y (string): Directory with all the images.
        """
        self.ispt=(filepath.split('.')[-1]=='pt')
        
        if self.ispt:
            self.f=torch.load(filepath)
            self.data_X=self.f[key_X].transpose()
            self.data_Y=self.f[key_Y].transpose()
        else:
            self.f=scipy.io.loadmat(filepath)
            self.data_X=self.f[key_X]
            self.data_Y=self.f[key_Y]
        
        #print(self.data_X.shape,self.data_Y.shape)
        if self.data_X.shape[0]!=self.data_Y.shape[0]:
            raise

    def __len__(self):
        return self.data_X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.FloatTensor(self.data_X[idx]),torch.FloatTensor(self.data_Y[idx])
    
