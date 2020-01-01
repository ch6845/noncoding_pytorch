import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,
                in_channels=4,
                out_channels=100,
                output_size = 919,
                kernel_size_list=[5,6,7,8,9,10,11,12,13,14,15],
                dropout_keep = 0.8
                ):
        super(CNN, self).__init__()
        
        self.batch_first=True
        
        self.conv_list=nn.ModuleList([
                            nn.Sequential(
                                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                                nn.ReLU(),
                                nn.MaxPool1d(1000 - kernel_size+1)
                            )
                            for kernel_size in kernel_size_list
                            ]
                        )
        
        self.dropout = nn.Dropout(dropout_keep)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(out_channels*len(kernel_size_list), output_size)
        
        
    def forward(self, x):

        # (batch_size,embed_size,len)->(len(kernel_size_list),batch_size,out_channels)
        conv_out_list=[conv(x).squeeze(2) for conv in self.conv_list]
        
        all_out = torch.cat(conv_out_list, 1)
        final_feature_map = self.dropout(all_out)
        
        final_out = self.fc(final_feature_map)
        
        return final_out