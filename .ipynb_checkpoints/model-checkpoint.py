import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSEA(nn.Module):
    def __init__(self):
        super(DeepSEA, self).__init__()
        
        self.batch_first=True
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(53*960, 925)
        self.fc2 = nn.Linear(925, 919)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.drop1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = x.view(-1, 53*960)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x

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
    
    
class RNN(nn.Module):
    def __init__(self,
                input_size=4,
                hidden_layers = 2,
                hidden_size = 32,
                
                is_bidirectional = True,
                
                output_size = 919,
                
                dropout_keep = 0.8,
                ):
        
        super(RNN, self).__init__()
        
        self.batch_first=False
        
        # Embedding Layer
        #self.embeddings = nn.Embedding(vocab_size,embed_size)
        #self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = hidden_layers,
                            #batch_first=True,
                            dropout = dropout_keep,
                            bidirectional = is_bidirectional)
        
        self.dropout = nn.Dropout(dropout_keep)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            hidden_size * hidden_layers * (1+is_bidirectional),
            output_size
        )
        
        
    def forward(self, x):        
        #embedded_sent = self.embeddings(x)

        lstm_out, (h_n,c_n) = self.lstm(x) 
        # (seq_len, batch, embed_size) ->
        # (batch, seq_len, num_directions * hidden_size),
        # ((num_layers * num_directions, batch, hidden_size),
        # (num_layers * num_directions, batch, hidden_size))
        
        final_feature_map = self.dropout(h_n)
        # (num_layers * num_directions, batch, hidden_size)

        final_feature_map = torch.cat([final_feature_map[i,:,:] for i in range(final_feature_map.shape[0])], dim=1)
        # (num_layers * num_directions, batch, hidden_size) ->
        # (batch, hidden_layers * num_directions * hidden_size)

        final_out = self.fc(final_feature_map)
        # (batch, hidden_layers * num_directions * hidden_size) -> (output_size)
  
        return final_out
 
