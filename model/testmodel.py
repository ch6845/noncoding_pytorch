import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqAttention(nn.Module):
    def __init__(self,
                input_size=4,
                hidden_layers = 2,
                hidden_size = 32,
                
                is_bidirectional = True,
                
                output_size = 919,
                
                dropout_keep = 0.8,
                ):
        
        super(Seq2SeqAttention, self).__init__()
        
        self.batch_first=False
        
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = hidden_layers,
                            #batch_first=True,
                            dropout = dropout_keep,
                            bidirectional = is_bidirectional)
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout_keep)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            hidden_size  * (1+is_bidirectional) * 2,
            output_size
        )
        
        
    def forward(self, x):        
        #embedded_sent = self.embeddings(x)

        lstm_out, (h_n,c_n) = self.lstm(x) 
        # (seq_len, batch, embed_size) ->
        # (seq_len, batch, num_directions * hidden_size),
        # ((hidden_layers * num_directions, batch, hidden_size),
        # (hidden_layers * num_directions, batch, hidden_size))
        
        final_feature_map = self.dropout(h_n)
        # (hidden_layers * num_directions, batch, hidden_size)

        final_feature_map = torch.cat([final_feature_map[i,:,:] for i in range(final_feature_map.shape[0])], dim=1)
        # (hidden_layers * num_directions, batch, hidden_size) ->
        # (batch, hidden_layers * num_directions * hidden_size)

        final_out = self.fc(final_feature_map)
        # (batch, hidden_layers * num_directions * hidden_size) -> (output_size)
  
        return final_out


class TestModel(nn.Module):
    def __init__(self,
                input_size=4,
                hidden_layers = 3,
                hidden_size = 32,
                
                is_bidirectional = True,
                
                output_size = 919,
                
                dropout_keep = 0.8,
                ):
        
        super(TestModel, self).__init__()
        
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
 

class TestModel2(nn.Module):
    def __init__(self,
                in_channels=4,
                out_channels_list=[100,200,400],
                output_size = 919,
                kernel_size_list=[5,6,7,8,9,10,11,12,13,14,15],
                dropout_keep = 0.8
                ):
        
        super(TestModel2, self).__init__()
        
        self.batch_first=True
        
        self.conv_list=nn.ModuleList([
                            nn.Sequential(
                                nn.Conv1d(in_channels=in_channels, out_channels=out_channels_list[0], kernel_size=kernel_size),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=int(kernel_size/2), stride=int(kernel_size/2)),
                                nn.Dropout(dropout_keep),
                                
                                nn.Conv1d(in_channels=out_channels_list[0], out_channels=out_channels_list[1], kernel_size=kernel_size),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=int(kernel_size/2), stride=int(kernel_size/2)),
                                nn.Dropout(dropout_keep),
                                
                                nn.Conv1d(in_channels=out_channels_list[1], out_channels=out_channels_list[2], kernel_size=kernel_size),
                                nn.ReLU(),
                                nn.Dropout(dropout_keep),
                            )
                            for kernel_size in kernel_size_list
                            ]
                        )
        
        # Fully-Connected Layer
        #self.fc2 = nn.Linear(out_channels*len(kernel_size_list), output_size)
        #self.fc2 = nn.Linear(out_channels*len(kernel_size_list), output_size)
        
        
    def forward(self, x):

        # (batch_size,embed_size,len)->(len(kernel_size_list),batch_size,out_channels)
        conv_out_list=[conv(x) for conv in self.conv_list]

        conv_out_list_cat = torch.cat(conv_out_list, 1)
        print(conv_out_list_cat.shape)
        final_out = self.fc(final_feature_map)
        
        
        x = x.view(-1, 53*960)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        
        return final_out