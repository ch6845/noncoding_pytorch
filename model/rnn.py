import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # (seq_len, batch, num_directions * hidden_size),
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
 
