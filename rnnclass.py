import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle


class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "rnnclass"
        return super().find_class(module, name)
        
        
        
        
class PredicterRNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_size, batch_size, num_layers):
            super(PredicterRNN, self).__init__()
    
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.batch_size = batch_size
            self.dropout_layer = nn.Dropout(p=0.2)
            
            ## 
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
            self.dense1 = nn.Linear(hidden_dim, int(hidden_dim/2))
            self.dense2 = nn.Linear(int(hidden_dim/2), output_size)
        
        def forward(self, input):
            input = input.view(-1, 25, 16)
            # print(input.size())
            # .view(state_dim, self.batch_size, -1))
            lstm_out, _ = self.lstm(input)

            output_space = self.dense1(lstm_out.view(self.batch_size, -1))
            output_space = torch.tanh(output_space)
            #output_space = self.dropout_layer(output_space)
            output_space = self.dense2(output_space)
            output_space = torch.tanh(output_space)
            output_scores = F.log_softmax(output_space, dim=1)
            return output_scores