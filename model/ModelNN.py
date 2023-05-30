import torch
import torch.nn as nn

# neural network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
    
    def forward(self, x):
        pass # placeholder for subclass implementation
    
class OneLayer(Net):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(OneLayer, self).__init__(input_size, hidden_size, output_size, dropout_rate)      
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) # dropout to prevent overfitting
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() # output to probability rather than bool
        
        # Initilize weights with Glorot
        # UNIFORM
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
       
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc_out(out)
        out = self.sigmoid(out)      

        return out
    
    
    
class TwoLayer(Net):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(TwoLayer, self).__init__(input_size, hidden_size, output_size, dropout_rate)      
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) # dropout to prevent overfitting
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) # dropout to prevent overfitting
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() # output to probability rather than bool
        
        # Initilize weights with Glorot
        # UNIFORM
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc_out(out)
        out = self.sigmoid(out)      

        return out
    
    
class ThreeLayer(Net):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(ThreeLayer, self).__init__(input_size, hidden_size, output_size, dropout_rate)      
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) # dropout to prevent overfitting
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) # dropout to prevent overfitting
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate) # dropout to prevent overfitting
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() # output to probability rather than bool
        
        # Initilize weights with Glorot
        # UNIFORM
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        nn.init.xavier_uniform_(self.fc_out.weight)
        
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc_out(out)
        out = self.sigmoid(out)      

        return out