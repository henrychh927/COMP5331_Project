import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
        def __init__(self, num_date, num_feature, hidden_size):
            super(MLP, self).__init__()
            self.num_date = num_date
            self.hidden_size  = hidden_size
            self.num_feature = num_feature
            self.fc0 = nn.Linear(self.num_feature, 1)
            self.fc1 = nn.Linear(self.num_date, self.hidden_size)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
            self.initial_portfolio = nn.Linear(self.hidden_size, 1)
            self.short_sale = nn.Linear(self.hidden_size, 1)
            self.reinvestment = nn.Linear(self.hidden_size, 1)
            
        def forward(self, x):
            # x: b, num_stock, date, feature
            
            x = self.fc0(x) # x: b, num_stock, date, 1
            x = x.squeeze(-1) # x: b, num_stock, date
            x = self.fc1(x) # x: b, num_stock, hidden_size
            x = self.relu(x) # x: b, num_stock, hidden_size
            a = self.initial_portfolio(x) # x: b, num_stock, 1
            a_s = self.short_sale(x) # x: b, num_stock, 1
            a_r = self.reinvestment(x) # x: b, num_stock, 1
            
            a = F.sigmoid(a) 
            a_s = F.sigmoid(a_s)
            a_r = F.sigmoid(a_r)
            
            # b, num_stock, 1
            return a-a_s+a_r
        
        
        
        
class LSTM(nn.Module):
    def __init__(self, num_feature, hidden_size, n_layer, dropout=0.1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.emb = nn.Linear(num_feature, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layer, dropout=dropout, batch_first=True)
        
        self.initial_portfolio = nn.Linear(self.n_layer*self.hidden_size, 1)
        self.short_sale = nn.Linear(self.n_layer*self.hidden_size, 1)
        self.reinvestment = nn.Linear(self.n_layer*self.hidden_size, 1)
        


    def forward(self, x):
         # x: b, num_stock, date, feature
        
        b, num_stock, date, feature = x.size()
        
        x = self.drop(self.emb(x))  # x: b, num_stock, date, hidden_size
        x = x.view(-1, date, self.hidden_size)  # x: b*num_stock, date, hidden_size
        _, (h, _) = self.lstm(x) #n_layer, b*num_stock, hidden_size
        x = h.permute((1, 0, 2)).reshape(b, num_stock, self.n_layer*self.hidden_size)   #b, num_stock, n_layer*hidden_size
        x = self.drop(x) 
        
        a = self.initial_portfolio(x) # x: b, num_stock, 1
        a_s = self.short_sale(x) # x: b, num_stock, 1
        a_r = self.reinvestment(x) # x: b, num_stock, 1
            
        a = F.sigmoid(a) 
        a_s = F.sigmoid(a_s)
        a_r = F.sigmoid(a_r)
        
        # b, num_stock, 1.num_feature, self.1)
        return a-a_s+a_r

    
    
    
    
class CNN(nn.Module):
    def __init__(self, num_feature, hidden_size, dropout=0.1, filter_sizes=[3, 4, 5], num_feature_maps=100):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_feature = num_feature
        self.dropout = nn.Dropout(dropout)
        self.filter_sizes = filter_sizes
        self.num_feature_maps = num_feature_maps
        
        self.emb = nn.Linear(self.num_feature, self.hidden_size)
        
        self.conv = nn.ModuleList([ 
            nn.Sequential(
                # input size: batch, in_channels, seq_length, embedding_size
                # output size: batch, out_channels, seq_length-filter_size+1, 1
                nn.Conv2d(in_channels=1, out_channels=self.num_feature_maps, kernel_size=(i, self.hidden_size)),
                nn.ReLU(),
                ) 
            for i in self.filter_sizes])
                
        self.initial_portfolio = nn.Linear(self.num_feature_maps*len(self.filter_sizes), 1)
        self.short_sale = nn.Linear(self.num_feature_maps*len(self.filter_sizes), 1)
        self.reinvestment = nn.Linear(self.num_feature_maps*len(self.filter_sizes), 1)

    def forward(self, x):
        # x: b, num_stock, date, feature
        
        b, num_stock, date, feature = x.size()
        x = self.emb(x)   # x: b, num_stock, date, hidden_size
        x = x.view(-1, date, self.hidden_size)   # x: b*num_stock, date, hidden_size
        x = [conv(x.unsqueeze(1)).squeeze() for conv in self.conv]

        x = [F.max_pool1d(i.squeeze(-1), kernel_size=i.size(-1)).squeeze(-1) for i in x ]

        # x: batch, num_feature_maps & num_filters
        x = torch.cat(x, -1).view(b, num_stock, -1)  # x: b, num_stock, self.num_feature_maps*len(self.filter_sizes)
        x = self.dropout(x)
                         
        a = self.initial_portfolio(x) # x: b, num_stock, 1
        a_s = self.short_sale(x) # x: b, num_stock, 1
        a_r = self.reinvestment(x) # x: b, num_stock, 1

        a = F.sigmoid(a) 
        a_s = F.sigmoid(a_s)
        a_r = F.sigmoid(a_r)

        # b, num_stock, 1
        return a-a_s+a_r
    
            
          
        

