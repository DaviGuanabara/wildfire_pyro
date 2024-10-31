import torch.nn as nn

class SpatialRegressor3(nn.Module):
    def __init__(self, hidden=32, features=4, prob=0.5):
        super(SpatialRegressor3, self).__init__()

        self.features = features
        self.hidden = hidden

        self.linear1_before_aggr = nn.Linear(self.features, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=hidden)
        self.activation1_before_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        self.dropout1_aggr = nn.Dropout(p=prob)

        self.linear12_before_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn12 = nn.BatchNorm1d(num_features=hidden)
        self.activation12_before_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        self.dropout12_aggr = nn.Dropout(p=prob)

        self.linear2_before_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=hidden)
        self.activation2_before_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh()

        ##############

        self.linear1_attn = nn.Linear(self.features, hidden, bias=False)
        self.bn1_attn = nn.BatchNorm1d(num_features=hidden)
        self.activation1_attn = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        self.dropout1_attn = nn.Dropout(p=prob)

        self.linear12_attn = nn.Linear(hidden, hidden, bias=False)
        self.bn12_attn = nn.BatchNorm1d(num_features=hidden)
        self.activation12_attn = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        self.dropout12_attn = nn.Dropout(p=prob)

        self.linear2_attn = nn.Linear(hidden, hidden, bias=False)
        self.bn2_attn = nn.BatchNorm1d(num_features=hidden)

        self.softmax = nn.Softmax(dim=1) # softmax in the neighbors dim
        ################

        self.dropout1 = nn.Dropout(p=prob)
        self.linear1_after_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=hidden)
        self.activation1_after_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()

        self.dropout12 = nn.Dropout(p=prob)
        self.linear12_after_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn12 = nn.BatchNorm1d(num_features=hidden)
        self.activation12_after_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()

        self.dropout2 = nn.Dropout(p=prob)
        self.linear2_after_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn4 = nn.BatchNorm1d(num_features=hidden)
        self.activation2_after_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh()

        self.bn5 = nn.BatchNorm1d(num_features=hidden)
        self.dropout3 = nn.Dropout(p=prob)
        self.linear2_regression = nn.Linear(hidden, 1)

    def forward(self, u, mask):
        # u: (batch, neighbors, features)
        # example: (2, 3, 4)

        # mask: (batch, neighbors, features)
        # example: (2, 3, 4)

        # ------- pass the input features from the set of neighbots in the MLP-theta,
        # responsible for calculating the attention weights

        batch_size = u.shape[0]

        # (batch * neighbors, features)
        input = u.view(-1, self.features)
        
        # (batch * neighbors, hidden)

        x_main = self.linear1_before_aggr(input)
        x_main = self.bn1(x_main)
        self.x_main = self.activation1_before_aggr(x_main)

        x_main_1 = self.dropout1_aggr(self.x_main)
        x_main_1 = self.linear12_before_aggr(x_main_1)
        x_main_1 = self.bn12(x_main_1)
        self.x_main_1 = self.activation12_before_aggr(x_main_1)

        x_main = x_main + self.x_main_1

        x_main_2 = self.dropout12_aggr(x_main)
        x_main_2 = self.linear2_before_aggr(x_main_2)
        x_main_2 = self.bn2(x_main_2)
        self.x_main_2 = self.activation2_before_aggr(x_main_2)

        x_main = x_main + self.x_main_2

        # (batch, neighbors, hidden)
        x_main = x_main.view(batch_size, -1, self.hidden)
   
        # attention leg

        # (batch * neighbors, features)
        x_attn = self.linear1_attn(input)
        x_attn = self.bn1_attn(x_attn)
        self.x_attn = self.activation1_attn(x_attn) #nn.ReLU()

        x_attn_1 = self.dropout1_attn(self.x_attn)
        x_attn_1 = self.linear12_attn(x_attn_1)
        x_attn_1 = self.bn12_attn(x_attn_1)
        self.x_attn_1 = self.activation12_attn(x_attn_1) #nn.ReLU()

        x_attn = x_attn + self.x_attn_1

        # (batch * neighbors, hidden)
        x_attn = self.dropout12_attn(x_attn)
        x_attn = self.bn2_attn(x_attn)
        x_attn = self.linear2_attn(x_attn)

        # (batch, neighbors, hidden)
        x_attn = x_attn.view(batch_size, -1, self.hidden)

        # (batch, neighbors, hidden)
        x_attn = x_attn.masked_fill(mask==0, -float("inf"))

        # (batch, neighbors, features)
        x_attn = self.softmax(x_attn) # softmax in the neighbors dim

        x = x_main * x_attn # element wise multiplication by the weights

        # (batch, features)
        x = x.sum(dim=1)

        # ------- pass the aggregated node vetor in the MLP-phi, which is responsible for the regression
        #x = self.dropout1(x)

        # (batch, features)
        x0 = self.dropout1(x)
        x0 = self.linear1_after_aggr(x0)
        x0 = self.bn3(x0)
        self.x0 = self.activation1_after_aggr(x0)
        
        x = x + self.x0

        x_1 = self.dropout12(x)
        x_1 = self.linear12_after_aggr(x_1)
        x_1 = self.bn12(x_1)
        self.x_1 = self.activation12_after_aggr(x_1)

        x = x + self.x_1

        x_2 = self.dropout2(x)
        x_2 = self.linear2_after_aggr(x_2)
        x_2 = self.bn4(x_2)
        self.x_2 = self.activation2_after_aggr(x_2)

        x = x + self.x_2

        x = self.bn5(x)
        x = self.dropout3(x)
        x = self.linear2_regression(x)

        return x
