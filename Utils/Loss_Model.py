## PyTorch dependencies
import torch.nn as nn
import torch

## External libaries
from .loss_functions import *

class Loss_model(nn.Module):
    
    def __init__(self,model,input_feat_size=512,embed_dim=2,loss='softmax',
                 device='cuda',weight=1):
        
        #inherit nn.module
        super(Loss_model,self).__init__()
        self.embed_dim = embed_dim
        self.loss = loss
        self.device = device
        self.num_classes = model.fc.out_features
        self.weight = weight
          
        #Regularization approach to use
        if loss == 'softmax':
            self.reg_method = nn.Sequential()
            
        elif loss == 'center': 
            if device == 'cuda':
                use_gpu = True
            else:
                use_gpu = False
            self.reg_method = CenterLoss(num_classes=model.fc.out_features, 
                                         feat_dim=self.embed_dim,use_gpu=use_gpu)
            
        #Learn ACE parameters (Random initalization) with single background stat
        elif loss == 'LACE': 
            self.reg_method = ACELoss(num_classes=model.fc.out_features,
                                      feat_dim=self.embed_dim,learn=True,
                                      single_stats=True,weight=self.weight,
                                      device=self.device)
        elif loss == 'AMC': 
            self.reg_method = AMCLoss(embed_dim, model.fc.out_features,device=device)
            
        else: #Angular loss
            self.reg_method = AngularPenaltySMLoss(embed_dim, model.fc.out_features, 
                                                   loss_type=loss,weight=self.weight)
        
    def forward(self,X,labels):

        if self.loss == 'softmax':
            loss = torch.FloatTensor(0).to(self.device) #No regularization loss
        else:
            loss, _ = self.reg_method(X,labels)
        
        #Return loss
        return loss
        
        
        
        
        
        