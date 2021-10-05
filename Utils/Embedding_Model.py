## PyTorch dependencies
import torch.nn as nn
import torch

## External libaries
from .loss_functions import *

class Embedding_model(nn.Module):
    
    def __init__(self,model,input_feat_size=512,embed_dim=2,loss='softmax',
                 device='cuda',weight=1):
        
        #inherit nn.module
        super(Embedding_model,self).__init__()
        self.embed_dim = embed_dim
        self.loss = loss
        self.device = device
        self.num_classes = model.fc.out_features
        self.weight = weight
        self.input_feat_size = input_feat_size
  
        #Define fully connected layer, backbone (feature extractor),
        #and embedding
        self.features = model
        
        if (embed_dim == input_feat_size) or (embed_dim is None):
            self.encoder = nn.Sequential()
            self.embed_dim = input_feat_size
        else:
            self.encoder = nn.Sequential(nn.Linear(input_feat_size,self.embed_dim),
                                      nn.PReLU(num_parameters=1))
          
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
            self.reg_method = AMCLoss(self.embed_dim, model.fc.out_features,device=device)
            
        elif loss == 'LACES_mean': 
            self.reg_method = ACELoss_Ablation(num_classes=model.fc.out_features,
                                      feat_dim=self.embed_dim,
                                      single_stats=True,weight=self.weight,
                                      device=self.device,learn_mean=True,
                                      learn_cov=False)
        elif loss == 'LACES_cov': 
            self.reg_method = ACELoss_Ablation(num_classes=model.fc.out_features,
                                      feat_dim=self.embed_dim,
                                      single_stats=True,weight=self.weight,
                                      device=self.device,learn_mean=False,
                                      learn_cov=True)
        elif loss == 'LACES_no_stats': 
            self.reg_method = ACELoss_Ablation(num_classes=model.fc.out_features,
                                      feat_dim=self.embed_dim,
                                      single_stats=True,weight=self.weight,
                                      device=self.device,learn_mean=False,
                                      learn_cov=False)
        else: #Angular loss
            self.reg_method = AngularPenaltySMLoss(self.embed_dim, model.fc.out_features, 
                                                   loss_type=loss,weight=self.weight)
            
        #Set size of output layer based on feature dimensionality (remove previous fully connected layer)
        if 'softmax':
            self.fc = nn.Linear(self.embed_dim,model.fc.out_features)
        else:
            self.fc = nn.Sequential()
       
        #Remove output layer from feature extractor
        model.fc = nn.Sequential()
        
    def forward(self,x,labels):

        #Pass in input through backbone
        x = self.features(x)
        
        #Pass through fully conneted layer and embedding model (in-line)
        x_embed = self.encoder(x)
        
        if self.loss == 'softmax':
            x_fc = self.fc(x_embed)
            loss = torch.zeros(1).to(self.device) #No regularization loss
            
        else:
            loss, x_fc = self.reg_method(x_embed,labels)
        
        #Return embedding, output (logits), and features
        return x_fc, x_embed, loss.unsqueeze(0)