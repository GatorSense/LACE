import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', 
                 eps=1e-7, s=None, m=None,weight=1):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 30.0 if not s else s
            self.m = 0.05 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc_ang = nn.Linear(in_features, out_features, bias=False)
        if (weight == 1): #Angular Softmax loss only
            self.fc = nn.Sequential()
        else:
            self.fc = nn.Linear(in_features,out_features)
        self.eps = eps
        self.weight = weight

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        #Pass features through separate classifier (i.e., output layer)
        x_out = self.fc(x)
        
        for W in self.fc_ang.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc_ang(x)
        labels = labels.long()
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L_denom = torch.log(denominator)
        L_nan = torch.isnan(L_denom)
        L_denom[L_nan != L_nan] = 1e-12
        L = (numerator - L_denom)
        
        if self.weight == 1:
            return -torch.mean(L), wf
        else:
            return -torch.mean(L), x_out
    
class AMCLoss(nn.Module):

    def __init__(self, in_features, out_features, s=None, m=None,device='cuda'):
        '''
        Angular Margin Contrastive Loss

        https://arxiv.org/pdf/2004.09805.pdf
        
        Code converted over from Tensorflow to Pytorch

        '''
        super(AMCLoss, self).__init__()
       
        self.m = 0.5 if not m else m
        self.s = 1.0 if not s else s
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.device = device

    def forward(self, X,labels=None):
        '''
        input shape (N, in_features)
        '''
        #L2 normalize features
        X = F.normalize(X, p=2, dim=1)
        batch_size = X.shape[0]

        #Pass through output layer
        wf = self.fc(X)
        
        #Get pseudo labels
        half = int(batch_size / 2)
        _, target_hard = torch.max(F.softmax(wf,dim=1), 1)
        
        #Get mask for correct and incorrect pseduo classes (neighboring pairs)
        #Error happens if batch size is not divisible
        try:
            neighbor_bool = torch.eq(target_hard[:half],target_hard[half:])
            inner = torch.sum(X[:half]*X[half:],axis=1)
        except:
            #Include some overlap with other sample to account for batch division issue
            neighbor_bool = torch.eq(target_hard[:half+1],target_hard[half:])
            inner = torch.sum(X[:half+1]*X[half:],axis=1)

        #Compute loss
        geo_desic = torch.acos(torch.clamp(inner,-1.0e-07,1.0e-07)) * self.s
        geo_losses = torch.where(neighbor_bool,torch.square(geo_desic),torch.square(F.relu(self.m - geo_desic))).clamp(min=1e-12)
       
        return torch.mean(geo_losses), wf
    

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    code from https://github.com/KaiyangZhou/pytorch-center-loss/blob/082ffa21c065426843f26129be51bb1cfd554806/center_loss.py#L4
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu


        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.fc = nn.Linear(feat_dim,num_classes)

    def forward(self, X, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = X.size(0)
        distmat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(X, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss, self.fc(X)
    
class ACELoss(nn.Module):
    """LACE loss term.
       Learn target signatures and background stats
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        device (str): device selected for memory allocation
    """

    def __init__(self, num_classes=11, feat_dim=2, device='cuda',learn=False,
                 single_stats=False,init_signatures=None,init_means=None,
                 init_covs=None,weight=1,eps=1e-7):
        super(ACELoss, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.learn = learn
        self.single_stats = single_stats
        self.weight = weight
        self.eps = eps
        
        #Compute bound for initialization
        bound = 1 / np.sqrt(feat_dim)
        
        if (weight == 1) or (weight==0): #ACE loss only
            self.fc = nn.Sequential()
        else:
            self.fc = nn.Linear(feat_dim,num_classes)
        
        if init_signatures is None:
            self.signatures = nn.init.uniform_(torch.randn(self.num_classes, 
                                                          self.feat_dim),a=-bound,b=bound)
        else:
            self.signatures = init_signatures
        
        if single_stats:
            self.b_means = nn.init.uniform_(torch.randn(1, self.feat_dim),a=-bound,b=bound)
            self.b_covs = torch.randn(self.feat_dim, self.feat_dim)
        else:
            if init_means is not None:
                self.b_means = init_means
                self.b_covs = init_covs
            else:
                self.b_means = nn.init.uniform_(torch.randn(self.num_classes,self.feat_dim),
                                               a=-bound,b=bound)
                self.b_covs = torch.randn(self.num_classes,self.feat_dim,self.feat_dim)
    
        if learn:
            self.signatures = nn.Parameter(self.signatures)
            self.b_means = nn.Parameter(self.b_means)
            self.b_covs = nn.Parameter(self.b_covs)
        else:
            self.signatures = self.signatures.to(device)
            self.b_means = self.b_means.to(device)
            self.b_covs = self.b_covs.to(device)
            
    def forward(self,X,labels):
     
        # Compute U (eigenvectors) and D (eigvenvalues) 
        try:
            U_mat, eigenvalues, _ = torch.svd(torch.mm(self.b_covs,self.b_covs.t()))
        except:
            U_mat, eigenvalues, _ = torch.svd(torch.mm(self.b_covs,self.b_covs.t())+
                                                      (1e-7*torch.eye(self.feat_dim,
                                                                           device=self.device)))
    
        #Compute D^-1/2 power
        D_mat = torch.diag_embed(torch.pow(eigenvalues, -1 / 2))

        #Compute matrix product DU^-1/2, should be
        #Perform transpose operation along DxD dimension (follow paper)
        DU = torch.matmul(D_mat, U_mat.T)
       
        #Center features (subtract background mean)
        # print(self.b_covs)
        X_centered = X-self.b_means
        
        #Compute x_hat
        xHat = torch.matmul(DU, X_centered.T)
        sHat = torch.matmul(DU, self.signatures.T)
        
        #Compute ACE score between features and signatures (NxC, one vs all), 
        #score between of -1 and 1 
        #L2 normalization done in function
        xHat = F.normalize(xHat.T,dim=1)
        sHat = F.normalize(sHat.T,dim=1)
        
        # ACE_targets = torch.mm(xHat,sHat) + torch.ones(batch_size,self.num_classes).to(self.device)
        ACE_targets = torch.mm(xHat,sHat.T)
    
        #Uniqueness of signatures in whitened dataspace
        #ACE scores in angular softmax
        labels = labels.long()
        numerator = torch.diagonal(ACE_targets.transpose(0, 1)[labels])
        excl = torch.cat([torch.cat((ACE_targets[i, :y], ACE_targets[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excl), dim=1)
        loss = (numerator - torch.log(denominator))
        loss = -torch.mean(loss)
     
        #Thing to try, maximize correct target, minimize others (use neq)
        return loss, ACE_targets
    

#LACE Ablation study layer
class ACELoss_Ablation(nn.Module):
    """LACE loss term.
       Learn target signatures and background stats
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        device (str): device selected for memory allocation
    """

    def __init__(self, num_classes=11, feat_dim=2, device='cuda',
                 single_stats=False,init_signatures=None,init_means=None,
                 init_covs=None,weight=1,eps=1e-7,learn_mean=False,learn_cov=False):
        super(ACELoss_Ablation, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.learn_mean = learn_mean
        self.learn_cov = learn_cov
        self.single_stats = single_stats
        self.weight = weight
        self.eps = eps
        
        #Compute bound for initialization
        bound = 1 / np.sqrt(feat_dim)
        
        if (weight == 1) or (weight==0): #ACE loss only
            self.fc = nn.Sequential()
        else:
            self.fc = nn.Linear(feat_dim,num_classes)
        
        if init_signatures is None:
            self.signatures = nn.init.uniform_(torch.randn(self.num_classes, 
                                                          self.feat_dim),a=-bound,b=bound)
        else:
            self.signatures = init_signatures
        
        if single_stats:
            self.b_means = nn.init.uniform_(torch.randn(1, self.feat_dim),a=-bound,b=bound)
            self.b_covs = torch.randn(self.feat_dim, self.feat_dim)
        else:
            if init_means is not None:
                self.b_means = init_means
                self.b_covs = init_covs
            else:
                self.b_means = nn.init.uniform_(torch.randn(self.num_classes,self.feat_dim),
                                               a=-bound,b=bound)
                self.b_covs = torch.randn(self.num_classes,self.feat_dim,self.feat_dim)
    
       #Learn signatures (always)
        self.signatures = nn.Parameter(self.signatures)
        
        if self.learn_mean:
            self.b_means = nn.Parameter(self.b_means)
        else:
            #Zero out b_means
            self.b_means = self.b_means * 0
            self.b_means = self.b_means.to(device)
    
        if self.learn_cov:
            self.b_covs = nn.Parameter(self.b_covs)
        else:
            if self.single_stats:
                self.b_covs = torch.eye(self.feat_dim)
            else:
                self.b_covs = torch.eye(self.feat_dim).unsqueeze(0).repeat(self.num_classes,1,1)
            self.b_covs = self.b_covs.to(device)
            
    def forward(self,X,labels):
     
        # Compute U (eigenvectors) and D (eigvenvalues)
        if self.learn_cov:
            try:
                U_mat, eigenvalues, _ = torch.svd(torch.mm(self.b_covs,self.b_covs.t()))
            except:
                U_mat, eigenvalues, _ = torch.svd(torch.mm(self.b_covs,self.b_covs.t())+
                                                          (1e-7*torch.eye(self.feat_dim,
                                                                               device=self.device)))
        
            #Compute D^-1/2 power
            D_mat = torch.diag_embed(torch.pow(eigenvalues, -1 / 2))
    
            #Compute matrix product DU^-1/2, should be
            #Perform transpose operation along DxD dimension (follow paper)
            DU = torch.matmul(D_mat, U_mat.T)
        else:
            DU = torch.eye(self.feat_dim).to(self.device)
       
        #Center features (subtract background mean)
        X_centered = X-self.b_means.to(self.device)
        
        #Compute x_hat
        xHat = torch.matmul(DU, X_centered.T)
        sHat = torch.matmul(DU, self.signatures.T)
        
        #Compute ACE score between features and signatures (NxC, one vs all), 
        #score between of -1 and 1 
        #L2 normalization done in function
        xHat = F.normalize(xHat.T,dim=1)
        sHat = F.normalize(sHat.T,dim=1)
        ACE_targets = torch.mm(xHat,sHat.T)
    
        #Uniqueness of signatures in whitened dataspace
        #ACE scores in angular softmax
        labels = labels.long()
        numerator = torch.diagonal(ACE_targets.transpose(0, 1)[labels])
        excl = torch.cat([torch.cat((ACE_targets[i, :y], ACE_targets[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excl), dim=1)
        loss = (numerator - torch.log(denominator))
        loss = -torch.mean(loss)
     
        #Thing to try, maximize correct target, minimize others (use neq)
        return loss, ACE_targets
    
    
