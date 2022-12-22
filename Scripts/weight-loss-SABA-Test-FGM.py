#!/usr/bin/env python
# coding: utf-8

# In[244]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torch.nn.functional as F


# In[245]:


import numpy as np
import random
import os
import collections
from tqdm.notebook import tqdm
from tensorboardX import SummaryWriter
from collections import Counter
import pandas as pd
from tqdm.notebook import tqdm
import math
import pickle


# In[246]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt


# # 初始化

# In[247]:


def set_seed(seed_id):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_id)
    random.seed(seed_id)
    os.environ['PYTHONHASHSEED'] = str(seed_id)
set_seed(2)


# In[248]:


class Args:
    mask_prob = 0.15
    seq_len = 41
    h_dim = 128
    h_head = 8
    class_num_label = 2
    class_num_name = 6
    num_layers = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dropout = 0.2
    cnn_out_channel = 128 # 记得转换成和embedding同维度
    kernel_size = 3
    topk = 5
    turn_dim = 128 
    folds = 10
    bidirectional = True
    batch_size = 512 
    aug_type_idx = 0
    aug_type_times = 1


# In[249]:


class Data:
    class Train:
        seqs,labels,usage,name = None,None,None,None
    class Test:
        seqs,labels,usage,name = None,None,None,None


# # 读取处理好的数据

# In[250]:


Args.token_dict,seqs,labels,usage,name = torch.load("/home/zf/DNA N4/dataset_Lin_pro_w1_s1.pkl")


# In[251]:


class CustomDataset(tud.Dataset):
    def __init__(self,seqs,labels,usage,name):
        Args.seq_len = seqs.shape[1] #更新序列长度
        self.seqs = seqs
        self.labels = labels
        self.usage = usage
        self.name = name
    def __len__(self):
        return self.seqs.shape[0]
    def __getitem__(self,idx):
        return self.seqs[idx],self.labels[idx],self.usage[idx],self.name[idx]
    @staticmethod
    def collate_fn(batch_list):
        batch_size = len(batch_list)
        seqs = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
        labels = torch.tensor([item[1] for item in batch_list])
        usage = torch.tensor([item[2] for item in batch_list])
        name = torch.tensor([item[3] for item in batch_list])
        pos_idx = torch.arange(seqs.shape[1])
        
        return seqs,labels,usage,name,pos_idx


# In[252]:


my_dataset = CustomDataset(seqs,labels,usage,name)
my_dataloader = tud.DataLoader(my_dataset,batch_size=16,shuffle=True,collate_fn=CustomDataset.collate_fn)


# In[253]:


next(iter(my_dataloader))


# In[254]:


"""分离训练集和验证集"""
train_seqs,eval_seqs = [],[]
train_labels,eval_labels = [],[]
train_usage,eval_usage = [],[]
train_name,eval_name = [],[]
for seq,label,use,name in zip(seqs,labels,usage,name):
    if use.item():
        train_seqs.append(seq)
        train_labels.append(label)
        train_usage.append(use)
        train_name.append(name)
    else:
        eval_seqs.append(seq)
        eval_labels.append(label)
        eval_usage.append(use)
        eval_name.append(name)    
Data.Train.seqs,Data.Train.labels,Data.Train.usage,Data.Train.name = torch.stack(train_seqs),torch.tensor(train_labels),torch.tensor(train_usage),torch.tensor(train_name)
Data.Test.seqs,Data.Test.labels,Data.Test.usage,Data.Test.name = torch.stack(eval_seqs),torch.tensor(eval_labels),torch.tensor(eval_usage),torch.tensor(eval_name)


# In[255]:


train_dataset = CustomDataset(Data.Train.seqs,Data.Train.labels,Data.Train.usage,Data.Train.name)
eval_dataset = CustomDataset(Data.Test.seqs,Data.Test.labels,Data.Test.usage,Data.Test.name)
train_dataloader = tud.DataLoader(train_dataset,batch_size=16,shuffle=True,collate_fn=CustomDataset.collate_fn)
eval_dataloader = tud.DataLoader(eval_dataset,batch_size=16,shuffle=True,collate_fn=CustomDataset.collate_fn)


# In[256]:


batch = next(iter(train_dataloader))


# In[257]:


batch


# # 构建base_line模型

# In[258]:


class Block(nn.Module):
    def __init__(self,use_features=Args.h_dim,dropout_rate=Args.dropout,class_num=Args.class_num_label,                 type_num=Args.class_num_name,lstm_hidden=Args.h_dim,cnn_features=Args.cnn_out_channel,                 kernel_size=Args.kernel_size,topk=Args.topk, turn_dim=Args.turn_dim):
        """use_features:使用的特征数"""
        """use_features:使用的特征数"""
        super(Block, self).__init__()
        self.use_features = use_features
        self.dropout_rate = dropout_rate
        self.bidirectional = Args.bidirectional
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.dropout = nn.Dropout(self.dropout_rate)
        self.type_embed = nn.Embedding(type_num, Args.h_dim) # 端编码
        self.lstm = nn.LSTM(use_features,lstm_hidden,batch_first=True,bidirectional=self.bidirectional) # 获取时序信息
        self.conv1d = nn.Conv1d(in_channels=use_features,out_channels=cnn_features,kernel_size=kernel_size,padding="same") # 获取分段信息
        self.globals = nn.Linear(use_features,lstm_hidden) # 获取全局信息，先映射一次
        
        self.topk = topk
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.avg_turn = nn.Linear(lstm_hidden, turn_dim)
        self.lstm_turn = nn.Linear(self.direction*lstm_hidden, turn_dim)
        self.conv_turn = nn.Linear(self.topk*cnn_features, turn_dim)
        self.embed_turn = nn.Linear(Args.h_dim, turn_dim)
#         self.token_embedding = nn.Embedding(len(Args.token_dict),Args.h_dim)
#         self.positional_embedding = nn.Embedding(Args.seq_len, Args.h_dim)
        self.layer_norm_l = nn.LayerNorm(turn_dim)
        self.layer_norm_r = nn.LayerNorm(turn_dim)
        self.layernorm = nn.LayerNorm(turn_dim)
        self.fc1 = nn.Linear(turn_dim,turn_dim*4)
        self.fc2 = nn.Linear(turn_dim*4,turn_dim)

        """初始化4个得分向量"""
        self.init_matrix = torch.eye(turn_dim,requires_grad=True).to(Args.device)
        self.rate_turn = nn.Linear(turn_dim,4) 
        

        
        """最终线性转化层"""
        self.fc = nn.Linear(turn_dim,class_num)
        
    @staticmethod
    def kmax_pooling(x, dim, k):    
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)
    
    def forward(self, batch_feature, pos_embedding, type_embed, return_next = False):
        """batch_feature:索引编码
           pos：位置编码
           batch_type：物种编码
        """

#         type_embed = self.type_embed(batch_type) 
#         pos_embedding = self.positional_embedding(pos)
#         batch_feature = self.token_embedding(batch_feature)
#         print(batch_feature.shape,pos_embedding.shape,type_embed.shape)
        batch_feature += pos_embedding #融合位置信息
        batch_feature += type_embed.unsqueeze(1)  #融合物种信息
        batch_feature = self.layernorm(batch_feature) #layernorm
        
        global_out = self.globals(batch_feature)
        lstm_out,(lstm_h,lstm_c) = self.lstm(batch_feature)
        lstm_hidden_all = torch.cat((lstm_c[0], lstm_c[1]),1)# 时序特征
        avg_output = torch.mean(global_out,dim=1)  #全局特征
        conv_out  = self.conv1d(batch_feature.permute(0,2,1))
        topk_res = Block.kmax_pooling(conv_out,2,self.topk)
        topk_res = topk_res.view(-1,self.cnn_features*self.topk) # 分区特征

        """融合信息"""
        topk_res = self.conv_turn(topk_res).unsqueeze(1)
        lstm_hidden_all = self.lstm_turn(lstm_hidden_all).unsqueeze(1)
        avg_output = self.avg_turn(avg_output).unsqueeze(1)

        all_message = torch.cat((topk_res,lstm_hidden_all,avg_output),1).to(Args.device)
        all_message = self.dropout(all_message) #dropout

        scores_arrays = self.rate_turn(self.init_matrix)
        scores = torch.matmul(self.layer_norm_l(all_message),scores_arrays)/math.sqrt(all_message.shape[1])
        weighted = torch.sum(scores,dim=-1)

        rating = torch.softmax(weighted,dim=-1).unsqueeze(2)
        weighted_message = all_message * rating
        merged_message = torch.sum(weighted_message,dim=1).squeeze()        
        logits = self.fc(merged_message)
        if  return_next:
            lstm_next = lstm_out[:,:,:self.lstm_hidden]  + lstm_out[:,:,self.lstm_hidden:]
            conv_out_next = conv_out.permute(0,2,1)
            next_input = lstm_next + conv_out_next + global_out + batch_feature
            next_input_norm = self.layernorm(next_input)#layernorm
            next_input_norm = F.relu(self.fc1(next_input_norm))
            next_input = self.layernorm(self.fc2(next_input_norm) + next_input)

#             print(next_input.shape,batch_feature.shape)
            return logits,torch.softmax(weighted,dim=0),next_input
        return logits,torch.softmax(weighted,dim=0)


# In[259]:


class Blocks(nn.Module):
    def __init__(self,use_features=Args.h_dim,dropout_rate=Args.dropout,class_num=Args.class_num_label,             type_num=Args.class_num_name,lstm_hidden=Args.h_dim,cnn_features=Args.cnn_out_channel,             kernel_size=Args.kernel_size,topk=Args.topk, turn_dim=Args.turn_dim, num_layers=Args.num_layers):
        super(Blocks,self).__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers-1)])
        self.last_block = Block()
        self.type_embed = nn.Embedding(type_num, Args.h_dim)
        self.token_embedding = nn.Embedding(len(Args.token_dict),Args.h_dim)
        self.positional_embedding = nn.Embedding(Args.seq_len, Args.h_dim)
    def forward(self, batch_feature, pos, batch_type):
        type_embed = self.type_embed(batch_type) 
        pos_embedding = self.positional_embedding(pos)
        batch_feature = self.token_embedding(batch_feature)
        for block in self.blocks:
            _,_,batch_feature = block(batch_feature, pos_embedding, type_embed,return_next = True)
#             batch_feature = batch_feature + batch_feature_next
        logits,weighted = self.last_block(batch_feature, pos_embedding, type_embed, return_next = False)
        return logits,torch.softmax(weighted,dim=0)


# In[260]:


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_names=['type_embed.weight', 'token_embedding.weight', 'positional_embedding.weight']):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            embed_bools = [emb_name == name for emb_name in emb_names]
#             print(embed_bools)
            if param.requires_grad and sum(embed_bools):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_names=['type_embed.weight', 'token_embedding.weight', 'positional_embedding.weight']):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and sum([emb_name == name for emb_name in emb_names]): 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# In[261]:


mymodel = Blocks().to(Args.device)


# In[262]:


batch


# In[263]:


res = mymodel(batch[0].to(Args.device),batch[-1].to(Args.device),batch[-2].to(Args.device))


# # 构建Trainer

# In[264]:


batch


# In[265]:


class Trainer:
    def __init__(self, epochs):
#         self.model = Block().to(Args.device)
#         self.criterion = nn.CrossEntropyLoss(ignore_index=-1).to(Args.device)
        self.criterion_sen = nn.CrossEntropyLoss(ignore_index=-1,reduction="none").to(Args.device)
#         self.optimizer = optim.AdamW(self.model.parameters(),lr=2e-4)
        self.epochs = epochs
#         self.writer = SummaryWriter("../tensorboard/实验1")
        self.best_fpr,self.best_tpr,self.best_epoch = [0]*Args.folds,[0]*Args.folds, [0]*Args.folds
        self.best_auc,self.best_acc,self.best_precision,        self.best_recall_sn,self.best_sp,self.best_f1,self.best_mcc =         [0]*Args.folds,[0]*Args.folds,[0]*Args.folds,[0]*Args.folds,[0]*Args.folds,[0]*Args.folds,[0]*Args.folds
        
    def train_one_epoch(self,dataloader,epoch,mode="Train"):
        self.model.train()    
        all_batch_loss = 0  
        fgm = FGM(self.model) #对抗训练
        for batch_data in dataloader:
#             print(batch_data)
            batch_feature, labels, usage, batch_type, pos = batch_data
            labels = labels.to(Args.device)
            label_logits,weighted = self.model(batch_feature.to(Args.device), pos.to(Args.device), batch_type.to(Args.device))
            pred_idx = torch.argmax(F.softmax(label_logits,dim=-1),dim=-1)
            train_loss = self.criterion_sen(label_logits,labels)
            train_loss = Trainer.caculate_weighted_loss(train_loss, batch_type, Args.aug_type_idx, Args.aug_type_times)
            all_batch_loss += train_loss
            train_loss.backward() #以此获取梯度
#             for name,parm in self.model.named_parameters():
#                 print(name,parm.grad)
            fgm.attack() # embedding被修改了
            label_logits,weighted = self.model(batch_feature.to(Args.device), pos.to(Args.device), batch_type.to(Args.device))
            train_loss_use = self.criterion_sen(label_logits,labels)
            train_loss_use = Trainer.caculate_weighted_loss(train_loss_use, batch_type, Args.aug_type_idx, Args.aug_type_times)
            train_loss_use.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复Embedding的参数
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        self.writer.add_scalar(mode+'_Loss/all_loss', all_batch_loss.item(), global_step=epoch)
    
    
    def evaluate(self,dataloader,epoch, mode):
        self.model.eval()
        all_label,all_pred_idx,all_pred_positive,all_type = [],[],[],[]
        all_batch_loss = 0 
        with torch.no_grad():
            for batch_data in dataloader:
                batch_feature, labels, usage, batch_type, pos = batch_data
                label_logits,weighted = self.model(batch_feature.to(Args.device), pos.to(Args.device), batch_type.to(Args.device))
#                 print(label_logits.shape)
                pred_idx = torch.argmax(F.softmax(label_logits,dim=-1),dim=-1)
                train_loss = self.criterion_sen(label_logits,labels.to(Args.device))
                train_loss = Trainer.caculate_weighted_loss(train_loss, batch_type, Args.aug_type_idx, Args.aug_type_times)
                all_label += list(labels.cpu().numpy())
                all_pred_idx += list(pred_idx.cpu().numpy())
                all_pred_positive += list(F.softmax(label_logits,dim=-1)[:,1].cpu().numpy()) #取预测为正例的结果
#                 print(all_pred_positive)
                all_type += list(batch_type.cpu().numpy())
                all_batch_loss += train_loss
        
            self.writer.add_scalar(mode+'/Loss', all_batch_loss.item(), global_step=epoch)
            tensor_all_label,tensor_all_pred_idx,tensor_all_pred_positive,tensor_all_type, =             torch.tensor(all_label),torch.tensor(all_pred_idx),torch.tensor(all_pred_positive),torch.tensor(all_type)
            
            A_label,A_pred_idx,A_pred_pos =             torch.masked_select(tensor_all_label,tensor_all_type==0),torch.masked_select(tensor_all_pred_idx,tensor_all_type==0),torch.masked_select(tensor_all_pred_positive,tensor_all_type==0)
            C_label,C_pred_idx,C_pred_pos =             torch.masked_select(tensor_all_label,tensor_all_type==1),torch.masked_select(tensor_all_pred_idx,tensor_all_type==1),torch.masked_select(tensor_all_pred_positive,tensor_all_type==1)
            D_label,D_pred_idx,D_pred_pos =             torch.masked_select(tensor_all_label,tensor_all_type==2),torch.masked_select(tensor_all_pred_idx,tensor_all_type==2),torch.masked_select(tensor_all_pred_positive,tensor_all_type==2)
            E_label,E_pred_idx,E_pred_pos =             torch.masked_select(tensor_all_label,tensor_all_type==3),torch.masked_select(tensor_all_pred_idx,tensor_all_type==3),torch.masked_select(tensor_all_pred_positive,tensor_all_type==3)
            Gpick_label,Gpick_pred_idx,Gpick_pred_pos =             torch.masked_select(tensor_all_label,tensor_all_type==4),torch.masked_select(tensor_all_pred_idx,tensor_all_type==4),torch.masked_select(tensor_all_pred_positive,tensor_all_type==4)
            Gsub_label,Gsub_pred_idx,Gsub_pred_pos =             torch.masked_select(tensor_all_label,tensor_all_type==5),torch.masked_select(tensor_all_pred_idx,tensor_all_type==5),torch.masked_select(tensor_all_pred_positive,tensor_all_type==5)

            """计算各种指标"""
            auc,acc,precision,recall_sn,sp,f1,mcc,fpr,tpr =            Trainer.caculate_show_score(tensor_all_label,tensor_all_pred_idx,tensor_all_pred_positive,mode+"/All",epoch,self.writer)
            if acc > self.best_acc[self.fold_idx] and mode=="Eval":
                self.best_epoch = epoch
                self.best_auc[self.fold_idx],self.best_acc[self.fold_idx],self.best_precision[self.fold_idx],                self.best_recall_sn[self.fold_idx],self.best_sp[self.fold_idx],self.best_f1[self.fold_idx],                self.best_mcc[self.fold_idx],self.best_fpr[self.fold_idx],self.best_tpr[self.fold_idx] = auc,acc,precision,recall_sn,sp,f1,mcc,fpr,tpr
                print("结果：auc:{},acc:{},precision:{},recall_sn:{},sp:{},f1:{},mcc:{},对应轮数为：{}轮".                      format(auc,acc,precision,recall_sn,sp,f1,mcc,epoch))
                plt.plot(fpr,tpr)
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.show()
#                 """存储最优模型"""
#                 torch.save(self.model.state_dict(),"../best_model/实验1 seed2/epoch_{}_acc_{}.pkl".format(epoch,acc))
                
            Trainer.caculate_show_score(A_label,A_pred_idx,A_pred_pos,mode+"/A",epoch,self.writer)
            Trainer.caculate_show_score(C_label,C_pred_idx,C_pred_pos,mode+"/C",epoch,self.writer)
            Trainer.caculate_show_score(D_label,D_pred_idx,D_pred_pos,mode+"/D",epoch,self.writer)
            Trainer.caculate_show_score(E_label,E_pred_idx,E_pred_pos,mode+"/E",epoch,self.writer)
            Trainer.caculate_show_score(Gpick_label,Gpick_pred_idx,Gpick_pred_pos,mode+"/Gpick",epoch,self.writer)
            Trainer.caculate_show_score(Gsub_label,Gsub_pred_idx,Gsub_pred_pos,mode+"/Gsub",epoch,self.writer)
            

        
    def train_epochs(self,Data):
        """k折交叉验证"""
        skf=StratifiedKFold(n_splits=Args.folds, random_state=1, shuffle=True)
        skf.get_n_splits(Data.Train.seqs,Data.Train.labels)
        self.fold_idx = 0

        self.writer = SummaryWriter("../tensorboard/实验4-test（A1）")
        self.model = Blocks().to(Args.device)
        self.optimizer = optim.AdamW(self.model.parameters(),lr=2e-4)
        train_dataset = CustomDataset(Data.Train.seqs,Data.Train.labels,Data.Train.usage,Data.Train.name)
        eval_dataset = CustomDataset(Data.Test.seqs,Data.Test.labels,Data.Test.usage,Data.Test.name)
        train_dataloader = tud.DataLoader(train_dataset,batch_size=324,shuffle=True,collate_fn=CustomDataset.collate_fn)
        eval_dataloader = tud.DataLoader(eval_dataset,batch_size=324,shuffle=True,collate_fn=CustomDataset.collate_fn)
        for epoch in tqdm(range(self.epochs)):
            self.train_one_epoch(train_dataloader,epoch)
            self.evaluate(train_dataloader,epoch,mode="Train")
            self.evaluate(eval_dataloader,epoch,mode="Eval")
        self.fold_idx += 1
            
    @staticmethod
    def caculate_show_score(y_true, y_pred_idx, y_pos, mode, epoch, writer,alpha=1e-9):
        
        TP = torch.sum((y_true == 1) & (y_pred_idx == 1)).item()
        FP = torch.sum((y_true == 0) & (y_pred_idx == 1)).item()
        TN = torch.sum((y_true == 0) & (y_pred_idx == 0)).item()
        FN = torch.sum((y_true == 1) & (y_pred_idx == 0)).item()
#         print(TP,FP,TN,FN)
        acc = (TP + TN) / (TP + TN + FP + FN + alpha)
        precision = TP / (TP + FP + alpha)
        recall_sn = TP / (TP + FN+ alpha)
        sp = TN / (TN + FP+ alpha)
        f1 = 2 * precision * recall_sn / (precision + recall_sn+ alpha)
        mcc = (TP * TN - FP * FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        auc = roc_auc_score(y_true, y_pos)
        
        writer.add_scalar(mode + "/ACC",acc,global_step = epoch)
        writer.add_scalar(mode + "/Precision",precision,global_step = epoch)
        writer.add_scalar(mode + "/Recall(Sn)",recall_sn,global_step = epoch)
        writer.add_scalar(mode + "/Sp",sp,global_step = epoch)
        writer.add_scalar(mode + "/F1",f1,global_step = epoch)
        writer.add_scalar(mode + "/MCC",mcc,global_step = epoch)
        writer.add_scalar(mode + "/AUC",auc,global_step = epoch)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pos, pos_label=1)
        
        return auc,acc,precision,recall_sn,sp,f1,mcc,fpr,tpr
    @staticmethod
    def caculate_weighted_loss(ori_loss, batch_type, cur_type, aug_weight):
        loss_weight = torch.masked_fill(torch.ones_like(batch_type),batch_type == cur_type, aug_weight).to(Args.device)
        weighted_loss = torch.mean(ori_loss * loss_weight)
        return weighted_loss


# In[266]:


trainer = Trainer(500)


# In[267]:


trainer.train_epochs(Data)


# In[268]:


trainer.best_fpr,trainer.best_tpr


# In[269]:


with open('weight-loss-SABA-Test-FGM（A1）.pk','wb') as file:
    pickle.dump((trainer.best_fpr,trainer.best_tpr),file)


# In[270]:


with open('weight-loss-SABA-Test-FGM（A1）.pk','rb') as file:
    b=pickle.load(file)
print(type(b))
print(b)

