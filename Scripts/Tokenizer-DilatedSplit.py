#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import torch
import pandas as pd
from collections import Counter
from tqdm.notebook import tqdm


# In[20]:


class Tokenizer():
    def __init__(self, CODE_SETS, WINDOWS_SIZE,stride=1):
        self.windows_size = WINDOWS_SIZE
        assert type(CODE_SETS)==list or CODE_SETS =="fit"
        
        if type(CODE_SETS)==list:
            self.code_sets = CODE_SETS
        elif CODE_SETS=="fit":
            self.code_sets = None
        
        if self.code_sets:
            self.rna_dict,self.dict_size = self.get_rna_dict()
            self.stride = stride
    def __get_all_word(self, windows_size):
        '''获取所有长度为windows_size的可重复组合'''
        from functools import reduce
        a, b = self.code_sets, windows_size  
        res = reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [a] * b)
        return res    
    
    def get_rna_dict(self):
        '''构建字典'''
        rna_dict = {'<PAD>':0,'<CLS_label>':1,'<CLS_name>':2,'<MASK>':3}
        num = len(rna_dict)
        for j in range(self.windows_size,0,-1):
            for word in self.__get_all_word(j):
                rna_dict[word] = num
                num += 1
        return rna_dict,len(rna_dict)
    @staticmethod
    def pad_code(ngram_encode):
        ngram_encode2 = np.zeros([len(ngram_encode),len(max(ngram_encode,key = lambda x: len(x)))]) # 生成样本数*最长序列长度
        for i,j in enumerate(ngram_encode):
            ngram_encode2[i][0:len(j)] = j
        return ngram_encode2
    
    @staticmethod
    def get_files_seq_token(pth:str, data_col_name:str, label_col_name:str, fit=False):
        """
        获取pth路径下，col_name这列序列中的字符分布,若fit==True,则使用该文件下的字符
        :return:字符分布，所有的序列，所有的label
        """
        df = pd.read_csv(pth)
#         res = Counter(sum(df[data_col_name].apply(lambda x:[i for i in x.strip()]),[]))
        if fit:
            self.code_sets = list(res.keys())
        return list(df[data_col_name].apply(lambda x:x.strip())),torch.tensor(df[label_col_name])
    
    
    def encode(self, rna_seq, dilate=1, padding=True, return_pt=True, concer_tail=False):
        
        mRNA_dic,_ = self.get_rna_dict()
#         print(mRNA_dic)
        n_gram = [] # 存储基因词
        n_gram_encode = [] # 存储基因词的encode
        n_gram_len = [] # 存储基因链长度
        len_rna_seq = len(rna_seq)
        for i in tqdm(range(len_rna_seq)):
            cur_rna_seq = []
            cur_rna_encode = []
            if "U" in rna_seq[i]:
                rna_seq[i] = rna_seq[i].replace("U","C")
            for j in range(0,len(rna_seq[i]),self.stride):
#                 print(self.windows_size)
                len_win = len(rna_seq[i][j:j+self.windows_size])
#                 print(len_win)
                if not concer_tail:
                    if  len_win == self.windows_size:
                        try:
                            cur_rna_seq.append(rna_seq[i][j:j+self.windows_size*dilate:dilate].upper())
                            cur_rna_encode.append(mRNA_dic[rna_seq[i][j:j+self.windows_size:dilate].upper()])
                        except Exception as e:
                            print(e)
                            print(rna_seq[i],i)
                else:
                        cur_rna_seq.append(rna_seq[i][j:j+self.windows_size].upper())
                        cur_rna_encode.append(mRNA_dic[rna_seq[i][j:j+self.windows_size].upper()])
            n_gram.append(cur_rna_seq)
            n_gram_encode.append(cur_rna_encode)
            n_gram_len.append(len(cur_rna_encode))
        if padding:
            n_gram_encode = tokenizer.pad_code(n_gram_encode)
        if return_pt:
            n_gram_encode = torch.LongTensor(n_gram_encode)  
        return n_gram, n_gram_encode, torch.LongTensor(n_gram_len), mRNA_dic


# In[21]:


tokenizer = Tokenizer([i for i in "ATCG"],1,stride=1) # 1:以几个核苷酸为一个单位编码， 1：步长


# In[22]:


df = pd.read_csv("/home/zf/DNA N4//Datasets_pro-merge.csv",index_col=0).reset_index(drop=True)


# In[23]:


df


# In[24]:


all_seqs,all_labels = tokenizer.get_files_seq_token("/home/zf/DNA N4//Datasets_pro-merge.csv","data","label")


# In[25]:


df["name_idx"] = df["name"].map({k[0]:v for v,k in enumerate(Counter(df["name"]).items())})
df["usage_idx"] = df["usage"].map({k[0]:v for v,k in enumerate(Counter(df["usage"]).items())})


# In[26]:


df


# In[ ]:


encode_res = tokenizer.encode(all_seqs,dilate=0) #dilate设定膨胀数量


# In[ ]:


torch.save([tokenizer.rna_dict,encode_res[1],all_labels,torch.LongTensor(df["usage_idx"]),torch.LongTensor(df["name_idx"])],"dataset_pro_w1_s1.pkl")


# In[ ]:


# torch.save([all_seqs,all_labels],"../Data/1/origin_seq.pkl")


# In[ ]:




