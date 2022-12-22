

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim

import os
import sys
import csv
import math
from functools import reduce
from collections import Counter

import numpy as np
import pandas as pd

from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QVBoxLayout,QSplitter,QWidget,QLabel,QTextEdit,QPushButton,QTableWidgetItem,
                              QMainWindow,QDesktopWidget,QApplication,QGridLayout,QTableWidget,QAbstractItemView,
                                QMessageBox,qApp,QFileDialog,QRadioButton,QHBoxLayout)
from PyQt5.QtCore import Qt,QObject,pyqtSignal

import qdarkgraystyle
from plot2 import *
WINDOES_TITLE='4mc'
SPECIES=['C. elegans','D. melanogaster','A. thaliana','E. coli','G. subterraneus','G. pickeringii']
MODEL_PATH={'Basic':'../models/basic.pkl','C. elegans':'../models/C.pkl','D. melanogaster':'../models/D.pkl','A. thaliana':'../models/A.pkl','E. coli':'../models/E.pkl','G. subterraneus':'../models/Gsub20.pkl','G. pickeringii':'../models/Gpick5.pkl'}
def is_number(s):
    """Check whether it is a number"""
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

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
        
        a, b = self.code_sets, windows_size  
        res = reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [a] * b)
        return res    
    
    def get_rna_dict(self):

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
    def get_files_seq_token(df, data_col_name:str):
        res = Counter(sum(df[data_col_name].apply(lambda x:[i for i in x.strip()]),[]))
        return res,df[data_col_name].apply(lambda x:x.strip())
    
    
    def encode(self,rna_seq,padding=True,return_pt=True,concer_tail=False):
        
        mRNA_dic,_ = self.get_rna_dict()
        n_gram = [] 
        n_gram_encode = [] 
        n_gram_len = [] 
        len_rna_seq = len(rna_seq)

        for i in range(len_rna_seq):
            cur_rna_seq = []
            cur_rna_encode = []
            if "U" in rna_seq[i]:
                rna_seq[i] = rna_seq[i].replace("U","C")
            for j in range(0,len(rna_seq[i]),self.stride):
                len_win = len(rna_seq[i][j:j+self.windows_size])
                if not concer_tail:
                    if  len_win == self.windows_size:
                        try:
                            cur_rna_seq.append(rna_seq[i][j:j+self.windows_size].upper())
                            cur_rna_encode.append(mRNA_dic[rna_seq[i][j:j+self.windows_size].upper()])
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
            n_gram_encode = Tokenizer.pad_code(n_gram_encode)
        if return_pt:
            n_gram_encode = torch.LongTensor(n_gram_encode)  
        return n_gram, n_gram_encode, torch.LongTensor(n_gram_len), mRNA_dic

class Args:
    seq_len=0
    token_dict=None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(tud.Dataset):

    def __init__(self,seqs,name):
        Args.seq_len = seqs.shape[1]
        self.seqs = seqs
        self.name = name

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self,idx):
        return self.seqs[idx],self.name[idx]

    @staticmethod
    def collate_fn(batch_list):
        batch_size = len(batch_list)
        seqs = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
        name = torch.tensor([item[1] for item in batch_list])
        pos_idx = torch.arange(seqs.shape[1])
        
        return seqs,pos_idx,name

class Block(nn.Module):
    def __init__(self,use_features=128,dropout_rate=0.2,class_num=2,\
                 type_num=6,lstm_hidden=128,cnn_features=128,\
                 kernel_size=3,topk=5, turn_dim=128):
        super(Block, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_features = use_features
        self.dropout_rate = dropout_rate
        self.bidirectional = True
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.dropout = nn.Dropout(self.dropout_rate)
        self.type_embed = nn.Embedding(type_num, 128)
        self.lstm = nn.LSTM(use_features,lstm_hidden,batch_first=True,bidirectional=self.bidirectional)
        self.conv1d = nn.Conv1d(in_channels=use_features,out_channels=cnn_features,kernel_size=kernel_size,padding="same")
        self.globals = nn.Linear(use_features,lstm_hidden)
        
        self.topk = topk
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.avg_turn = nn.Linear(lstm_hidden, turn_dim)
        self.lstm_turn = nn.Linear(self.direction*lstm_hidden, turn_dim)
        self.conv_turn = nn.Linear(self.topk*cnn_features, turn_dim)
        self.embed_turn = nn.Linear(128, turn_dim)

        self.layer_norm_l = nn.LayerNorm(turn_dim)
        self.layer_norm_r = nn.LayerNorm(turn_dim)
        self.layernorm = nn.LayerNorm(turn_dim)
        self.fc1 = nn.Linear(turn_dim,turn_dim*4)
        self.fc2 = nn.Linear(turn_dim*4,turn_dim)

        self.init_matrix = torch.eye(turn_dim,requires_grad=True).to(Args.device)
        self.rate_turn = nn.Linear(turn_dim,4) 

        self.fc = nn.Linear(turn_dim,class_num)
        
    @staticmethod
    def kmax_pooling(x, dim, k):    
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)
    
    def forward(self, batch_feature, pos_embedding, type_embed, return_next = False):

        batch_feature += pos_embedding
        batch_feature += type_embed.unsqueeze(1)
        batch_feature = self.layernorm(batch_feature)
        
        global_out = self.globals(batch_feature)
        lstm_out,(lstm_h,lstm_c) = self.lstm(batch_feature)
        lstm_hidden_all = torch.cat((lstm_c[0], lstm_c[1]),1)
        avg_output = torch.mean(global_out,dim=1)
        conv_out  = self.conv1d(batch_feature.permute(0,2,1))
        topk_res = Block.kmax_pooling(conv_out,2,self.topk)
        topk_res = topk_res.view(-1,self.cnn_features*self.topk)

        topk_res = self.conv_turn(topk_res).unsqueeze(1)
        lstm_hidden_all = self.lstm_turn(lstm_hidden_all).unsqueeze(1)
        avg_output = self.avg_turn(avg_output).unsqueeze(1)

        all_message = torch.cat((topk_res,lstm_hidden_all,avg_output),1).to(self.device)
        all_message = self.dropout(all_message)

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
            next_input_norm = self.layernorm(next_input)
            next_input_norm = F.relu(self.fc1(next_input_norm))
            next_input = self.layernorm(self.fc2(next_input_norm) + next_input)

            return logits,torch.softmax(weighted,dim=0),next_input
        return logits,torch.softmax(weighted,dim=0)

class Blocks(nn.Module):
    def __init__(self,use_features=128,dropout_rate=0.2,class_num=2,\
             type_num=6,lstm_hidden=128,cnn_features=128,\
             kernel_size=3,topk=5, turn_dim=128, num_layers=4):
        super(Blocks,self).__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers-1)])
        self.last_block = Block()
        self.type_embed = nn.Embedding(type_num, 128)
        self.token_embedding = nn.Embedding(len(Args.token_dict),128)
        self.positional_embedding = nn.Embedding(Args.seq_len, 128)
    def forward(self, batch_feature, pos, batch_type):
        type_embed = self.type_embed(batch_type) 
        pos_embedding = self.positional_embedding(pos)
        batch_feature = self.token_embedding(batch_feature)
        for block in self.blocks:
            _,_,batch_feature = block(batch_feature, pos_embedding, type_embed,return_next = True)
        logits,weighted = self.last_block(batch_feature, pos_embedding, type_embed, return_next = False)
        return logits,torch.softmax(weighted,dim=0)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_names=['type_embed.weight', 'token_embedding.weight', 'positional_embedding.weight']):
        for name, param in self.model.named_parameters():
            embed_bools = [emb_name == name for emb_name in emb_names]
            if param.requires_grad and sum(embed_bools):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_names=['type_embed.weight', 'token_embedding.weight', 'positional_embedding.weight']):

        for name, param in self.model.named_parameters():
            if param.requires_grad and sum([emb_name == name for emb_name in emb_names]): 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def predict_main(df,model_path):
    # print(fname[0])
    tsne(fname[0])
    tokenizer = Tokenizer([i for i in "ATCG"],1,stride=1)
    tokens,all_seqs = tokenizer.get_files_seq_token(df,'data')

    Args.token_dict=tokenizer.rna_dict
    encode_res = tokenizer.encode(all_seqs)[1]
    name_idx = torch.LongTensor(df["name"].map({k[0]:v for v,k in enumerate(Counter(df["name"]).items())}))

    dataset = CustomDataset(encode_res,name_idx)
    dataloader = tud.DataLoader(dataset,batch_size=16,shuffle=False,collate_fn=CustomDataset.collate_fn)

    state_dict=torch.load(model_path,map_location=torch.device('cpu'))
    mymodel=Blocks().to(Args.device)
    mymodel.load_state_dict(state_dict)
    mymodel.eval()

    res=torch.tensor([])
    with torch.no_grad():
        for batch_data in dataloader:
            batch_feature,pos,batch_type=batch_data
            label_logits,weighted = mymodel(batch_feature.to(Args.device), pos.to(Args.device), batch_type.to(Args.device))
            pred_idx = torch.argmax(F.softmax(label_logits,dim=-1),dim=-1)
            res=torch.concat([res,pred_idx.detach().cpu().long()])
    
    df['label']=res
    # tsne()
    return df

class fileBtnWidget(QWidget,QObject):
    input_signal=pyqtSignal(object)
    input_state_signal=pyqtSignal(object)
    """File Widget"""
    def __init__(self,name):
        super().__init__()
        self.name=name
        self.res=None
        self.initUI()

    def initUI(self):
        if self.name=='import':
            importBtn = QPushButton("Import", self)
            importBtn.clicked.connect(self.importButtonClicked)
            fileGrid = QGridLayout()
            fileGrid.addWidget(importBtn,0,0)
            self.setLayout(fileGrid)
        if self.name=='export':
            exportBtn = QPushButton("Export", self)
            exportBtn.clicked.connect(self.exportButtonClicked)
            fileGrid = QGridLayout()
            fileGrid.addWidget(exportBtn,0,0)
            self.setLayout(fileGrid)

    def clear_res(self):
        self.res=None

    def toMB(self,bytesize):
        return f'{bytesize/1024/1024:.2f}'
      
    def importButtonClicked(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file','','Text files (*.csv)')
        # print(fname[0])

        if fname[0]:
            self.input_state_signal.emit('Loading data......')
            data=pd.read_csv(fname[0])
            self.input_signal.emit(data)
            self.input_state_signal.emit('File loaded successfully')
      
    def exportButtonClicked(self):
        if self.res is not None:
            fname,ok= QFileDialog.getSaveFileName(self, 'Save file','','csv(*.csv)')
            if fname:
                self.res.to_csv(fname,index=False)
                    
                QMessageBox.about(self,'Success',"File saved successfully.")
        else:
            QMessageBox.about(self,'Error',"The result is empty.")
    
    def refresh_result_data(self,data):
        self.res=data

class PdtBtnWidget(QWidget,QObject):
    '''Predict functional related widgets'''

    predict_signal=pyqtSignal(object)
    input_signal=pyqtSignal(object)
    clear_signal=pyqtSignal()

    def __init__(self):
        super().__init__()
        self.input_data=None
        self.model_path=MODEL_PATH['Basic']
        self.initUI()

    def initUI(self):

        self.pic = QLabel(self)
        # self.pic.setText("显示图片")
        self.pic.setFixedSize(850, 600)
        # self.pic.move(-100,100)
        self.pic.setStyleSheet("QLabel{background:grey;}")

        exampleBtn = QPushButton("tsne", self)
        exampleBtn.clicked.connect(self.examplebuttonClicked)

        predictBtn = QPushButton("Predict", self)
        predictBtn.clicked.connect(self.prebuttonClicked)

        clearBtn = QPushButton("Clear", self)
        clearBtn.clicked.connect(self.clearbuttonClicked)

        exitBtn = QPushButton("Exit", self)
        exitBtn.clicked.connect(self.exitClicked)

        pdtGrid = QGridLayout()
        pdtGrid.setRowStretch(1,1)
        pdtGrid.addWidget(predictBtn,4,0,1,1)
        # pdtGrid.setRowStretch(1,1)
        pdtGrid.addWidget(exampleBtn,4,1,1,1)
        pdtGrid.addWidget(clearBtn,4,2,1,1)
        pdtGrid.addWidget(exitBtn,4,3,1,1)
        self.setLayout(pdtGrid)
        # self.layout.addstrech

    def refresh_input_data(self,input_data):
        self.input_data=input_data

    def refresh_model_path(self,model_name):
        self.model_path=MODEL_PATH[model_name]

    def prebuttonClicked(self):

        if self.input_data is not None:
            res=predict_main(self.input_data,self.model_path)
            self.predict_signal.emit(res)
        else:
            QMessageBox.about(self,'Error','Please input your data.')
    
    # #清除按钮操作
    def clearbuttonClicked(self):
        self.input_data=None
        self.clear_signal.emit()
        self.pic.setPixmap(QPixmap(""))

    #输入案例操作
    def examplebuttonClicked(self):
        self.showImage = QPixmap(r"F:\360MoveData\Users\Peacewalker\Desktop\DNA N4\UI\ui\4mc\tsne.png")
        self.pic.setPixmap(self.showImage.scaled(850,600))
        # text="""
        # You can click import to upload data。
        #
        # ITwo columns are required for input data sequence data and name.
        # name stands for species
        #
        # C:'C. elegans'
        # D:'D. melanogaster'
        # A:'A. thaliana'
        # E:'E. coli',
        # Gsub:'G. subterraneus'
        # Gpick:'G. pickeringii'
        #
        # """
        # QMessageBox.about(self,'Data format example',text)
        # data=pd.read_csv('../data/demo.csv')
        # self.input_signal.emit(data)

    def exitClicked(self,event):
        reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            # event.ignore()
            pass

class ModelBtnWidget(QWidget,QObject):
    """Model selection radio button"""

    check_signal = pyqtSignal(object)

    def __init__(self,model_name,check,font_size=14):
        super().__init__()
        self.model_name=model_name
        self.check=check
        self.font_size=font_size
        self.initUI()

    def initUI(self):
        layout=QHBoxLayout(self)
        meanModelTitle = QLabel(self.model_name)
        meanModelTitle.setFont(QFont('Arial', self.font_size))
        self.meanModelBtn=QRadioButton()
        self.meanModelBtn.setChecked(self.check)
        self.meanModelBtn.clicked.connect(lambda:self.setModelClicked(self.model_name))

        layout.addWidget(self.meanModelBtn)
        layout.addWidget(meanModelTitle)

    def getBtnName(self):
        return self.model_name

    def setModelClicked(self,btnname):
        self.check_signal.emit(btnname)
        self.meanModelBtn.setChecked(True)

class MdlWidget(QWidget,QObject):
    """Model Selection Widget"""

    model_select_signal=pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):


        # self.pic.move(12)


        self.model_list=[]

        layout=QGridLayout()

        # basicTitle = QLabel('Basic Model')
        # basicTitle.setFont(QFont('Arial', 17))

        # spTitle = QLabel('Specie Enhance Model')
        # spTitle.setFont(QFont('Arial', 17))

        # self.m0=ModelBtnWidget('Basic',True)
        # self.model_list.append(self.m0)
        # self.m0.check_signal.connect(self.buttonClicked)
        #
        # self.m2=ModelBtnWidget(SPECIES[0],False)
        # self.model_list.append(self.m2)
        # self.m2.check_signal.connect(self.buttonClicked)
        #
        # self.m3=ModelBtnWidget(SPECIES[1],False)
        # self.model_list.append(self.m3)
        # self.m3.check_signal.connect(self.buttonClicked)
        #
        # self.m4=ModelBtnWidget(SPECIES[2],False)
        # self.model_list.append(self.m4)
        # self.m4.check_signal.connect(self.buttonClicked)
        #
        # self.m5=ModelBtnWidget(SPECIES[3],False)
        # self.model_list.append(self.m5)
        # self.m5.check_signal.connect(self.buttonClicked)
        #
        # self.m6=ModelBtnWidget(SPECIES[4],False)
        # self.model_list.append(self.m6)
        # self.m6.check_signal.connect(self.buttonClicked)
        #
        # self.m7=ModelBtnWidget(SPECIES[5],False)
        # self.model_list.append(self.m7)
        # self.m7.check_signal.connect(self.buttonClicked)


        # layout.addWidget(basicTitle,0,0,1,3)
        # layout.addWidget(self.m0,1,0,1,1)

        # layout.addWidget(spTitle,2,0,1,1)
        # layout.addWidget(self.m2,3,0,1,1)
        # layout.addWidget(self.m3,3,1,1,1)
        # layout.addWidget(self.m4,3,2,1,1)
        # layout.addWidget(self.m5,4,0,1,1)
        # layout.addWidget(self.m6,4,1,1,1)
        # layout.addWidget(self.m7,4,2,1,1)

        self.setLayout(layout)

    def buttonClicked(self,btnname):
        for i,mbtn in enumerate(self.model_list):
            if mbtn.getBtnName()!=btnname:
                self.model_list[i].meanModelBtn.setChecked(False)
        self.model_select_signal.emit(btnname)
# from PyQt5 import QtCore, QtGui, QtWidgets
# class Ui_Form(object):
#     def setupUi(self, Form):
#         Form.setObjectName("Form")
#         Form.resize(645, 500)
#         self.label = QtWidgets.QLabel(Form)
#         self.label.setGeometry(QtCore.QRect(140, 100, 381, 291))
#         self.label.setPixmap(QPixmap('F:/360MoveData/Users/Peacewalker/Desktop/DNA N4/UI/ui/4mc/tsne.png'))# 图片路径
#         self.label.setText("")
#         self.label.setObjectName("label")
#
#         self.retranslateUi(Form)
#         QtCore.QMetaObject.connectSlotsByName(Form)
#
#     def retranslateUi(self, Form):
#         _translate = QtCore.QCoreApplication.translate
#         Form.setWindowTitle(_translate("Form", "Form"))

class RightFuncBtnWidget(QWidget):
    """Fucntion Widget"""

    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.pdtBtnWidget=PdtBtnWidget()
        # self.mdlWidget=MdlWidget()

        # Set the signal
        # self.mdlWidget.model_select_signal.connect(self.pdtBtnWidget.refresh_model_path)
        grid = QGridLayout()
        grid.setSpacing(10)

        # grid.addWidget(self.mdlWidget,0,0,2,1)
        grid.addWidget(self.pdtBtnWidget,2,0,1,1)

        self.setLayout(grid)

class MyTable(QTableWidget):
    """Table Widget"""

    def __init__(self,parent=None):
        super(MyTable,self).__init__(parent)
        # set data readOnly
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # update data
        self.updateData([],[],np.array([]))
    
    # set header
    def setTableHeader(self,header):
        if header is not None:
            self.setHorizontalHeaderLabels(list(header))
		
    # set row name
    def setTableRowName(self,row_name=None):
        if row_name is not None:
            row_name=[str(i) for i in row_name]
            self.setVerticalHeaderLabels(row_name)
    
    def removeBefore(self):
        #initialize header
        self.setColumnCount(0)
        #set header
        self.setTableHeader('')
        
        rowcount = self.rowCount()
        while rowcount>0:
            rowcount = self.rowCount()
            self.removeRow(rowcount-1)

    def updateData(self,header,array,row_name=None):
        self.removeBefore()
        if array is not None and len(array)>0:
            
            # set column count
            max_columns_len=np.max([len(i) for i in array])
            max_header_len=0
            if header is not None:
                max_header_len=len(header)
            self.setColumnCount(max(max_header_len,max_columns_len))

            # set header
            self.setTableHeader(header)

            # set array
            for i in range(len(array)):
                rowcount = self.rowCount()
                self.insertRow(rowcount)
                self.cur_line=array[i]
									
                for j in range(len(self.cur_line)):
                    if is_number(self.cur_line[j]) and isinstance(self.cur_line[j],str) and self.cur_line[j].isdigit() is False:
                        self.setItem(i,j,QTableWidgetItem('%.4f'%float(self.cur_line[j])))
                    else:
                        self.setItem(i,j,QTableWidgetItem(str(self.cur_line[j])))
            
            self.resizeColumnsToContents()
            self.horizontalHeader().setStretchLastSection(True)

            # set row count and name
            self.setRowCount(len(array))
            self.setTableRowName(row_name)
            
class InputWidget(QWidget):
    """Upload files in this widget"""

    def __init__(self):
        super().__init__()
        self.init_UI()

    def init_UI(self):

        grid=QGridLayout()

        dataTitle = QLabel('Please upload your data')
        dataTitle.setFont(QFont('Arial', 20))
        self.dataTable=MyTable()
        self.importBtn=fileBtnWidget('import')

        # Set the signal
        self.importBtn.input_signal.connect(self.refreshTable)

        grid.addWidget(dataTitle,0,0,1,1)
        grid.addWidget(self.importBtn,0,1,1,1)
        grid.addWidget(self.dataTable,1,0,3,2)

        self.setLayout(grid)

    def refreshTable(self,data):
        self.dataTable.updateData(data.columns,data.values)

class ResultWidget(QWidget):
    """Displays the results and provides an export interface"""

    def __init__(self):
        super().__init__()
        self.init_UI()
    
    def init_UI(self):
        reTitle = QLabel('Prediction result')
        reTitle.setFont(QFont('Arial', 20))
        self.exportBtn=fileBtnWidget('export')
        self.reTable=MyTable()

        grid=QGridLayout(self)
        grid.addWidget(reTitle, 0, 0,1,3)
        grid.addWidget(self.reTable,1,0,3,3)
        grid.addWidget(self.exportBtn, 4, 0,1,3)
    
    def refresh_result_data(self,data):
        self.reTable.updateData(data.columns,data.values)

class ContentWidget(QWidget):
    """Main Widget contains subwidgets"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):        
        
        # SubWidget
        self.inputWidget=InputWidget()
        self.rfBtn=RightFuncBtnWidget()
        self.resultWidget=ResultWidget()

        # Bind the signal
        self.inputWidget.importBtn.input_signal.connect(self.rfBtn.pdtBtnWidget.refresh_input_data)
        self.rfBtn.pdtBtnWidget.input_signal.connect(self.rfBtn.pdtBtnWidget.refresh_input_data)
        self.rfBtn.pdtBtnWidget.input_signal.connect(self.inputWidget.refreshTable)
        self.rfBtn.pdtBtnWidget.predict_signal.connect(self.resultWidget.refresh_result_data)
        self.rfBtn.pdtBtnWidget.predict_signal.connect(self.resultWidget.exportBtn.refresh_result_data)
        self.rfBtn.pdtBtnWidget.clear_signal.connect(self.inputWidget.dataTable.removeBefore)
        self.rfBtn.pdtBtnWidget.clear_signal.connect(self.resultWidget.reTable.removeBefore)
        self.rfBtn.pdtBtnWidget.clear_signal.connect(self.resultWidget.exportBtn.clear_res)
        # Set the layout
        layout = QVBoxLayout()

        splitter_up = QSplitter(Qt.Horizontal)
        splitter_up.addWidget(self.inputWidget)
        splitter_up.addWidget(self.rfBtn)

        splitter= QSplitter(Qt.Vertical)
        splitter.addWidget(splitter_up)
        splitter.addWidget(self.resultWidget)
        
        layout.addWidget(splitter)

        self.setLayout(layout)
        self.setGeometry(300, 300, 350, 300)
        self.show() 

class MainWindow(QMainWindow):
    """Main Window"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(WINDOES_TITLE)
        self.center()
        self.setStyleSheet(qdarkgraystyle.load_stylesheet_pyqt5())
        
        self.cw = ContentWidget()
        self.setCentralWidget(self.cw)
        self.statusBar().showMessage('')

        # Set the state
        # self.cw.inputWidget.importBtn.input_state_signal.connect(self.refresh_statusBar)
        self.show()

    def center(self):
        """Center the window"""
        self.resize(3840,2160)
        self.setWindowState(Qt.WindowMaximized)
        qr=self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def refresh_statusBar(self,state):
        self.statusBar().showMessage(state)

if __name__=="__main__":
    os.chdir(sys.path[0])
    app=QApplication(sys.argv)
    # ui = Ui_Form()
    # ui.setupUi(MainWindow)
    window=MainWindow()
    sys.exit(app.exec_())

