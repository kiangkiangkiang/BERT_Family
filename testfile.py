import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch
from torchtext import datasets
from transformers import BertTokenizer

##############################test######################################
tmp = datasets.STSB(root='.data', split=('train', 'dev', 'test'))
t = BertTokenizer.from_pretrained("bert-base-chinese")
vocab = t.vocab
print("字典大小：", len(vocab))
tmp = list(vocab)
tmp[16000:16010]
import random
random.sample(tmp, 10)
vocab["瓶"]
##############################test######################################



########## Modules ##########
class BERT_Family(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def Load_Pretrained_Model(self, modelName: str):
        tokenizer = BertTokenizer.from_pretrained(modelName)
        return 
    
    def Data_Formatting(self, rawData, rawTarget):
        """ 
        rawData: n by p1, n: observations (total sequence). p1: number of sequences in each case.

        rawTarget: n by p2, n: observations (total sequence). p2: number of labels in each case.
        """
        #Prepare for one type data format for downstream task
        formattedData = 0
        return formattedData
    
    def Tokenize(self, document):
        return document
        
    def Padding(self, document):
        return document

class BF_Classification(BERT_Family):
    def __init__(self) -> None:
        super().__init__()


class BF_QA(BERT_Family):
    def __init__(self) -> None:
        super().__init__()

class Configurations(object):
  def __init__(self):
    self.batch_size = 64
    self.linear_hidden_layer = 5
    self.epoch = 10000
    self.learning_rate = 0.0001

def test(a = []):
    return a

########## Preprocessing ##########
class MyDataset(data.Dataset):
    def __init__(self) -> None:
       super().__init__()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], idx

#dataset / dataloader
#



########## Train and Pred and Eval... ##########
#build



