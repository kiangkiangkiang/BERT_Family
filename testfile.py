import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch
from torchtext import datasets
from transformers import BertTokenizer
import pandas as pd

##############################test######################################
tmp = datasets.STSB(root='.data', split=('train', 'dev', 'test'))
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
vocab = t.vocab
print("字典大小：", len(vocab))
tmp = list(vocab)
tmp[16000:16010]
import random
random.sample(tmp, 10)
vocab["瓶"]

df_train = pd.read_csv("~/Downloads/train.csv")
df_train.iloc[2,:]
empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train[empty_title]
# 剔除過長的樣本以避免 BERT 無法將整個輸入序列放入記憶體不多的 GPU
MAX_LENGTH = 30

df_train[df_train.title1_zh.apply(lambda x: len(x) < MAX_LENGTH)]
pd.DataFrame(df_train)
x = df_train[['title1_zh','title2_zh']]
y = df_train.label

test = '[CLS]' + x.iloc[0,0] + '[SEP]' + x.iloc[0,1] + '[SEP]' 
tmp2 = tokenizer(x.iloc[0,0] + '[SEP]' + x.iloc[0,1], padding="max_length", max_length=100)
tmp3 = tokenizer(x.iloc[0,0] + '[SEP]' + x.iloc[0,1], padding="max_length", max_length=100)
tokenizer(x.iloc[0,0] + '[SEP]' + x.iloc[0,1], padding="max_length", max_length=50, truncation = True)
tmp3["token_type_ids"] = 1

tmp = tokenizer.tokenize(test)
ids =  tokenizer.convert_tokens_to_ids(tmp)
tmp2["input_ids"] == ids
ss = {}
for i, ele in enumerate(pd.unique(df_train.label)):
    ss[ele] = i

ss
ss[df_train.label[5]]
df_train.shape[0]
t = BertTokenizer.from_pretrained('bert-base-uncased') 

temp = df_train[['title1_zh','title2_zh']]

temp2 = temp.iloc[[1,32,300,555,10007]]
temp2.values
temp2.iloc[:,0]

tokenizer(temp2.iloc[:,0].values)
tokenizer.tokenize(temp2.iloc[:,0])

torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
tokenizer(x.iloc[0,0] + '[SEP]' + x.iloc[0,1], padding="max_length", max_length=50, truncation = True)


training = Classification_Dataset(temp, temp2)
train_loader = data.DataLoader(training, batch_size=64, shuffle=True)
train_iter = infinite_iter(train_loader)
a, b, c = next(train_iter)
a
a.shape
##############################test######################################



########## Modules ##########
class BERT_Family(nn.Module):
    def __init__(self, tokenizer = 'bert-base-uncased') -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def Load_Pretrained_Model(self, modelName: str):
        return 
    
    def Data_Formatting(self, rawData, rawTarget):


        #build target dictionary
        rawTarget_dict = {}
        for i, ele in enumerate(pd.unique(rawTarget)):
            rawTarget_dict[ele] = i


        

        formattedData = 0
        return formattedData
    







class BF_Classification(BERT_Family):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class BF_QA(BERT_Family):
    def __init__(self) -> None:
        super().__init__()

class Configurations(object):
  def __init__(self):
    self.batch_size = 64
    self.linear_hidden_layer = 5
    self.epoch = 10000
    self.learning_rate = 0.0001
    self.BF_MaxLength = 30

def test(a = []):
    return a

########## Preprocessing ##########
class Classification_Dataset(data.Dataset):
    def __init__(self, rawData, rawTarget, maxLength = 100) -> None:
        super().__init__()
        """ 
        rawData: n by p1, n: observations (total sequence). p1: number of sequences in each case.

        rawTarget: n by p2, n: observations (total sequence). p2: number of labels in each case.
        """
        #Prepare for one type data format for downstream task
        self.rawData, self.rawTarget = pd.DataFrame(rawData), pd.DataFrame(rawTarget)
        
        
    def __len__(self):
        return self.rawData.shape[0]
    def __getitem__(self, idx):
        token = tokenizer(self.rawData.iloc[idx, 0], padding="max_length", max_length=50, truncation = True)
        return torch.tensor(token["input_ids"], dtype=torch.long), \
                torch.tensor(token["token_type_ids"], dtype=torch.long), \
                    torch.tensor(token["attention_mask"], dtype=torch.long)

#dataset / dataloader
#

def infinite_iter(data_loader):
  it = iter(data_loader)
  while True:
    try:
      ret = next(it)
      yield ret
    except StopIteration:
      it = iter(data_loader)

########## Train and Pred and Eval... ##########
#build



