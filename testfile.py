import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch
from torchtext import datasets
from transformers import BertTokenizer
import pandas as pd
from transformers import BertForSequenceClassification
from functools import reduce


########## Modules ##########
#不想export出去的
#One or two sentence with one dim label
class Classification_Dataset(data.Dataset):
    def __init__(self, rawData, rawTarget, tokenizer, maxLength = 100) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.rawData, self.rawTarget = pd.DataFrame(rawData), rawTarget
        assert self.rawData.shape[1] <= 2, "Only accept one or two sequences as the input argument."

        self.rawTarget_dict = {}
        for i, ele in enumerate(pd.unique(rawTarget)):
            self.rawTarget_dict[ele] = i
        self.maxLength = maxLength

    def __len__(self):
        return self.rawData.shape[0]

    def __getitem__(self, idx):
        if self.rawData.shape[1] == 1:
            result = self.tokenizer.encode_plus(self.rawData.iloc[idx, 0], self.rawData.iloc[idx, 1], padding="max_length", max_length=self.maxLength, truncation = True, return_tensors = 'pt')
        else:
            result = self.tokenizer.encode_plus(self.rawData.iloc[idx, 0], padding="max_length", max_length=self.maxLength, truncation = True, return_tensors = 'pt')
        return result, torch.tensor(self.rawTarget_dict[self.rawTarget[idx]])

#這個class我想做一個比較客製化的？
class BERT_Family(nn.Module):
    def __init__(self, pretrainedModel = 'bert-base-uncased', maxLength = 100,\
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()
        print("Using device: ", device)
        self.device = device
        self.pretrainedModel = pretrainedModel
        self.tokenizer = BertTokenizer.from_pretrained(pretrainedModel)
        self.maxLength = maxLength
        self.dataset, self.dataLoader, self.dataIter = None, None, None
        self.model, self.iteration = None, 0
        self.status = {"BERT_Type": ["BERT_Family"],\
                        "hasData": False,\
                        "hasModel": False,\
                        "isTrained": False,\
                        "trainingTimes": 0,\
                        "accumulateEpoch": 0}

    def _Infinite_Iter(self):
        it = iter(self.dataLoader)
        while True:
            try:
                ret = next(it)
                yield ret
            except StopIteration:
                it = iter(self.dataLoader)

    #為什麼不直接吃一個model就不用判斷了，因為這樣寫起來比較ＯＯ
    def Show_Model_Architecture(self) -> None:
        assert self.status["hasModel"], "No model in the BERT_Family object."
        for name, module in self.model.named_children():
            if name == "bert":
                for n, _ in module.named_children():
                    print(f"{name}:{n}")
            else:
                print("{:15} {}".format(name, module))

    def Show_Model_Configuration(self) -> None:
        assert self.status["hasModel"], "No model in the BERT_Family object."
        print(self.model.config)

    def Show_Status(self) -> None:
        print("\n".join("{}\t{}".format(k, v) for k, v in self.status.items()))
    

    def Training(self, epochs = 50, optimizer = None):
        assert self.status["hasModel"], "No model in the BERT_Family object."
        if not optimizer: optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.status["isTrained"] = True
        self.model.to(self.device)
        self.model.train()
        print("start2")
        #start train
        for epoch in range(epochs):
            for _ in range(self.iteration):
                running_loss = 0.0
                inputData, target = next(self.dataIter)
                tokens_tensors, segments_tensors, masks_tensors = inputData['input_ids'].to(self.device), inputData['token_type_ids'].to(self.device), inputData['attention_mask'].to(self.device)
                tokens_tensors, segments_tensors, masks_tensors = tokens_tensors.squeeze(1), segments_tensors.squeeze(1), masks_tensors.squeeze(1)
                target = target.to(self.device)

                optimizer.zero_grad()

                # forward pass
                outputs = self.model(input_ids=tokens_tensors, 
                                    token_type_ids=segments_tensors, 
                                    attention_mask=masks_tensors, 
                                    labels=target)

                loss = outputs[0]
                # backward
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # 計算分類準確率
            #_, acc = get_predictions(model, trainloader, compute_acc=True)

            print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))
            self.status["accumulateEpoch"] += 1
        self.status["trainingTimes"] += 1
        return loss
    

#Sequence Classification
class BF_Classification(BERT_Family):
    def __init__(self, **kwargs) -> None:
        print("bas")
        super().__init__(**kwargs)
        self.labelLength = None
        self.status["BERT_Type"].append("BF_Classification")

    def Set_Dataset(self, rawData, rawTarget, batchSize = 100, **kwargs) -> None:
        """ 
        Input:
        rawData: n by p, n: observations (total sequence). p: number of sequences in each case.
        rawTarget: a list of n-length.
        **kwargs: The argument in DataLoader

        Return 3 object:
        dataset, dataloader, dataloader with iter
        """
        self.status["hasData"] = True
        self.dataset = Classification_Dataset(rawData = rawData, rawTarget = rawTarget, tokenizer = self.tokenizer, maxLength = self.maxLength)
        self.dataLoader = data.DataLoader(self.dataset, batch_size=batchSize, **kwargs)
        self.iteration = int(rawData.shape[0] / batchSize)
        self.labelLength = len(self.dataset.rawTarget_dict)
        self.dataIter = self._Infinite_Iter()
        #return self.dataset, self.dataLoader, self.dataIter
    
    def Create_Model(self, labelLength, pretrainedModel = None, **kwargs) -> None:
        assert self.labelLength & (self.labelLength == labelLength), "Mismatch on the length of labels."
        self.status["hasModel"] = True
        if not pretrainedModel: pretrainedModel = self.pretrainedModel
        self.model = BertForSequenceClassification.from_pretrained(pretrainedModel, num_labels = labelLength, **kwargs)   
        #return self.model


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


            
#build customize model                 
         
#dataset / dataloader
#

def Get_Learnable_Parameters_Size(model):
    tmp = [p for p in model.parameters() if p.requires_grad]
    return sum(list(map(lambda x: len(x.view(-1)), tmp)))


########## Train and Pred and Eval... ##########
#build




##############################test######################################
class testaaa(object):
    def __init__(self) -> None:
        pass
    def myTest(self, a, b):
        return a+b
t = testaaa()
d
import os
os.getcwd()
#df_train = pd.read_csv("~/Downloads/train.csv")
df_train = pd.read_csv("~/BERT_Family/data/news/train.csv")
empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train = df_train[~ empty_title]
temp = df_train[['title1_zh', "title2_zh"]]
#temp_2 = df_train[['title1_zh']]
temp_t = df_train[['label']]
temp_t = temp_t.values.squeeze()
del b
c = BF_Classification(pretrainedModel = "bert-base-chinese", maxLength = 70)
c.Set_Dataset(rawData = temp, rawTarget = temp_t, batchSize=64, shuffle=True)
c.Create_Model(labelLength=c.labelLength)
c.Show_Model_Architecture()
c.Show_Status()
loss = c.Training(20)

s = next(c.dataIter)
test = s[0]['input_ids']
test.squeeze(1).shape
torch.tensor(s[0])

tokens_tensors, segments_tensors, masks_tensors = loss['input_ids'].to("cpu"), loss['token_type_ids'].to("cpu"), loss['attention_mask'].to("cpu")

a, b = next(b.dataIter)
len(a)
len(b.dataset)
b.dataset.rawData.shape
params_size = Get_Learnable_Parameters_Size(b.model) # == b.model.num_parameters()

eval("b.Show_Status()")
type(b)
type(b)
isinstance(b, "BERT_Family")
isinstance({}, dict) 
s
torch.tensor(s)
len(s[0].view(-1))
len(s)
s.shape()
type(s)
sum(list(map(lambda x: len(x.view(-1)), s)))
for i in s:
    print(i.shape)
len(s[13])

#test change

test = torch.nn.Sequential(b.model, torch.nn.Linear(10, 10), torch.nn.Linear(50, 10))

a, b = next(train_iter)
a, b



tmp = datasets.STSB(root='.data', split=('train', 'dev', 'test'))
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
vocab = t.vocab
print("字典大小：", len(vocab))
tmp = list(vocab)
tmp[16000:16010]
import random
random.sample(tmp, 10)
vocab["瓶"]

print(123)

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

temp = df_train[['title1_zh', "title2_zh"]]
temp_t = df_train[['label']]
pd.unique(temp_t)
pd.DataFrame(temp_t)
pd.Series(temp_t)

temp2 = temp.iloc[[1,32,300,555,10007]]
temp2.values
temp2.iloc[:,0]

tokenizer(temp2.iloc[:,0].values)
tokenizer.tokenize(temp2.iloc[:,0])

torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
s = tokenizer(x.iloc[0,0] + '[SEP]' + x.iloc[0,1], padding="max_length", max_length=50, truncation = True)
s


a
b
a['input_ids']
a, b, c, d= next(train_iter)
a
a["input_ids"]
type(a)
dir(a)
b[4,]
len(b[1,])
temp[1,0]
temp.iloc[1,1]
tokenizer.encode_plus(temp.iloc[1,0], temp.iloc[1, 1], padding="max_length", max_length=100, truncation = True)

a['input_ids'].view(64, 100).shape
a
a[0,:]
a.shape
print(a.iloc[100])

def fn(x, y):
    return temp.iloc[:, x] + '[SEP]' + temp.iloc[:, y] + '[SEP]'
a = reduce(fn, [0,1])
a = pd.DataFrame(a)
a.iloc[2,:]
s = reduce(lambda x,y: temp.iloc[:, x] + '[SEP]' + temp.iloc[:, y] + '[SEP]', [0,1,2])
s
torch.concat((torch.tensor([0]*10), torch.tensor([1]*30)))
a = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0, 0])
b = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
torch.concat((a,b))

##############################test######################################
