import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch
from torchtext import datasets
from transformers import BertTokenizer
import pandas as pd
from transformers import BertForSequenceClassification
from functools import reduce
#from accelerate import 
from tqdm.auto import tqdm


########## Modules ##########
#不想export出去的
#One or two sentence with one dim label
class Classification_Dataset(data.Dataset):
    def __init__(self, rawData, rawTarget, tokenizer, maxLength = 100) -> None:
        super().__init__()
        print("123")
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
        if self.rawData.shape[1] == 2:
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
        self.dataset, self.dataLoader = None, None
        self.model, self.iteration, self.batchSize = None, 0, 100
        #self. = ()
        self.status = {"BERT_Type": ["BERT_Family"],\
                        "hasData": False,\
                        "hasModel": False,\
                        "isTrained": False,\
                        "accumulateEpoch": 0,\
                        "accumulateIteration": 0}

    def Infinite_Iter(self, dataLoader):
        it = iter(dataLoader)
        while True:
            try:
                ret = next(it)
                yield ret
            except StopIteration:
                it = iter(dataLoader)

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
        self.model.train()
        print("start2")
        #start train
        for epoch in range(epochs):
            for df in tqdm(self.dataLoader):

                running_loss = 0.0
                input = [t for t in df if t is not None]
                optimizer.zero_grad()
                
                # forward pass
                outputs = self.model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device), 
                                labels = input[1].to(self.device))
                
                #outputs = self.model(**input[0].to(self.device), labels = input[1].to(self.device))
                loss = outputs[0]
                # backward
                loss.backward()
                #self..backward(loss)
                optimizer.step()
                running_loss += loss.item()
                self.status["accumulateIteration"] += 1
            # 計算分類準確率
            #_, acc = get_predictions(model, trainloader, compute_acc=True)
            

            print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))
            self.status["accumulateEpoch"] += 1
        return loss
    
#Sequence Classification 暫時好了
class BF_Classification(BERT_Family):
    def __init__(self, **kwargs) -> None:
        print("a")
        super().__init__(**kwargs)
        self.labelLength = None
        self.status["BERT_Type"].append("BF_Classification")

    def Set_Dataset(self, rawData, rawTarget, batchSize = 100, **kwargs):
        """ 
        Input:
        rawData: n by p, n: observations (total sequence). p: number of sequences in each case.
        rawTarget: a list of n-length.
        **kwargs: The argument in DataLoader

        Return 3 object:
        dataset, dataloader, dataloader with iter
        """
        self.dataset = Classification_Dataset(rawData = rawData, rawTarget = rawTarget, tokenizer = self.tokenizer, maxLength = self.maxLength)
        self.dataLoader = data.DataLoader(self.dataset, batch_size=self.batchSize, **kwargs)
        #self.dataIter = self._Infinite_Iter(self.dataLoader)

        self.batchSize = batchSize
        self.status["hasData"] = True
        self.iteration = int(rawData.shape[0] / batchSize)
        self.labelLength = len(self.dataset.rawTarget_dict)
        

    def Create_Model(self, labelLength, pretrainedModel = None, **kwargs) -> None:
        assert (self.labelLength is not None) & (self.labelLength == labelLength), "Mismatch on the length of labels."
        self.status["hasModel"] = True
        if not pretrainedModel: pretrainedModel = self.pretrainedModel
        self.model = BertForSequenceClassification.from_pretrained(pretrainedModel, num_labels = labelLength, **kwargs).to(self.device)   
        #return self.model
    
    def Forecasting(self):
        pass

    def Testing(self, model, testingData, testingTarget, compute_acc=True, **kwargs):
        dataset = Classification_Dataset(rawData = testingData, rawTarget = testingTarget, tokenizer = self.tokenizer, maxLength = self.maxLength)
        dataloader = data.DataLoader(dataset, batch_size=self.batchSize, **kwargs)
        predictions = None
        correct = 0
        total = 0
        tmp = None
        with torch.no_grad():
            # 遍巡整個資料集
            for df in tqdm(dataloader):
                # 將所有 tensors 移到 GPU 上
                
                input = [t for t in df if t is not None]        
                # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
                # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
                #tokens_tensors, segments_tensors, masks_tensors = tmp[0]["input_ids"].squeeze(1), tmp[0]["token_type_ids"].squeeze(1), tmp[0]["attention_mask"].squeeze(1)
                #label = tmp[1]
                outputs = model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device))
                
                logits = outputs[0]
                print(logits)
                _, pred = torch.max(logits.data, 1)
                
                # 用來計算訓練集的分類準確率
                if compute_acc:
                    total += input[1].size(0)
                    correct += (pred == input[1]).sum().item()
                    
                # 將當前 batch 記錄下來
                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred))
        
        if compute_acc:
            acc = correct / total
            return predictions, acc
        return predictions
       
    """ # 讓模型跑在 GPU 上並取得訓練集的分類準確率
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    _, acc = get_predictions(model, trainloader, compute_acc=True)
    print("classification acc:", acc) """

#12/30測試one seq和two seq的問題是否都能做，接著開始實作QA類型問題或是多label問題，最後實作視覺化function -> 結束
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


def Auto_Build_Model(data = None, target = None, pretrainedModel = None, **kwargs):
    if not pretrainedModel:
        #prepare pretrainedModel
        pass
    pass
#build customize model                 
         
#dataset / dataloader
#

def Get_Learnable_Parameters_Size(model):
    tmp = [p for p in model.parameters() if p.requires_grad]
    return sum(list(map(lambda x: len(x.view(-1)), tmp)))


########## Train and Pred and Eval... ##########
#build




##############################test######################################

""" other
##########in ubutn
from zipfile import ZipFile
dataDir = "/home/ubuntu/work/BERT_Family/data/news/news.zip"#in ubutn
tmp = ZipFile("/BERT_Family/data/news/news.zip")
from io import StringIO
from zipfile import Path
zipped = Path(dataDir, at="train.csv")
df_train = pd.read_csv(StringIO(zipped.read_text()))
#########


##########in mac
#dataDir = "~/Downloads/news.csv"
df_train = pd.read_csv(dataDir)
############


empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train = df_train[~ empty_title]
temp = df_train[['title1_zh', "title2_zh"]]
#temp_2 = df_train[['title1_zh']]
temp_t = df_train[['label']]
temp_t = temp_t.values.squeeze()

import random
testSize = 5000
testIdx = random.sample(range(temp.shape[0]), testSize)

psize = 100
pIdx = random.sample(range(temp.shape[0]), testSize)

x = temp.iloc[testIdx]
y = temp_t[testIdx]
testx = temp.iloc[pIdx]
testy = temp_t[pIdx]
c = BF_Classification(pretrainedModel = "bert-base-chinese", maxLength = 70)
c.Set_Dataset(rawData = x, rawTarget = y, batchSize=200, shuffle=True)
c.Create_Model(labelLength=c.labelLength)
c.Show_Model_Architecture(); c.Show_Status()
a = c.Training(1)


os.system("echo %PYTORCH_CUDA_ALLOC_CONF%")

import GPUtil
from GPUtil import showUtilization as gpu_usage
import gc
kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
gc.collect()

torch.cuda.empty_cache()
gpu_usage()
a[1].shape
a, b = c.Testing(model = c.model, testingData = testx, testingTarget = testy);b


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


torch.concat((torch.tensor([0]*10), torch.tensor([1]*30)))
a = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0, 0])
b = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
torch.concat((a,b))




##QA
import os
print(os.getcwd())
from zipfile import ZipFile
#dataDir = "/home/ubuntu/work/BERT_Family/data/news/news.zip"#in ubutn
tmp = ZipFile("BERT_Family/data/QA_data.zip")
from io import StringIO
from zipfile import Path
zipped = Path(tmp, at="hw7_train.json")
#df_train = pd.read_csv(StringIO(zipped.read_text()))
tmp = "BERT_Family/data/QA_data/hw7_train.json"

df_train = pd.read_json(tmp)
import json
myData = 0
with open(tmp, 'r', encoding="utf-8") as reader:
    myData = json.load(reader)
myData["questions"], myData["paragraphs"]
del myData

 """


""" #CoLA -> OK
tmp = "data/glue_data/coLA/train.tsv"
d = pd.read_csv(tmp, sep="\t")
target = d.iloc[:100,1]
df = d.iloc[:100, 3]
b = BF_Classification(pretrainedModel = "bert-base-uncased", maxLength = 50)
b.Set_Dataset(df, target, batchSize=100, shuffle=True)
b.Create_Model(b.labelLength)
b.Show_Model_Architecture(); b.Show_Status()
a = b.Training(1)
 """

#news
##########in mac
#dataDir = "~/Downloads/news.csv"
df_train = pd.read_csv("BERT_Family/data/news/")
############


empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train = df_train[~ empty_title]
temp = df_train[['title1_zh', "title2_zh"]]
#temp_2 = df_train[['title1_zh']]
temp_t = df_train[['label']]
temp_t = temp_t.values.squeeze()

import random
testSize = 5000
testIdx = random.sample(range(temp.shape[0]), testSize)

psize = 100
pIdx = random.sample(range(temp.shape[0]), testSize)

x = temp.iloc[testIdx]
y = temp_t[testIdx]
testx = temp.iloc[pIdx]
testy = temp_t[pIdx]
c = BF_Classification(pretrainedModel = "bert-base-chinese", maxLength = 70)
c.Set_Dataset(rawData = x, rawTarget = y, batchSize=200, shuffle=True)
c.Create_Model(labelLength=c.labelLength)
c.Show_Model_Architecture(); c.Show_Status()
a = c.Training(1)


##############################test######################################
