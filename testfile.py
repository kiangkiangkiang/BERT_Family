import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from transformers import BertTokenizerFast
import pandas as pd
from transformers import BertForSequenceClassification
#from accelerate import 
from tqdm.auto import tqdm


########## Modules ##########
#不想export出去的
#One or two sentence with one dim label
class Classification_Dataset(Dataset):
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
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrainedModel)
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
            print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))
            self.status["accumulateEpoch"] += 1
        return loss
    
#Sequence Classification
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
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batchSize, **kwargs)
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

    def Testing(self, model, testingData, testingTarget, **kwargs):
        dataset = Classification_Dataset(rawData = testingData, rawTarget = testingTarget, tokenizer = self.tokenizer, maxLength = self.maxLength)
        dataloader = DataLoader(dataset, batch_size=self.batchSize, **kwargs)
        predictions = None
        correct = 0
        total = 0
        tmp = None
        with torch.no_grad():
            for df in tqdm(dataloader):                
                input = [t for t in df if t is not None]        
                outputs = model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device))
                input[1] = input[1].to(self.device)
                
                logits = outputs[0]
                print(logits)
                _, pred = torch.max(logits.data, 1)
                
                total += input[1].size(0)
                correct += (pred == input[1]).sum().item()
                    
                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred))
        acc = correct / total
        return predictions, acc
       
    """ # 讓模型跑在 GPU 上並取得訓練集的分類準確率
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    _, acc = get_predictions(model, trainloader, compute_acc=True)
    print("classification acc:", acc) """

#
class BF_QA(BERT_Family):
    def __init__(self) -> None:
        super().__init__()
        self.status["BERT_Type"].append("BF_QA")



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


 #CoLA -> OK
#tmp = "data/glue_data/coLA/train.tsv"
from zipfile import ZipFile, Path
from io import StringIO
dataDir = "/home/ubuntu/work/BERT_Family/data/CoLA.zip"
zipped = Path(dataDir, at="CoLA/train.tsv")
d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
target = d.iloc[:,1]
df = d.iloc[:, 3]
b = BF_Classification(pretrainedModel = "bert-base-uncased", maxLength = 50)
b.Set_Dataset(df, target, batchSize=100, shuffle=True)
b.Create_Model(b.labelLength)
b.Show_Model_Architecture(); b.Show_Status()
a = b.Training(10)

#evaluation
zipped = Path(dataDir, at="CoLA/dev.tsv")
d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
target = d.iloc[:,1]
df = d.iloc[:, 3]
pred, acc = b.Testing(b.model, df, target)
63

#testing
zipped = Path(dataDir, at="CoLA/test.tsv")
d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
target = d.iloc[:,1]
df = d.iloc[:, 3]
b.Testing(b.model, df, target)
""" #news -> OK
##########in mac
#dataDir = "~/Downloads/news.csv"
from zipfile import ZipFile
dataDir = "BERT_Family/data/news/news.zip"#in ubutn
tmp = ZipFile(dataDir)
from io import StringIO
from zipfile import Path
zipped = Path(dataDir, at="train.csv")
df_train = pd.read_csv(StringIO(zipped.read_text()))

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
testSize = 100
testIdx = random.sample(range(temp.shape[0]), testSize)

psize = 100
pIdx = random.sample(range(temp.shape[0]), testSize)

x = temp.iloc[testIdx]
y = temp_t[testIdx]
testx = temp.iloc[pIdx]
testy = temp_t[pIdx]
b = BF_Classification(pretrainedModel = "bert-base-chinese", maxLength = 70)
b.Set_Dataset(rawData = x, rawTarget = y, batchSize=100, shuffle=True)
b.Create_Model(labelLength=b.labelLength)
b.Show_Model_Architecture(); b.Show_Status()
a = b.Training(1)
"""


""" # QA

tmp = "BERT_Family/data/QA_data/"
df_train = pd.read_json(tmp)
import json
myData = 0
with open(tmp, 'r', encoding="utf-8") as reader:
    myData = json.load(reader)

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]
train_questions, train_paragraphs = read_data(tmp + "hw7_train.json")
dev_questions, dev_paragraphs = read_data(tmp + "hw7_dev.json")
test_questions, test_paragraphs = read_data(tmp + "hw7_test.json")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 


train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)



class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 150
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 150

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask
train_questions[1]
train_questions_tokenized[1]
train_paragraphs_tokenized[1]
train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 32

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

len(myData["paragraphs"])

##############################test######################################
 """