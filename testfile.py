import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from transformers import BertTokenizerFast
import pandas as pd
from transformers import BertForSequenceClassification, BertForQuestionAnswering
#from accelerate import 
from tqdm.auto import tqdm
import json

########## Modules ##########
#不想export出去的
#One or two sentence with one dim label
class Classification_Dataset(Dataset):
    def __init__(self, rawData, tokenizer, rawTarget = None, maxLength = 100) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.rawData, self.rawTarget = pd.DataFrame(rawData), rawTarget
        self.maxLength = maxLength
        assert self.rawData.shape[1] <= 2, "Only accept one or two sequences as the input argument."
        if rawTarget is not None:
            self.rawTarget_dict = {}
            for i, ele in enumerate(pd.unique(rawTarget)):
                self.rawTarget_dict[ele] = i


    def __len__(self):
        return self.rawData.shape[0]


    def __getitem__(self, idx):
        if self.rawData.shape[1] == 2:
            result = self.tokenizer.encode_plus(self.rawData.iloc[idx, 0], self.rawData.iloc[idx, 1], padding="max_length", max_length=self.maxLength, truncation = True, return_tensors = 'pt')
        else:
            result = self.tokenizer.encode_plus(self.rawData.iloc[idx, 0], padding="max_length", max_length=self.maxLength, truncation = True, return_tensors = 'pt')
        return result, torch.tensor(self.rawTarget_dict[self.rawTarget[idx]]) if self.rawTarget is not None else result


class BERT_Family(nn.Module):
    def __init__(self, pretrainedModel = 'bert-base-uncased', maxLength = 100,\
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()
        print("Using device: ", device)
        self.device = device
        self.pretrainedModel = pretrainedModel
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrainedModel)
        self.maxLength = maxLength
        self.trainDataLoader, self.devDataLoader, self.testDataLoader =  None, None, None
        self.model, self.batchSize = None, 100
        self.status = {"BERT_Type": ["BERT_Family"],\
                        "hasData": False,\
                        "hasModel": False,\
                        "isTrained": False,\
                        "accumulateEpoch": 0}


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

    
    def Load_Model(self) -> None:
        pass


    def Save_Model(self) -> None:
        pass


class BF_Classification(BERT_Family):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.labelLength = None
        self.status["BERT_Type"].append("BF_Classification")


    def Set_Dataset(self, rawData: pd.DataFrame, rawTarget: list, tokenizer = None, dataType = "train", batchSize = 100, **kwargs):
        """ 
        Input:
        rawData: n by p, n: observations (total sequence). p: number of sequences in each case.
        rawTarget: a list of n-length.
        type: train, dev, test, other
        **kwargs: The argument in DataLoader

        Return 3 object:
        dataset, dataloader, dataloader with iter
        """ 
        if tokenizer is None: tokenizer = self.tokenizer
        tmpDataset = Classification_Dataset(rawData = rawData, rawTarget = rawTarget, tokenizer = self.tokenizer, maxLength = self.maxLength)
        if dataType not in ["train", "test", "dev"]: return DataLoader(tmpDataset, batch_size=batchSize, **kwargs)
        exec("self." + dataType + "DataLoader" + " = DataLoader(tmpDataset, batch_size=batchSize, **kwargs)")

        self.batchSize = batchSize
        self.status["hasData"] = True
        self.labelLength = len(tmpDataset.rawTarget_dict)
        

    def Create_Model(self, labelLength:int, pretrainedModel = None, **kwargs):
        assert (self.labelLength is not None) & (self.labelLength == labelLength), "Mismatch on the length of labels."
        self.status["hasModel"] = True
        if not pretrainedModel: pretrainedModel = self.pretrainedModel
        self.model = BertForSequenceClassification.from_pretrained(pretrainedModel, num_labels = labelLength, **kwargs).to(self.device)   
        #這裡再新增多一點東西 roberta
        return self.model
    

    def Forecasting(self, data, model, tokenizer, batchSize = 100, **kwargs):
        tmpDataset = Classification_Dataset(rawData = data, tokenizer = tokenizer, maxLength = self.maxLength)
        dataLoader = DataLoader(tmpDataset, batch_size=batchSize, **kwargs)
        predictions = None
        with torch.no_grad():
            for df in dataLoader if eval else tqdm(dataLoader):                
                input = [t for t in df if t is not None]        
                outputs = model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device))
                _, pred = torch.max(outputs[0].data, 1)
                predictions = pred if predictions is None else torch.cat((predictions, pred))
        return predictions


    def Testing(self, model, dataLoader, eval = False, **kwargs):
        predictions = None
        total, correct, loss= 0, 0, 0
        with torch.no_grad():
            for df in dataLoader if eval else tqdm(dataLoader):                
                input = [t for t in df if t is not None]        
                outputs = model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device))
                logits = outputs[0]
                _, pred = torch.max(logits.data, 1)
                labels = input[1].to(self.device)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                predictions = pred if predictions is None else torch.cat((predictions, pred))
        acc = correct / total
        return predictions, acc, loss
       

    def Training(self, trainDataLoader, devDataLoader = None, epochs = 50, optimizer = None, eval = False):
        #這裡要補checkpoint
        assert self.status["hasModel"], "No model in the BERT_Family object."
        if not optimizer: optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.status["isTrained"] = True
        outputs = None

        self.model.train()
        print("Start Training ...")
        for epoch in range(epochs):
            total, correct, running_loss = 0, 0, 0
            for df in tqdm(trainDataLoader):
                input = [t for t in df if t is not None]
                optimizer.zero_grad()
                outputs = self.model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device), 
                                labels = input[1].to(self.device))
                outputs[0].backward()
                optimizer.step()
                running_loss += outputs[0].item()
                _, pred = torch.max(outputs[1], 1)
                total += input[1].size(0)
                correct += (pred == input[1]).sum().item()

            if eval & (devDataLoader is not None):
                self.model.eval()
                _, evalAcc, evalLoss = self.Testing(self.model, devDataLoader, eval = True)
                self.model.train()
                print('[epoch %d] loss: %.3f, TrainACC: %.3f, EvalACC: %.3f, EvalLoss: %.3f' %(epoch + 1, running_loss, correct/total, evalAcc, evalLoss))
            else:
                print('[epoch %d] loss: %.3f, ACC: %.3f' %(epoch + 1, running_loss, correct/total))
            self.status["accumulateEpoch"] += 1
        


class BF_QA(BERT_Family):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.status["BERT_Type"].append("BF_QA")

    
    def Create_Model(self, pretrainedModel = None, **kwargs):
        self.status["hasModel"] = True
        if not pretrainedModel: pretrainedModel = self.pretrainedModel
        self.model = BertForQuestionAnswering.from_pretrained(pretrainedModel, **kwargs).to(self.device)   
        #這裡再新增多一點東西 roberta
        return self.model
            

    def Translate(self):
        pass
    
    
    def Set_Dataset(self, questionsDic: dict, paragraphsList: list, tokenizer = None, dataType = "train", batchSize = 100, **kwargs):
        assert self.__Test_Data_Valid(questionsDic, paragraphsList)
        if tokenizer is None: tokenizer = self.tokenizer
        if dataType in ["test", "dev"]: batchSize = 1

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        questionsTokenized = tokenizer([q["question_text"] for q in questionsDic], add_special_tokens=False)
        paragraphsTokenized = tokenizer(paragraphsList, add_special_tokens=False)
        tmpDataset = QA_Dataset(dataType, questionsDic, questionsTokenized, paragraphsTokenized)

        if dataType not in ["train", "test", "dev"]: return DataLoader(tmpDataset, batch_size=batchSize, shuffle=True, pin_memory=True, **kwargs)
        exec("self." + dataType + "DataLoader" + " = DataLoader(tmpDataset, batch_size=batchSize, **kwargs)")
        self.batchSize = batchSize
        self.status["hasData"] = True


    def Testing(self, model, data, output, tokenizer, eval = False):
        # There is a bug and room for improvement in postprocessing 
        # Hint: Open your prediction file to see what is wrong    
        answer = ''
        max_prob = float('-inf')
        num_of_windows = data[0].shape[1]
        for k in range(num_of_windows):
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)
            prob = start_prob + end_prob
            
            # Replace answer if calculated probability is larger than previous windows
            if prob > max_prob:
                max_prob = prob
                # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
        return answer.replace(' ','')


    def Training(self, trainDataLoader, devDataLoader = None, epochs = 50, optimizer = None, eval = False, logging_step = 100):
        #這裡要補checkpoint
        assert self.status["hasModel"], "No model in the BERT_Family object."
        logging_step = logging_step
        if not optimizer: optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.status["isTrained"] = True
        output = None

        self.model.train()
        print("Start Training ...")
        for epoch in range(epochs):
            running_loss = train_acc = 0
            for data in tqdm(trainDataLoader):	
                optimizer.zero_grad()
                data = [i.to(self.device) for i in data]
                output = self.model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
                start_index = torch.argmax(output.start_logits, dim=1)
                end_index = torch.argmax(output.end_logits, dim=1)
                
                train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
                running_loss += output.loss

                output.loss.backward()
                optimizer.step()
            print('[epoch %d] loss: %.3f, ACC: %.3f' %(epoch + 1, running_loss, train_acc))
            self.status["accumulateEpoch"] += 1


    def __Test_Data_Valid(self, questionsDic: dict, paragraphsList: list, detectNum = 10) -> bool:
        keysDomain = ['paragraph_id', 'question_text', 'answer_text', 'answer_start', 'answer_end']
        n = len(questionsDic) - 1
        if not (questionsDic[n] == (len(paragraphsList) - 1)): return False, "Mismatch length between questionsDic and paragraphsList."
        
        detectList = torch.cat((torch.round(torch.rand(detectNum) * n), torch.tensor([0]), torch.tensor([n])))
        results = torch.tensor([keys in questionsDic[int(i)].keys() for i in detectList for keys in keysDomain])
        print(results)
        return torch.all(results), "Mismatch keys between questionsDic and keysDomain."



class QA_Dataset(Dataset):
    def __init__(self, dataType, questions, tokenizedQuestions, tokenizedParagraphs, maxQuestionLen = 40, maxParagraphLen = 150, stride = 80):
        self.dataType = dataType
        self.questions = questions
        self.tokenizedQuestions = tokenizedQuestions
        self.tokenizedParagraphs = tokenizedParagraphs
        self.maxQuestionLen = maxQuestionLen
        self.maxParagraphLen = maxParagraphLen
        self.stride = stride
        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.maxSeqLen = 1 + self.maxQuestionLen + 1 + self.maxParagraphLen + 1


    def __len__(self):
        return len(self.questions)


    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenizedQuestion = self.tokenizedQuestions[idx]
        tokenizedParagraph = self.tokenizedParagraphs[question["paragraph_id"]]
        if self.dataType in ["test", "dev"]:
            inputIdsList, tokenTypeIdsList, attentionMaskList = [], [], []
            #Split paragraph into several windows
            for i in range(0, len(tokenizedParagraph), self.stride):
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                inputIdsQuestion = [101] + tokenizedQuestion.ids[:self.maxQuestionLen] + [102]
                inputIdsParagraph = tokenizedParagraph.ids[i : i + self.maxParagraphLen] + [102]
                
                # Pad sequence and obtain inputs to model
                inputIds, tokenTypeIds, attentionMask = Padding(inputIdsQuestion, inputIdsParagraph, maxSeqLen = self.maxSeqLen)
                inputIdsList.append(inputIds)
                tokenTypeIdsList.append(tokenTypeIds)
                attentionMaskList.append(attentionMask)
            return torch.tensor(inputIdsList), torch.tensor(tokenTypeIdsList), torch.tensor(attentionMaskList)
        else:
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenizedParagraph  
            ansStartToken = tokenizedParagraph.char_to_token(question["answer_start"])
            ansEndToken = tokenizedParagraph.char_to_token(question["answer_end"])

            mid = (ansStartToken + ansEndToken) // 2 #create a window which contain the answer
            paragraphStart = max(0, min(mid - self.maxParagraphLen // 2, len(tokenizedParagraph) - self.maxParagraphLen))
            paragraphEnd = paragraphStart + self.maxParagraphLen
    
            inputIdsQuestion = [101] + tokenizedQuestion.ids[:self.maxQuestionLen] + [102] #(101: CLS, 102: SEP)
            inputIdsParagraph = tokenizedParagraph.ids[paragraphStart : paragraphEnd] + [102]		
            
            # Convert answer's start/end positions in tokenizedParagraph to start/end positions in the window  
            ansStartToken += len(inputIdsQuestion) - paragraphStart
            ansEndToken += len(inputIdsQuestion) - paragraphStart

            inputIds, tokenTypeIds, attentionMask = Padding(inputIdsQuestion, inputIdsParagraph, maxSeqLen = self.maxSeqLen)
            return torch.tensor(inputIds), torch.tensor(tokenTypeIds), torch.tensor(attentionMask), ansStartToken, ansEndToken

            
def Padding(inputIdsQuestion, inputIdsParagraph, maxSeqLen):
    paddingLen = maxSeqLen - len(inputIdsQuestion) - len(inputIdsParagraph)
    inputIds = inputIdsQuestion + inputIdsParagraph + [0] * paddingLen
    tokenTypeIds = [0] * len(inputIdsQuestion) + [1] * len(inputIdsParagraph) + [0] * paddingLen
    attentionMask = [1] * (len(inputIdsQuestion) + len(inputIdsParagraph)) + [0] * paddingLen
    return inputIds, tokenTypeIds, attentionMask

def Auto_Build_Model(data = None, target = None, pretrainedModel = None, **kwargs):
    if not pretrainedModel:
        #prepare pretrainedModel
        pass
    pass

def Read_Json_Data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data

def Get_Learnable_Parameters_Size(model):
    tmp = [p for p in model.parameters() if p.requires_grad]
    return sum(list(map(lambda x: len(x.view(-1)), tmp)))

def Infinite_Iter(dataLoader):
    it = iter(dataLoader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(dataLoader)

#build customize model                 
         
#dataset / dataloader
#




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

""" #test
from datasets import load_dataset
num = 200
dataset = load_dataset('glue', 'mrpc', split='train')
dataset["sentence1"]
df = pd.DataFrame([dataset["sentence1"], dataset["sentence2"]])
df = df.transpose()
target = dataset["label"]
#dev
dev = load_dataset('glue', 'mrpc', split='validation')
dev["sentence1"]
dfdev = pd.DataFrame([dev["sentence1"], dev["sentence2"]])
dfdev = dfdev.transpose()
targetdev = dev["label"]
#test
test = load_dataset('glue', 'mrpc', split='test')
test["sentence1"]
dftest = pd.DataFrame([test["sentence1"], test["sentence2"]])
dftest = dftest.transpose()
targettest = test["label"]


target = target[:num]
df = df.iloc[:num, :]

targetdev = targetdev[:100]
dfdev = dfdev.iloc[:100, :]

targettest = targettest[:100]
dftest = dftest.iloc[:100, :]


b = BF_Classification(pretrainedModel = "bert-base-uncased", maxLength = 50)
b.Set_Dataset(df, target, dataType = "train", batchSize=100, shuffle=True)
b.Set_Dataset(dfdev, targetdev, dataType = "dev", batchSize=100, shuffle=True)
b.Create_Model(b.labelLength)
b.Show_Model_Architecture(); b.Show_Status()
a = b.Training(trainDataLoader=b.trainDataLoader, devDataLoader=b.devDataLoader, epochs = 1, eval=True)
f = b.Forecasting(data = dftest, model = b.model, tokenizer = "bert-base-uncased", batchSize = 100)
"""

""" idk

#evaluation
dataset = load_dataset('glue', 'mrpc', split='test')
dataset["sentence1"]
df = pd.DataFrame([dataset["sentence1"], dataset["sentence2"]])
df = df.transpose()
target = dataset["label"]
pred, acc = b.Testing(b.model, df, target); acc
 """

""" #MRPC 0.7386

from datasets import load_dataset
dataset = load_dataset('glue', 'mrpc', split='train')
dataset["sentence1"]
df = pd.DataFrame([dataset["sentence1"], dataset["sentence2"]])
df = df.transpose()
target = dataset["label"]
b = BF_Classification(pretrainedModel = "bert-base-uncased", maxLength = 50)
b.Set_Dataset(df, target, batchSize=100, shuffle=True)
b.Create_Model(b.labelLength)
b.Show_Model_Architecture(); b.Show_Status()
a = b.Training(5)

#evaluation
dataset = load_dataset('glue', 'mrpc', split='test')
dataset["sentence1"]
df = pd.DataFrame([dataset["sentence1"], dataset["sentence2"]])
df = df.transpose()
target = dataset["label"]
pred, acc = b.Testing(b.model, df, target); acc
 """

"""  #CoLA -> OK, 0.83, epoch=10 (bert-base-uncased)
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
"""

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



# QA
from zipfile import ZipFile, Path
from io import StringIO
dataDir = "BERT_Family/data/QA_data.zip"
#dataDir = "/home/ubuntu/work/BERT_Family/data/QA_data.zip"
zipped = Path(dataDir, at="QA_data/" + "hw7_train.json")


def read_data(zipped):
    with open(zipped.read_text(), 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

#d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
#tmp = "BERT_Family/data/QA_data/"
""" 
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]
 """
zipped = Path(dataDir, at="QA_data/" + "hw7_train.json")
train_questions, train_paragraphs = read_data(zipped)

zipped = Path(dataDir, at="QA_data/" + "hw7_dev.json")
dev_questions, dev_paragraphs = read_data(zipped)

zipped = Path(dataDir, at="QA_data/" + "hw7_test.json")
test_questions, test_paragraphs = read_data(zipped)

""" 
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
 """
b = BF_QA(pretrainedModel = "bert-base-chinese")
b.Set_Dataset(train_questions, train_paragraphs, dataType = "train", batchSize=100, shuffle=True, pin_memory=True)
b.Create_Model()
b.Show_Model_Architecture(); b.Show_Status()
b.Training(trainDataLoader = b.trainDataLoader, epochs = 2)
# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
""" train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
 """



##############################test######################################

