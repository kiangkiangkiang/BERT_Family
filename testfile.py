import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from transformers import BertTokenizerFast
import pandas as pd
from transformers import BertForSequenceClassification, BertForQuestionAnswering, BertForTokenClassification
#from accelerate import 
from tqdm.auto import tqdm
import json
from datasets import load_dataset, dataset_dict

########## Modules ##########
#One or two sentence with one dim label

class BERTFamily(nn.Module):
    def __init__(self, pretrained_model = 'bert-base-uncased', max_length = 100,\
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()
        print("Using device: ", device)
        self.device = device
        self.pretrained_model = pretrained_model
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        self.max_length = max_length
        self.train_data_loader, self.dev_data_loader, self.test_data_loader =  None, None, None
        self.model, self.batch_size = None, 100
        self.status = {"BERT_Type": ["BERTFamily"],
                        "hasData\t": False,
                        "hasModel": False,
                        "isTrained": False,
                        "accumulateEpoch": 0}
        self.down_stream_task_domain = {"Downstream task\t": "BF_Family class", 
                                    "Sequence Classification": "BFClassification",
                                    "Question Answering": "BFQA",
                                    "Token Classification": "BFTokenClassification"}


    def show_model_architecture(self) -> None:
        if self.status["hasModel"] is None: print("No model in the BERTFamily object."); return
        for name, module in self.model.named_children():
            if name == "bert":
                for n, _ in module.named_children():
                    print(f"{name}:{n}")
            else:
                print("{:15} {}".format(name, module))


    def show_model_configuration(self) -> None:
        assert self.status["hasModel"], "No model in the BERTFamily object."
        print(self.model.config)


    def show_status(self) -> None:
        print("\n".join("{}\t{}".format(k, v) for k, v in self.status.items()))  

    
    def show_all_task_in_BERTFamily(self) -> None:
        print("\n".join("{}\t{}".format(k, v) for k, v in self.down_stream_task_domain.items()))  


    def load_model(self, model = None) -> None:
        self.status["hasModel"] = True
        pass


    def forecasting(self) -> None:
        #For subclass inherit
        pass


    def save_model(self) -> None:
        pass


    def load_dataset_dict(self, data:dataset_dict.DatasetDict, downStreamTask = "Sequence Classification") -> None:
        assert downStreamTask in list(self.down_stream_task_domain.keys())[1:], "This version does not implement " + downStreamTask + " task."
        assert type(data).__name__ == "DatasetDict", "Only accept dataset_dict.DatasetDict class."

        pass


class BFClassification(BERTFamily):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.label_length = None
        self.status["BERT_Type"].append("BFClassification")


    class ClassificationDataset(Dataset):
        def __init__(self, raw_data, tokenizer, raw_target = None, max_length = 100) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.raw_data, self.raw_target = pd.DataFrame(raw_data), raw_target
            self.max_length = max_length
            assert self.raw_data.shape[1] <= 2, "Only accept one or two sequences as the input argument."
            if raw_target is not None:
                self.raw_target_dict = {}
                for i, ele in enumerate(pd.unique(raw_target)):
                    self.raw_target_dict[ele] = i


        def __len__(self):
            return self.raw_data.shape[0]


        def __getitem__(self, idx):
            if self.raw_data.shape[1] == 2:
                result = self.tokenizer.encode_plus(self.raw_data.iloc[idx, 0], self.raw_data.iloc[idx, 1], padding="max_length", max_length=self.max_length, truncation = True, return_tensors = 'pt')
            else:
                result = self.tokenizer.encode_plus(self.raw_data.iloc[idx, 0], padding="max_length", max_length=self.max_length, truncation = True, return_tensors = 'pt')
            return result, torch.tensor(self.raw_target_dict[self.raw_target[idx]]) if self.raw_target is not None else result


    def set_dataset(self, raw_data: pd.DataFrame, raw_target: list, tokenizer = None, data_type = "train", batch_size = 100, **kwargs):
        """ 
        Input:
        raw_data: n by p, n: observations (total sequence). p: number of sequences in each case.
        raw_target: a list of n-length.
        type: train, dev, test, other
        **kwargs: The argument in DataLoader

        Return 3 object:
        dataset, dataloader, dataloader with iter
        """ 
        if tokenizer is None: tokenizer = self.tokenizer
        tmp_dataset = ClassificationDataset(raw_data = raw_data, raw_target = raw_target, tokenizer = self.tokenizer, max_length = self.max_length)
        if data_type not in ["train", "test", "dev"]: return DataLoader(tmp_dataset, batch_size=batch_size, **kwargs)
        exec("self." + data_type + "_data_loader" + " = DataLoader(tmp_dataset, batch_size=batch_size, **kwargs)")

        self.batch_size = batch_size
        self.status["hasData\t"] = True
        self.label_length = len(tmp_dataset.raw_target_dict)
        

    def create_model(self, label_length:int, pretrained_model = None, **kwargs):
        assert (self.label_length is not None) & (self.label_length == label_length), "Mismatch on the length of labels."
        self.status["hasModel"] = True
        if not pretrained_model: pretrained_model = self.pretrained_model
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels = label_length, **kwargs).to(self.device)   
        #這裡再新增多一點東西 roberta
        return self.model
    

    def forecasting(self, data, model, tokenizer, batch_size = 100, **kwargs):
        tmp_dataset = ClassificationDataset(raw_data = data, tokenizer = tokenizer, max_length = self.max_length)
        data_loader = DataLoader(tmp_dataset, batch_size=batch_size, **kwargs)
        predictions = None
        with torch.no_grad():
            for df in data_loader if eval else tqdm(data_loader):                
                input = [t for t in df if t is not None]        
                outputs = model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device))
                _, pred = torch.max(outputs[0].data, 1)
                predictions = pred if predictions is None else torch.cat((predictions, pred))
        return predictions


    def training(self, train_data_loader, dev_data_loader = None, epochs = 50, optimizer = None, eval = False):
        #這裡要補checkpoint
        assert self.status["hasModel"], "No model in the BERTFamily object."
        if not optimizer: optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.status["isTrained"] = True
        outputs = None

        self.model.train()
        print("Start training ...")
        for epoch in range(epochs):
            total, correct, running_loss = 0, 0, 0
            for df in tqdm(train_data_loader):
                optimizer.zero_grad()
                input = [t for t in df if t is not None]
                outputs = self.model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device), 
                                labels = input[1].to(self.device))
                outputs[0].backward()
                optimizer.step()
                running_loss += outputs[0].item()
                _, pred = torch.max(outputs[1], -1)
                
                #
                
                total += input[1].size(0)
                correct += (pred ==input[1]).sum().item()

            if eval & (dev_data_loader is not None):
                self.model.eval()
                _, evalAcc, evalLoss = self.testing(self.model, dev_data_loader, eval = True)
                self.model.train()
                print('[epoch %d] loss: %.3f, TrainACC: %.3f, EvalACC: %.3f, EvalLoss: %.3f' %(epoch + 1, running_loss, correct/total, evalAcc, evalLoss))
            else:
                print('[epoch %d] loss: %.3f, ACC: %.3f' %(epoch + 1, running_loss, correct/total))
            self.status["accumulateEpoch"] += 1


    def testing(self, model, data_loader, eval = False, **kwargs):
        predictions = None
        total, correct, loss= 0, 0, 0
        with torch.no_grad():
            for df in data_loader if eval else tqdm(data_loader):                
                input = [t for t in df if t is not None]        
                outputs = model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device))
                logits = outputs[0]
                _, pred = torch.max(logits.data, -1)
                labels = input[1].to(self.device)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                predictions = pred if predictions is None else torch.cat((predictions, pred))
        acc = correct / total
        return predictions, acc, loss

       
class BFQA(BERTFamily):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.status["BERT_Type"].append("BFQA")


    class QADataset(Dataset):
        def __init__(self, data_type, questions, tokenized_questions, tokenized_paragraphs, max_question_len = 40, max_paragraph_len = 150, stride = 80):
            self.data_type = data_type
            self.questions = questions
            self.tokenized_questions = tokenized_questions
            self.tokenized_paragraphs = tokenized_paragraphs
            self.max_question_len = max_question_len
            self.max_paragraph_len = max_paragraph_len
            self.stride = stride
            # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
            self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1


        def __len__(self):
            return len(self.questions)


        def __getitem__(self, idx):
            question = self.questions[idx]
            tokenized_question = self.tokenized_questions[idx]
            tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]
            if self.data_type in ["test", "dev"]:
                input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
                #Split paragraph into several windows
                for i in range(0, len(tokenized_paragraph), self.stride):
                    # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                    input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                    input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                    
                    # Pad sequence and obtain inputs to model
                    input_ids, token_type_ids, attention_mask = padding(input_ids_question, input_ids_paragraph, max_seq_len = self.max_seq_len)
                    input_ids_list.append(input_ids)
                    token_type_ids_list.append(token_type_ids)
                    attention_mask_list.append(attention_mask)
                return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)
            else:
                # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
                ans_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
                ans_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

                mid = (ans_start_token + ans_end_token) // 2 #create a window which contain the answer
                paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
                paragraph_end = paragraph_start + self.max_paragraph_len
        
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] #(101: CLS, 102: SEP)
                input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
                
                # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
                ans_start_token += len(input_ids_question) - paragraph_start
                ans_end_token += len(input_ids_question) - paragraph_start

                input_ids, token_type_ids, attention_mask = padding(input_ids_question, input_ids_paragraph, max_seq_len = self.max_seq_len)
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), ans_start_token, ans_end_token

        
    def create_model(self, pretrained_model = None, **kwargs):
        self.status["hasModel"] = True
        if not pretrained_model: pretrained_model = self.pretrained_model
        self.model = BertForQuestionAnswering.from_pretrained(pretrained_model, **kwargs).to(self.device)   
        #這裡再新增多一點東西 roberta
        return self.model
    
    
    def set_dataset(self, questions_dict: dict, paragraphs_list: list, tokenizer = None, data_type = "train", batch_size = 100, **kwargs):
        assert self.__test_data_valid(questions_dict, paragraphs_list)
        if tokenizer is None: tokenizer = self.tokenizer
        if data_type in ["test", "dev"]: batch_size = 1

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        questions_tokenized = tokenizer([q["question_text"] for q in questions_dict], add_special_tokens=False)
        paragraphs_tokenized = tokenizer(paragraphs_list, add_special_tokens=False)
        tmp_dataset = QADataset(data_type, questions_dict, questions_tokenized, paragraphs_tokenized)

        if data_type not in ["train", "test", "dev"]: return DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, **kwargs)
        exec("self." + data_type + "DataLoader" + " = DataLoader(tmp_dataset, batch_size=batch_size, **kwargs)")
        self.batch_size = batch_size
        self.status["hasData\t"] = True


    def evaluate(self, data, output, tokenizer = None):
        answer = ''
        max_prob = float('-inf')
        num_of_windows = data[0].shape[1]
        if tokenizer is None: tokenizer = self.tokenizer
        
        for k in range(num_of_windows):
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)
            prob = start_prob + end_prob
            if prob > max_prob:
                max_prob = prob
                answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
        return answer.replace(' ','')


    def translate(self, model, data_loader):
        print("Evaluating Test Set ...")
        result = []
        model.eval()
        with torch.no_grad():
            for data in tqdm(data_loader):
                output = model(input_ids=data[0].squeeze(dim=0).to(self.device), token_type_ids=data[1].squeeze(dim=0).to(self.device),
                            attention_mask=data[2].squeeze(dim=0).to(self.device))
                result.append(self.evaluate(data, output))
        return result


    def training(self, train_data_loader, dev_data_loader = None, epochs = 50, optimizer = None, eval = False, logging_step = 100):
        #這裡要補checkpoint
        assert self.status["hasModel"], "No model in the BERTFamily object."
        logging_step = logging_step
        if not optimizer: optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.status["isTrained"] = True
        output = None

        self.model.train()
        print("Start training ...")
        for epoch in range(epochs):
            running_loss = train_acc = 0
            for data in tqdm(train_data_loader):	
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


    def __test_data_valid(self, questions_dict: dict, paragraphs_list: list, detect_num = 10) -> bool:
        keys_domain = ['paragraph_id', 'question_text', 'answer_text', 'answer_start', 'answer_end']
        n = len(questions_dict) - 1
        if not (questions_dict[n] == (len(paragraphs_list) - 1)): return False, "Mismatch length between questions_dict and paragraphs_list."
        
        detect_list = torch.cat((torch.round(torch.rand(detect_num) * n), torch.tensor([0]), torch.tensor([n])))
        results = torch.tensor([keys in questions_dict[int(i)].keys() for i in detect_list for keys in keys_domain])
        print(results)
        return torch.all(results), "Mismatch keys between questions_dict and keys_domain."


class BFTokenClassification(BERTFamily):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.label_names = None
        self.status["BERT_Type"].append("BFTokenClassification")
    

    class TokenDataset(Dataset):
        def __init__(self, data, tokenizer, max_length = 50) -> None:
            super().__init__()
            self.data = data
            self.max_length = max_length
            self.tokenizer = tokenizer
            print(123)
        

        def __len__(self):
            return len(self.data)


        def __getitem__(self, index):
            self.data[index]
            token = self.tokenizer.encode_plus(self.data[index]["tokens"], truncation=True, max_length=self.max_length, is_split_into_words=True, padding="max_length", add_special_tokens=False, return_tensors="pt")

            #initial label
            label = self.data[index]["ner_tags"]
            label_ids = []
            prev_word = None
            for word_ids in token.word_ids():  
                if word_ids is None:
                    label_ids.append(-100)
                elif word_ids != prev_word:  
                    label_ids.append(label[word_ids])
                else:
                    label_ids.append(-100)
                prev_word = word_ids

            return token, torch.tensor(label_ids)


    def set_dataset(self, data: dataset_dict.DatasetDict, tokenizer = None, max_length = 50, batch_size = 100, data_type = "train", **kwargs):
        """ 
        Input example: data: wnut["train"]

        """
        if tokenizer is None: tokenizer = self.tokenizer
        tmp_dataset = TokenDataset(data, tokenizer = self.tokenizer, max_length = max_length)
        if data_type not in ["train", "test", "dev"]: return DataLoader(tmp_dataset, batch_size=batch_size, **kwargs)
        exec("self." + data_type + "DataLoader" + " = DataLoader(tmp_dataset, batch_size=batch_size, **kwargs)")

        self.batch_size = batch_size
        self.label_names = data.features["ner_tags"].feature.names
        self.status["hasData\t"] = True

    def create_model(self, label_names = None, pretrained_model = None, **kwargs):
        if label_names is not None: self.label_names = label_names
        self.status["hasModel"] = True
        if not pretrained_model: pretrained_model = self.pretrained_model
        id2label = {i: label for i, label in enumerate(self.label_names)}
        label2id = {v: k for k, v in id2label.items()}
        self.model = BertForTokenClassification.from_pretrained(pretrained_model, id2label = id2label, label2id = label2id, **kwargs).to(self.device)
        return self.model
    
    def forecasting(self, data, model, tokenizer, batch_size = 100, **kwargs):
        pass



 
         
def padding(seq1_ids, seq2_ids, max_seq_len):
    paddingLen = max_seq_len - len(seq1_ids) - len(seq2_ids)
    input_ids = seq1_ids + seq2_ids + [0] * paddingLen
    token_type_ids = [0] * len(seq1_ids) + [1] * len(seq2_ids) + [0] * paddingLen
    attention_mask = [1] * (len(seq1_ids) + len(seq2_ids)) + [0] * paddingLen
    return input_ids, token_type_ids, attention_mask

def auto_build_model(data = None, target = None, pretrained_model = None, **kwargs):
    if not pretrained_model:
        #prepare pretrained_model
        pass
    pass

def read_json_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data

def get_learnable_parameters_size(model):
    tmp = [p for p in model.parameters() if p.requires_grad]
    return sum(list(map(lambda x: len(x.view(-1)), tmp)))

def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)

#build customize model                 
         
#dataset / dataloader
#




########## Train and Pred and Eval... ##########
#build




##############################test######################################

""" other
##########in ubutn
from zipfile import ZipFile
dataDir = "/home/ubuntu/work/BERTFamily/data/news/news.zip"#in ubutn
tmp = ZipFile("/BERTFamily/data/news/news.zip")
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
c = BFClassification(pretrained_model = "bert-base-chinese", max_length = 70)
c.set_dataset(raw_data = x, raw_target = y, batch_size=200, shuffle=True)
c.create_model(label_length=c.label_length)
c.show_model_architecture(); c.show_status()
a = c.training(1)


os.system("echo %PYTORCH_CUDA_ALLOC_CONF%")

import GPUtil
from GPUtil import showUtilization as gpu_usage
import gc
kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
gc.collect()

torch.cuda.empty_cache()
gpu_usage()
a[1].shape
a, b = c.testing(model = c.model, testingData = testx, testingTarget = testy);b


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
#dataDir = "/home/ubuntu/work/BERTFamily/data/news/news.zip"#in ubutn
tmp = ZipFile("BERTFamily/data/QA_data.zip")
from io import StringIO
from zipfile import Path
zipped = Path(tmp, at="hw7_train.json")
#df_train = pd.read_csv(StringIO(zipped.read_text()))
tmp = "BERTFamily/data/QA_data/hw7_train.json"

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


b = BFClassification(pretrained_model = "bert-base-uncased", max_length = 50)
b.set_dataset(df, target, data_type = "train", batch_size=100, shuffle=True)
b.set_dataset(dfdev, targetdev, data_type = "dev", batch_size=100, shuffle=True)
b.create_model(b.label_length)
b.show_model_architecture(); b.show_status()
a = b.training(train_data_loader=b.train_data_loader, dev_data_loader=b.dev_data_loader, epochs = 1, eval=True)
f = b.forecasting(data = dftest, model = b.model, tokenizer = "bert-base-uncased", batch_size = 100)
"""

""" idk

#evaluation
dataset = load_dataset('glue', 'mrpc', split='test')
dataset["sentence1"]
df = pd.DataFrame([dataset["sentence1"], dataset["sentence2"]])
df = df.transpose()
target = dataset["label"]
pred, acc = b.testing(b.model, df, target); acc
 """

#MRPC 0.7386
from datasets import load_dataset
dataset = load_dataset('glue', 'mrpc', split='train')
dataset["sentence1"]
df = pd.DataFrame([dataset["sentence1"], dataset["sentence2"]])
df = df.transpose()
target = dataset["label"]
b = BFClassification(pretrained_model = "bert-base-uncased", max_length = 50)
b.set_dataset(df, target, batch_size=100, shuffle=True)
b.create_model(b.label_length)
b.show_model_architecture(); b.show_status()
a = b.training(5)

#evaluation
dataset = load_dataset('glue', 'mrpc', split='test')
dataset["sentence1"]
df = pd.DataFrame([dataset["sentence1"], dataset["sentence2"]])
df = df.transpose()
target = dataset["label"]
pred, acc = b.testing(b.model, df, target); acc


"""  #CoLA -> OK, 0.83, epoch=10 (bert-base-uncased)
#tmp = "data/glue_data/coLA/train.tsv"
from zipfile import ZipFile, Path
from io import StringIO
dataDir = "/home/ubuntu/work/BERTFamily/data/CoLA.zip"
zipped = Path(dataDir, at="CoLA/train.tsv")
d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
target = d.iloc[:,1]
df = d.iloc[:, 3]
b = BFClassification(pretrained_model = "bert-base-uncased", max_length = 50)
b.set_dataset(df, target, batch_size=100, shuffle=True)
b.create_model(b.label_length)
b.show_model_architecture(); b.show_status()
a = b.training(10)

#evaluation
zipped = Path(dataDir, at="CoLA/dev.tsv")
d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
target = d.iloc[:,1]
df = d.iloc[:, 3]
pred, acc = b.testing(b.model, df, target)
63

#testing
zipped = Path(dataDir, at="CoLA/test.tsv")
d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
target = d.iloc[:,1]
df = d.iloc[:, 3]
b.testing(b.model, df, target) 
"""

""" #news -> OK
##########in mac
#dataDir = "~/Downloads/news.csv"
from zipfile import ZipFile
dataDir = "BERTFamily/data/news/news.zip"#in ubutn
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
b = BFClassification(pretrained_model = "bert-base-chinese", max_length = 70)
b.set_dataset(raw_data = x, raw_target = y, batch_size=100, shuffle=True)
b.create_model(label_length=b.label_length)
b.show_model_architecture(); b.show_status()
a = b.training(1)
"""
""" 
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]
 """


""" # QA
from zipfile import ZipFile, Path
from io import StringIO
import io
#dataDir = "BERTFamily/data/QA_data.zip"
dataDir = "/home/ubuntu/work/BERTFamily/data/QA_data.zip"

def read_data(dataDir, name):
    with ZipFile(dataDir, "r") as z:
        data = z.read("QA_data/" + name)
        data = json.load(io.BytesIO(data))
    return data["questions"], data["paragraphs"]


#d = pd.read_csv(StringIO(zipped.read_text()), sep="\t")
#tmp = "BERTFamily/data/QA_data/"

train_questions, train_paragraphs = read_data(dataDir, "hw7_train.json")


b = BFQA(pretrained_model = "bert-base-chinese")
b.set_dataset(train_questions, train_paragraphs, data_type = "train", batch_size=50, shuffle=True, pin_memory=True)
b.create_model()
b.show_model_architecture(); b.show_status()
b.training(train_data_loader = b.train_data_loader, epochs = 3)
# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair

 """


""" #test
class BFTokenClassification(BERTFamily):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
 

from transformers import AutoTokenizer
from datasets import load_dataset
data = load_dataset("conll2003")
a = data["train"].features['tokens']
a.feature
data["train"][3]



data["train"][0]




for i, u in enumerate(data["train"][0]["ner_tags"]):
    print(i, u)
task = "ner"
tokenize_and_align_labels(data["train"][0])
tmp = tokenizer(data["train"][0]["tokens"], truncation=True, is_split_into_words=True)
tmp.word_ids()

tokenized_wnut["train"][0]



tmp = tokenizer(data["train"][0]["tokens"], truncation=True, is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tmp["input_ids"])


tmp
example = data["train"][0]
example[f"{task}_tags"]
tmp.word_ids(batch_index=0)
##############################test######################################

def haha(a):
    return a+1
tokenized_data = data.map(tokenize_and_align_labels, batched=True)


from datasets import load_dataset

wnut = load_dataset("wnut_17")


wnut
type(wnut) == "datasets.dataset_dict.DatasetDict"
wnut.__annotations__
type(wnut).__name__ == "DatasetDict"


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
tokenized_wnut = data.map(tokenize_and_align_labels, batched=True)
tokenizer(data["train"][0]["tokens"], truncation=True, max_length=50, is_split_into_words=True)

tmp = tokenizer(data["train"][0]["tokens"], truncation=True, max_length=50, is_split_into_words=True, padding="max_length")
tmp.word_ids()





len(d4)

#nf = data["train"].features["ner_tags"]
#ln = nf.feature.names

id2label = {i: label for i, label in enumerate(ln)}
label2id = {v: k for k, v in id2label.items()}

from transformers import AutoModelForTokenClassification
m = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", id2label = id2label, label2id = label2id)


d = BFTokenClassification()
d.set_dataset(data = data, data_type = "train")
m = BertForTokenClassification.from_pretrained("roberta-base", id2label = id2label, label2id = label2id)
m.config.num_labels
d4[0]
d4[0].shape
m.train()
output = m(input_ids = d4[0].squeeze(1), token_type_ids = d4[1].squeeze(1), attention_mask = d4[2].squeeze(1), labels = d4[3])
len(output)
output[1].shape
test = torch.empty(2)
test[0] = 
tmp = torch.max(output[1], -1)
tmp[1].shape
m

 """


#test hugging face
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
if "BFTokenClassification" in self.status["BERT_Type"]:
                    pred = pred.view(-1)
                    label = input[1].view(-1).to(self.device)
                    compare_ids = torch.where(label!=-100)
                    correct = sum(pred[compare_ids] == label[compare_ids])
                    total = len(compare_ids[0])
                else:


# token classification
#start experiment
wnut = load_dataset("wnut_17")
d = BFTokenClassification()
d.set_dataset(data = wnut["train"], data_type = "train", batch_size = 100)
d.create_model()
d.show_model_architecture(); d.show_status()
d.training(train_data_loader = d.train_data_loader, epochs=10, eval=False)



dict = 'something awful'  # Bad Idea... pylint: disable=redefined-builtin

