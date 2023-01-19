import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
import pandas as pd
from transformers import AutoModelForSequenceClassification, BertForQuestionAnswering, BertForTokenClassification
from tqdm.auto import tqdm
import json
from datasets import load_dataset, dataset_dict
from typing import List, Optional, Iterable, Any, Union, Dict
import numpy as np
from datetime import datetime
import os
import warnings
########## Modules ##########
#One or two sentence with one dim label

class BERTFamily(nn.Module):
    def __init__(
        self, 
        pretrained_model: str='bert-base-uncased', 
        tokenizer: str='bert-base-uncased',
        max_length: int=100,
        device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ):

        super().__init__()
        print("Using device: ", device)
        self.device = device
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.train_data_loader, self.validation_data_loader, self.test_data_loader =  None, None, None
        self.model, self.batch_size = None, 100
        self.status = {"BERT_Type": ["BERTFamily"],
                        "pretrained_model": self.pretrained_model,
                        "tokenizer": self.tokenizer,
                        "has_train_data": False,
                        "has_validation_data": False,
                        "has_test_data": False,
                        "hasModel": False,
                        "isTrained": False,
                        "train_acc": 0.0,
                        "train_loss": 0.0,
                        "test_acc": 0.0,
                        "test_loss": 0.0,
                        "accumulateEpoch": 0
                       }
        self.down_stream_task_domain = {"Downstream task\t": "BF_Family class", 
                                        "Sequence Classification": "BFClassification",
                                        "Question Answering": "BFQA",
                                        "Token Classification": "BFTokenClassification"}
        self.model_name = "model_" + datetime.now().strftime("%Y%m%d%H%M%S")


    def show_model_architecture(self) -> None:
        assert self.status["hasModel"], "No model in the BERTFamily object."
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

    
    def save_model(self, path:str) -> None:
        assert self.status["hasModel"] == True, "No model can be save. Please use load_model() to create a specific model."
        PATH = path if path is not None else os.getcwd() + "/model_history/"
        if not os.path.exists(PATH):
            warnings.warn('Cannot find the directory. Auto build model in'+PATH, RuntimeWarning)
            os.mkdir(PATH) 
        torch.save(self.model.state_dict(), PATH + self.model_name + ".pth")



class BFClassification(BERTFamily):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_length = None
        self.target_key2value = None
        self.target_value2key = None
        self.status["BERT_Type"].append("BFClassification")
        

    def set_dataset(
        self, 
        raw_data: pd.DataFrame, 
        raw_target: Optional[Iterable]=None, 
        data_type: str="train", 
        batch_size: int=128,
        model_name: str=None,
        **kwargs
        ) -> DataLoader:

        """ 
        Input:
        raw_data: n by p, n: observations (total sequence). p: number of sequences in each case.
        raw_target: a list of n-length.
        type: train, validation, test, other
        **kwargs: The argument in DataLoader

        Return 3 object:
        dataset, dataloader, dataloader with iter
        """ 

        print("Using ", self.tokenizer, " to tokenize...")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        
        tmp_dataset = ClassificationDataset(raw_data=raw_data, raw_target=raw_target, 
                                            tokenizer=tokenizer, max_length=self.max_length)

        if data_type in ["train", "test", "validation"]:     
            exec("self." + data_type + "_data_loader" + " = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, **kwargs)")
            self.batch_size = batch_size
            self.status["has_"+data_type+"_data"] = True
            self.target_key2value = tmp_dataset.target_key2value
            self.target_value2key = tmp_dataset.target_value2key
            self.label_length = len(tmp_dataset.target_key2value)
            if model_name is not None:
                self.model_name = model_name
            return eval("self." + data_type + "_data_loader")

        return DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, **kwargs)


    def load_model(self, model_path: Any=None, **kwargs):
        assert self.label_length is not None, "Mismatch on the length of labels."
        self.status["hasModel"] = True
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model, num_labels=self.label_length, **kwargs) 
        if model_path is not None:
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
        return self.model
    

    def inference(
        self, 
        model: Any, 
        data: pd.DataFrame, 
        batch_size: int=128, 
        **kwargs
        ) -> list:

        tmp_dataset = self.ClassificationDataset(raw_data=data, tokenizer=self.tokenizer, max_length=self.max_length)
        data_loader = DataLoader(tmp_dataset, batch_size=batch_size, **kwargs)
        predictions = []
        model.to(self.device)
        with torch.no_grad():
            for df in data_loader if eval else tqdm(data_loader):                
                input = [t for t in df if t is not None]        
                outputs = model(input_ids=input[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=input[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=input[0]["attention_mask"].squeeze(1).to(self.device))
                pred = torch.max(outputs[0].data, 1).indices.tolist()
                predictions += pred
        return [self.target_value2key[i] for i in predictions]


    def train(
        self, 
        train_data_loader: DataLoader, 
        validation_data_loader: Optional[DataLoader]=None, 
        epochs: int=5, 
        optimizer: Any=None, 
        lr_scheduler: Any=None,
        eval: bool=False,
        save_model: bool=True,
        save_path: str=None
        ) -> None:

        assert self.status["hasModel"], "No model in the BERTFamily object."
        if not optimizer: 
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if not lr_scheduler: 
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.5)
        self.status["isTrained"] = True
        
        print("Start training ...")
        self.model.to(self.device)
        self.model.train()
        for epoch in range(epochs):
            total, correct, running_loss = 0, 0, 0
            for df in tqdm(train_data_loader):
                optimizer.zero_grad()

                outputs = self.model(input_ids=df[0]["input_ids"].squeeze(1).to(self.device), 
                                    token_type_ids=df[0]["token_type_ids"].squeeze(1).to(self.device), 
                                    attention_mask=df[0]["attention_mask"].squeeze(1).to(self.device), 
                                    labels = df[1].to(self.device))

                outputs[0].backward()
                optimizer.step()
                running_loss += outputs[0].item()
                pred = torch.max(outputs[1], -1).indices.to("cpu")

                total += df[1].size(0)
                correct += (pred == df[1]).sum().item()

            lr_scheduler.step()
            if eval & (validation_data_loader is not None):
                eval_result = self.evaluation(self.model, validation_data_loader, eval=True)
                print('[epoch %d] TrainACC: %.3f, loss: %.3f, EvalACC: %.3f, EvalLoss: %.3f' %(epoch + 1, correct/total, running_loss, eval_result["Accurancy"], eval_result["Loss"]))
            else:
                print('[epoch %d] ACC: %.3f, loss: %.3f' %(epoch + 1, correct/total, running_loss))
            self.status["train_acc"] = correct/total
            self.status["train_loss"] = running_loss
            self.status["accumulateEpoch"] += 1
            
            if save_model:
                self.save_model(path = save_path)


    def evaluation(
        self, 
        model: Any,
        eval_data_loader:DataLoader,
        eval: bool=False
        ) -> Dict[str, float]:

        model.eval()
        model.to(self.device)
        total, correct, loss= 0, 0, 0
        with torch.no_grad():
            for df in eval_data_loader if eval else tqdm(eval_data_loader):                
                outputs = model(input_ids=df[0]["input_ids"].squeeze(1).to(self.device), 
                                token_type_ids=df[0]["token_type_ids"].squeeze(1).to(self.device), 
                                attention_mask=df[0]["attention_mask"].squeeze(1).to(self.device),
                                labels=df[1].to(self.device))
                loss += outputs[0].item()
                pred = torch.max(outputs[1].data, -1).indices.to("cpu")
                total += df[1].size(0)
                correct += (pred == df[1]).sum().item()
        acc = float(correct / total)
        model.train()
        return {"Accurancy": acc, "Loss": loss}



class ClassificationDataset(Dataset):
        def __init__(
            self, 
            raw_data: pd.DataFrame, 
            raw_target: Optional[Iterable]=None, 
            tokenizer: Any=None, 
            max_length: int=100
            ):

            super().__init__()
            if tokenizer:
                self.tokenizer = tokenizer
            self.raw_data, self.raw_target = raw_data, raw_target
            self.max_length = max_length
            assert self.raw_data.shape[1] <= 2, "Only accept one or two sequences as the input argument."
            if raw_target is not None:
                self.target_key2value = {}
                self.target_value2key = {}
                for i, ele in enumerate(pd.unique(raw_target)):
                    self.target_key2value[ele] = i
                    self.target_value2key[i] = ele


        def __len__(self):
            return self.raw_data.shape[0]


        def __getitem__(self, idx):
            if self.raw_data.shape[1] == 2:
                result = self.tokenizer.encode_plus(self.raw_data.iloc[idx, 0], 
                                                    self.raw_data.iloc[idx, 1], 
                                                    padding="max_length", 
                                                    max_length=self.max_length, 
                                                    truncation=True, 
                                                    return_tensors='pt')
            else:
                result = self.tokenizer.encode_plus(self.raw_data.iloc[idx, 0], 
                                                    padding="max_length", 
                                                    max_length=self.max_length, 
                                                    truncation=True, 
                                                    return_tensors='pt')

            return result, torch.tensor(self.target_key2value[self.raw_target[idx]]) if self.raw_target is not None else result



def auto_build_model(
    dataset: Any=None, 
    dataset_x_features: Union[List[int], List[str]]=None, 
    dataset_y_features: Union[List[int], List[str]]=None, 
    x_dataframe: pd.DataFrame=None, 
    y: list=None, 
    batch_size: int=128, 
    data_type: List[str]=["train"],
    **kwargs
    ) -> BFClassification:
    '''
    **kwargs: Parameters for BFClassification object. Mostly change tokenizer, pre-trained-model...
    '''

    if dataset is not None:
        assert (dataset_x_features is not None) & (dataset_y_features is not None), "Missing x, y features name."
        x_dataframe, y, data_type = load_dataset_dict(data=dataset, x=dataset_x_features, y=dataset_y_features, data_type=data_type)
    result_model = BFClassification(**kwargs)
    if len(data_type) == 1:
        result_model.set_dataset(raw_data=x_dataframe, raw_target=y, batch_size=batch_size, data_type=data_type[0])
    else:
        for i in range(len(data_type)):
            result_model.set_dataset(raw_data=x_dataframe[i], raw_target=y[i], batch_size=batch_size, data_type=data_type[i])
            
    result_model.load_model()
    result_model.show_model_architecture()
    result_model.show_status()
    return result_model


def load_dataset_dict(
    data: Any=None, 
    x: Union[List[int], List[str]]=None, 
    y: Union[List[int], List[str]]=None, 
    data_type: List[str]=["train"]
    ) -> tuple[pd.DataFrame, list, List[str]]:

    """ 
    Input: 
    x, y need to be a list containing either a number list that the features locate in dataframe or a string list with the features name.
    ex. x=[0, 1], which specific the location 1, 2 are the x-features in data respectively.
    """

    assert len(y) == 1, "The dimension of y is not 1 (multilvariable task is not applied now)."
    if type(data).__name__ == "DatasetDict":
        dataset_dict = sorted(data.values(), key=len, reverse=True)
        data_type = ["train", "test", "validation"]
        result_x, result_y, result_type = [], [], []
        for count, dataset in enumerate(dataset_dict):
            tmpx, tmpy = dataset2dataframe(data=dataset, x=x, y=y)
            result_x.append(tmpx); result_y.append(tmpy); result_type.append(data_type[count])
            if count == 2: 
                return result_x, result_y, result_type
    elif type(data).__name__ == "Dataset":
        tmpx, tmpy = dataset2dataframe(data=data, x=x, y=y)
        return tmpx, tmpy, data_type
    else:
        raise AttributeError(type(data).__name__ + " type data cannot be handled. Please input a 'Dataset' or 'DatasetDict' type data.") 


def dataset2dataframe(
    data: Any=None, 
    x: Union[List[int], List[str]]=None, 
    y: Union[List[int], List[str]]=None, 
    ) -> tuple[pd.DataFrame, list]:

    assert type(data).__name__ == "Dataset", "Only accept Dataset class."
    data_pd = pd.DataFrame(data)
    if isinstance(x[0], str):
        return data_pd[x], list(data_pd[y].iloc[:, 0])
    else:
        return data_pd.iloc[:, x], list(data_pd.iloc[:, y].iloc[:, 0])


def padding(seq1_ids, seq2_ids, max_seq_len):
    paddingLen = max_seq_len - len(seq1_ids) - len(seq2_ids)
    input_ids = seq1_ids + seq2_ids + [0] * paddingLen
    token_type_ids = [0] * len(seq1_ids) + [1] * len(seq2_ids) + [0] * paddingLen
    attention_mask = [1] * (len(seq1_ids) + len(seq2_ids)) + [0] * paddingLen
    return input_ids, token_type_ids, attention_mask


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


#test result: cola:0.827, mrpc:0.801



dataset = load_dataset('glue', "wnli")
mymodel = auto_build_model(dataset=dataset, 
                        dataset_x_features=["sentence1", "sentence2"],
                        dataset_y_features=["label"],
                        batch_size=64,
                        tokenizer="albert-base-v2",
                        pretrained_model="albert-base-v2")
mymodel.train(train_data_loader=mymodel.train_data_loader, 
            validation_data_loader=mymodel.validation_data_loader, 
            epochs=1,
            eval=True)

#del mymodel.model
#mymodel.load_model(model_path="/home/ubuntu/work/model_history/model_20230119031015.pth")



""" For All Dataset
import gc
#a = ['rte', 'wnli']
#b = [["sentence1", "sentence2"], ["sentence1", "sentence2"]]
dataset_name = ['cola', 'mrpc', "sst2", "qnli"] #mnli
x_name = [['sentence'], ["sentence1", "sentence2"], ["sentence"], ["question", "sentence"]]
test_epochs = 10
train_result = []
test_result = []
for x, each_dataset in zip(x_name, dataset_name):
    gc.collect()
    print("Start", each_dataset, "evaluation.")
    dataset = load_dataset('glue', each_dataset)
    mymodel = auto_build_model(dataset=dataset, 
                            dataset_x_features=x,
                            dataset_y_features=["label"],
                            batch_size=64,
                            tokenizer="albert-base-v2",
                            pretrained_model="albert-base-v2")
    mymodel.train(train_data_loader=mymodel.train_data_loader, 
                validation_data_loader=mymodel.validation_data_loader, 
                epochs=test_epochs,
                eval=True)
    train_result.append(mymodel.status["train_acc"])
    test_result.append(mymodel.evaluation(model=mymodel.model, eval_data_loader=mymodel.test_data_loader, eval=False))
    print("End of", each_dataset, ". Test Acc and Loss are", test_result[-1], ".")
    del mymodel
 """