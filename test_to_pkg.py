import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from transformers import BertTokenizerFast
import pandas as pd
from transformers import BertForSequenceClassification, BertForQuestionAnswering, BertForTokenClassification
from tqdm.auto import tqdm
import json
from datasets import load_dataset, dataset_dict
from typing import List, Optional, Iterable, Any, Union, Dict
import numpy as np
########## Modules ##########
#One or two sentence with one dim label

class BERTFamily(nn.Module):
    def __init__(
        self, 
        pretrained_model: str='bert-base-uncased', 
        max_length: int=100,
        device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ):

        super().__init__()
        print("Using device: ", device)
        self.device = device
        self.pretrained_model = pretrained_model
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        self.max_length = max_length
        self.train_data_loader, self.validation_data_loader, self.test_data_loader =  None, None, None
        self.model, self.batch_size = None, 100
        self.status = {"BERT_Type": ["BERTFamily"],
                        "has_train_data": False,
                        "has_validation_data": False,
                        "has_test_data": False,
                        "hasModel": False,
                        "isTrained": False,
                        "accumulateEpoch": 0}
        self.down_stream_task_domain = {"Downstream task\t": "BF_Family class", 
                                    "Sequence Classification": "BFClassification",
                                    "Question Answering": "BFQA",
                                    "Token Classification": "BFTokenClassification"}


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


    def load_model(self, path: str) -> None:
        self.status["hasModel"] = True
        pass


    def forecasting(self) -> None:
        #For subclass inherit
        pass


    def save_model(self) -> None:
        pass



class BFClassification(BERTFamily):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_length = None
        self.status["BERT_Type"].append("BFClassification")


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
                self.raw_target_dict = {}
                for i, ele in enumerate(pd.unique(raw_target)):
                    self.raw_target_dict[ele] = i


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

            return result, torch.tensor(self.raw_target_dict[self.raw_target[idx]]) if self.raw_target is not None else result


    def set_dataset(
        self, 
        raw_data: pd.DataFrame, 
        raw_target: Optional[Iterable]=None, 
        tokenizer: Any=None,
        data_type: str="train", 
        batch_size: int=128, 
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

        if tokenizer is None: 
            tokenizer = self.tokenizer
        tmp_dataset = self.ClassificationDataset(raw_data=raw_data, raw_target=raw_target, 
                                                 tokenizer=tokenizer, max_length=self.max_length)

        if data_type in ["train", "test", "validation"]:     
            exec("self." + data_type + "_data_loader" + " = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, **kwargs)")
            self.batch_size = batch_size
            self.status["has_"+data_type+"_data"] = True
            self.label_length = len(tmp_dataset.raw_target_dict)
            return eval("self." + data_type + "_data_loader")

        return DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, **kwargs)


    def create_model(
        self, label_length:int, pretrained_model: Optional[str]=None, **kwargs
        ) -> BertForSequenceClassification:

        assert (self.label_length is not None) & (self.label_length == label_length), "Mismatch on the length of labels."
        self.status["hasModel"] = True
        if not pretrained_model: 
            pretrained_model = self.pretrained_model
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=label_length, **kwargs) 
        return self.model
    
    #要改
    def inference(
        self, model: Any, data:pd.DataFrame, tokenizer: Any=None, batch_size: int=128, **kwargs
        ) -> list:

        #這裡要改 是否還必要轉乘dataloader?  感覺不用
        tmp_dataset = self.ClassificationDataset(raw_data=data, tokenizer=tokenizer, max_length=self.max_length)
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
        return list(predictions)


    def train(
        self, 
        train_data_loader: DataLoader, 
        validation_data_loader: Optional[DataLoader]=None, 
        epochs: int=5, 
        optimizer: Any=None, 
        lr_scheduler: Any=None,
        eval: bool=False
        ) -> None:

        assert self.status["hasModel"], "No model in the BERTFamily object."
        if not optimizer: 
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
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
            self.status["accumulateEpoch"] += 1


    def evaluation(
        self, 
        model:BertForSequenceClassification,
        eval_data_loader:DataLoader,
        eval: bool=False
        ) -> Dict[str, float]:

        model.eval()
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



def auto_build_model(
    dataset: Any=None, 
    dataset_x_features: Union[List[int], List[str]]=None, 
    dataset_y_features: Union[List[int], List[str]]=None, 
    x_dataframe: pd.DataFrame=None, 
    y: list=None, 
    pretrained_model: Optional[str]=None,
    max_length: int=100, 
    batch_size: int=128, 
    data_type: List[str]=["train"]
    ) -> BFClassification:

    if dataset:
        assert (dataset_x_features is not None) & (dataset_y_features is not None), "Missing x, y features name."
        x_dataframe, y, data_type = load_dataset_dict(data=dataset, x=dataset_x_features, y=dataset_y_features, data_type=data_type)

    pretrained_model = "bert-base-uncased" if not pretrained_model else pretrained_model
    result_model = BFClassification(pretrained_model=pretrained_model, max_length=max_length)

    if len(data_type) == 1:
        result_model.set_dataset(raw_data=x_dataframe, raw_target=y, batch_size=batch_size, data_type=data_type[0])
    else:
        for i in range(len(data_type)):
            result_model.set_dataset(raw_data=x_dataframe[i], raw_target=y[i], batch_size=batch_size, data_type=data_type[i])
            
    result_model.create_model(result_model.label_length)
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


import gc


#test
dataset = load_dataset('glue', 'mrpc')
dataset
mymodel = auto_build_model(dataset=dataset, 
                           dataset_x_features=['sentence1', 'sentence2'],
                           dataset_y_features=["label"],
                           batch_size=100)
gc.collect()
mymodel.show_status()

mymodel.train(train_data_loader=mymodel.train_data_loader, 
              validation_data_loader=mymodel.validation_data_loader, epochs=1,
              eval=True)

mymodel.evaluation(model=mymodel.model, data_loader=mymodel.test_data_loader, eval=False)