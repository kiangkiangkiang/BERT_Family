import pandas as pd
from typing import Optional, Iterable, Any, Dict
from .Dataset import ClassificationDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
from ..BERT_Family import BERTFamily

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
        dataset, dataloader, dataloader with iter.
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

        tmp_dataset = ClassificationDataset(raw_data=data, tokenizer=AutoTokenizer.from_pretrained(self.tokenizer), max_length=self.max_length)
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

