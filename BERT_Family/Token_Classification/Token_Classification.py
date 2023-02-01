from ..BERT_Family import BERTFamily
from .Dataset import TokenDataset
from torch.utils.data import DataLoader
from datasets import dataset_dict
from transformers import BertForTokenClassification

class BFTokenClassification(BERTFamily):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.label_names = None
        self.status["BERT_Type"].append("BFTokenClassification")
    



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
