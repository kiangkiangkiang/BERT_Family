import torch
import torch.nn as nn
from datetime import datetime
import os, warnings

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
