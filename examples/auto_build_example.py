from datasets import load_dataset
from ..BERT_Family.auto_process import *

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