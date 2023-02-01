from typing import List, Any, Union
import pandas as pd
from .Text_Classification.Classifier import *

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
