from torch.utils.data import Dataset
import torch

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

