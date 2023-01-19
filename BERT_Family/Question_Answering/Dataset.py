from torch.utils.data import Dataset
from ..data_process import padding
import torch

class QADataset(Dataset):
    def __init__(self, 
        data_type:str, 
        questions, 
        tokenized_questions, 
        tokenized_paragraphs, 
        max_question_len: int=40, 
        max_paragraph_len: int=150, 
        stride: int=80
        ):

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
