from ..BERT_Family import BERTFamily
from .Dataset import *
from transformers import BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm

class BFQA(BERTFamily):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.status["BERT_Type"].append("BFQA")


        
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
