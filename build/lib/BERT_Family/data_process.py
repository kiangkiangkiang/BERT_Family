import json

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
