import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(examples, tokenizer=tokenizer):

    tokenized_input_text = tokenizer(examples["input_texts"], return_offsets_mapping=True)

    start_positions = []
    end_positions = []
    offsets = tokenized_input_text["offset_mapping"]
    for idx, offset in enumerate(offsets):
        # idx: the idx th sample in batch
        s = examples["slot_values"][idx][0]
        e = examples["slot_values"][idx][1]
        for i, pos in enumerate(offset):
            if pos[0] == s:
                start = i
            if pos[1] == e:
                end = i
        if(start is None or end is None):
            print(tokenized_input_text)
        start_positions.append(start)
        end_positions.append(end)

    tokenized_slot_desc = tokenizer(examples["slot_descs"])

    cat_text = [ intext + desctext for intext, desctext in zip(examples["input_texts"] ,examples["slot_descs"]) ]
    tokenized_cat_desc = tokenizer(cat_text)

    res = {"input_text_ids": tokenized_input_text["input_ids"],
           "input_text_attention_mask": tokenized_input_text["attention_mask"],
           "slot_desc_ids": tokenized_slot_desc["input_ids"],
           "slot_desc_attention_mask": tokenized_slot_desc["attention_mask"],
           "cat_desc_ids": tokenized_cat_desc["input_ids"],
           "cat_desc_attention_mask": tokenized_cat_desc["attention_mask"],
           "start_positions": start_positions,
           "end_positions": end_positions}

    return res


def my_collate(data, tokenizer=tokenizer):
    intext = [d['input_texts'] for d in data]
    slodesc = [d['slot_descs'] for d in data]
    action_type = [d['action_types'] for d in data]

    input_text_ids = [d["input_text_ids"] for d in data]
    input_text_attention_mask = [d["input_text_attention_mask"] for d in data]
    input_text = {"input_ids": input_text_ids, "attention_mask": input_text_attention_mask}
    input_text = tokenizer.pad(input_text)

    slot_desc_ids = [d["slot_desc_ids"] for d in data]
    slot_desc_attention_mask = [d["slot_desc_attention_mask"] for d in data]
    slot_desc = {"input_ids": slot_desc_ids, "attention_mask": slot_desc_attention_mask}
    slot_desc = tokenizer.pad(slot_desc)

    slot_type = [d["slot_types"] for d in data]
    slot_value = [d["slot_values"] for d in data]
    start_positions = [d["start_positions"] for d in data]
    end_positions = [d["end_positions"] for d in data]

    cat_ids = [ d['cat_desc_ids'] for d in data ]
    cat_attention_mask = [ d['cat_desc_attention_mask'] for d in data ]
    cattext = {"input_ids": cat_ids, "attention_mask": cat_attention_mask}
    cattext = tokenizer.pad(cattext)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = {"input_text_ids": torch.tensor(input_text["input_ids"]).to(device),
           "input_text_attention_mask": torch.tensor(input_text["attention_mask"]).to(device),
           "slot_desc_ids": torch.tensor(slot_desc["input_ids"]).to(device),
           "slot_desc_attention_mask": torch.tensor(slot_desc["attention_mask"]).to(device),
           "slot_type": torch.tensor(slot_type).to(device),
           "slot_value": torch.tensor(slot_value).to(device),
           "input_text": intext,
           "slot_desc": slodesc,
           "action_type": action_type,
           "start_positions": torch.tensor(start_positions).to(device),
           "end_positions": torch.tensor(end_positions).to(device),
           "cat_ids": torch.tensor(cattext["input_ids"]).to(device),
           "cat_attention_mask": torch.tensor(cattext["attention_mask"]).to(device)
           }
    return res


def createDataLoader(datas, batch_size, collate_fn=my_collate):
    d = datas.toDict()
    dataset = Dataset.from_dict(d)

    #tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=batch_size,
    #                                remove_columns=["dialogue_ids", "input_texts", "slot_descs",
    #                                               "slot_desc_keys"])
    tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)

    #print(tokenized_dataset['slot_values'])

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader

