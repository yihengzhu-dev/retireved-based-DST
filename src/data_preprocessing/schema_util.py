import json
from collections import defaultdict

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from .data_tokennizer import tokenizer
from src.RBDST.common import common

def create_slot_description_from_schema(schema_file_path):

    schema_json = json.load(open(schema_file_path, 'r', encoding='utf8'))

    slot2description = {}

    for service in schema_json:
        service_name = service['service_name']
        #service_description = f"{service_name}-{service['description']}"
        slots = [ slot for slot in service["slots"]]

        intents = service['intents']
        user_intent_act = ['INFORM_INTENT', 'NEGATE_INTENT', 'AFFIRM_INTENT','NEGATE', 'REQUEST_ALTS', 'THANK_YOU', 'GOODBYE']
        intents.append({"name": "NONE", "description": ""})
        for intent in intents:
            intent_name = intent['name']

            intent_description = {}
            intent_description["service_name"] = service_name
            intent_description['intent_name'] = intent_name
            intent_description['intent_description'] = intent['description']
            key = f"{service_name}-{intent_name}"
            intent_description_str = json.dumps(intent_description)
            slot2description[key] = intent_description_str

            for act in user_intent_act:
                key = f"{service_name}-{intent_name}-{act}"
                intent_description = {}
                intent_description["service_name"] = service_name
                intent_description['intent_name'] = intent_name
                intent_description['intent_description'] = intent['description']
                intent_description['act'] = act
                intent_description_str = json.dumps(intent_description)
                slot2description[key] = intent_description_str

            for slot in slots:
                user_act = ['INFORM', 'AFFIRM', 'SELECT']
                slot_name = slot['name']

                slot_description = {}
                slot_description["service_name"] = service_name
                slot_description['intent_name'] = intent_name
                slot_description['intent_description'] = intent['description']
                slot_description['act'] = 'REQUEST'
                slot_description['slot_name'] = slot_name
                slot_description['slot_description'] = slot['description']
                slot_description_str = json.dumps(slot_description)

                key = f"{service_name}-{intent_name}-{slot_name}-REQUEST"
                slot2description[key] = slot_description_str

                for act in user_act:
                        intent_name = intent['name']
                        slot_description = {}
                        slot_description["service_name"] = service_name
                        slot_description['intent_name'] = intent_name
                        slot_description['intent_description'] = intent['description']
                        slot_description['act'] = act
                        slot_description['slot_name'] = slot_name
                        slot_description['slot_description'] = slot['description']
                        if slot['is_categorical'] == False:
                            slot_description_str = json.dumps(slot_description)
                            key = f"{service_name}-{intent_name}-{slot_name}-{act}"
                            slot2description[key] = slot_description_str
                        else:
                            possible_values = slot['possible_values']
                            possible_values.append("dontcare")
                            for value in possible_values:
                                key = f"{service_name}-{intent_name}-{slot_name}-{value}-{act}"
                                slot_description_v = {}
                                slot_description_v["service_name"] = service_name
                                slot_description_v['intent_name'] = intent_name
                                slot_description_v['intent_description'] = intent['description']
                                slot_description_v['act'] = act
                                slot_description_v['slot_name'] = slot_name
                                slot_description_v['slot_value'] = value
                                slot_description_v['slot_description'] = slot['description']
                                slot_description_v_str = json.dumps(slot_description_v)
                                slot2description[key] = slot_description_v_str

    return slot2description


def text_tokenize(examples, tokenizer=tokenizer):
    return tokenizer(examples['text'])

def text_collate(batch, tokenizer=tokenizer):
    input_text = tokenizer.pad(batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "input_ids": torch.tensor(input_text["input_ids"]).to(device),
        "attention_mask": torch.tensor(input_text["attention_mask"]).to(device)
    }

def create_slot_desc_db(slot2description):

    slot_desc_db = []
    for key, desc in slot2description.items():

        desc_json = json.loads(desc)
        if 'act' in desc_json.keys():
            slot_desc_db.append(desc)

    return slot_desc_db

def update_slotDB_index(model):
    slot_desc_db = common.slotdescdb
    index = common.index
    index.reset()
    dataset = Dataset.from_dict({"text": slot_desc_db})
    batch_size = 32
    tokenized_dataset = dataset.map(text_tokenize, batched=True, batch_size=batch_size)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=text_collate, shuffle=False)
    #progress_bar = tqdm(len(dataloader))
    for idx, batch in enumerate(dataloader):
        emb = model.encode(batch)
        #emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        index.add(emb.cpu().numpy())
        #progress_bar.update(1)


def create_slot_categorical_dict(schema_file_path):
    slot_dict = defaultdict(lambda: defaultdict(bool))
    schema_json = json.load(open(schema_file_path, 'r', encoding='utf8'))
    for service in schema_json:
        service_name = service['service_name']
        slots = service['slots']
        for slot in slots:
            slot_name = slot['name']
            slot_dict[service_name][slot_name] = slot['is_categorical']

    # missing slot in schema
    slot_dict['Restaurants_1']['count'] = False
    return slot_dict
