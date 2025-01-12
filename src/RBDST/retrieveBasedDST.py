import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn, clamp
from transformers import DistilBertModel, DistilBertConfig, PreTrainedModel

from src.RBDST.common import common
from src.data_preprocessing.data_preprocessing import create_slot_sample_dataloader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json

from transformers.modeling_outputs import BaseModelOutput

class RetrieveBasedDSTConfig(DistilBertConfig):
    model_type = "retrieveBasedDST"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RetrieveBasedDSTModelOutput(BaseModelOutput):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class RetrieveBasedDST(PreTrainedModel):
    config_class = RetrieveBasedDSTConfig

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.retrieveHeader = nn.Linear(config.dim, 256)
        self.slotHeader = nn.Linear(config.dim, 2)
        self.slotValueHeader = nn.Linear(config.dim, 2)
        self.retrieveLossFn = nn.CrossEntropyLoss()
        self.slotLossFn = nn.CrossEntropyLoss(reduction='sum')
        self.valueLossFn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)

        self.config = config

    def forward(self, batch):
        """

        1. calculate input text embeddings and slot desc ebeddings
        3. calculate slot value for non-categorical with inform, affirm, select action

        """

        input_text_ids = batch['input_text_ids']
        input_text_attention_mask = batch['input_text_attention_mask']
        slot_desc_ids = batch['slot_desc_ids']
        slot_desc_attention_mask = batch['slot_desc_attention_mask']
        slot_type = batch['slot_type']

        action_type = batch['action_type']
        input_text = batch['input_text']

        bs, seq_len = input_text_ids.shape

        x_outputs = self.distilbert(input_ids=input_text_ids, attention_mask=input_text_attention_mask)
        x_embeddings = x_outputs.last_hidden_state[:, 0, :]
        x_embeddings = self.dropout(x_embeddings)
        x_embeddings = self.retrieveHeader(x_embeddings)

        if slot_desc_ids is not None:
            y_outputs = self.distilbert(input_ids=slot_desc_ids, attention_mask=slot_desc_attention_mask)
            y_embeddings = y_outputs.last_hidden_state[:, 0, :]
            y_embeddings = self.dropout(y_embeddings)
            y_embeddings = self.retrieveHeader(y_embeddings)

            #similarity = cosine_similarity(x_embeddings.unsqueeze(1), y_embeddings.unsqueeze(0), dim=-1)
            similarity = torch.matmul(x_embeddings, y_embeddings.transpose(1, 0))

            retrieve_loss_fn = self.retrieveLossFn
            target = torch.tensor([i for i in range(bs)]).to(self.device)
            retrieve_loss = retrieve_loss_fn(similarity, target)

            x = x_embeddings.clone().cpu().detach().numpy()
            slotsamples = create_slot_sample_dataloader(x, input_text, self.config.k, mode="train")
            ss_loss = 0.0
            ss_n = 0
            for ss in slotsamples:

                label = ss['label']
                ssbs = label.shape[0]

                ss_output = self.distilbert(input_ids=ss['input_ids'], attention_mask=ss['input_attention_mask'])
                ss_embeddings = ss_output.last_hidden_state[:, 0, :]
                ss_embeddings = self.dropout(ss_embeddings)
                ss_logits = self.slotHeader(ss_embeddings)

                ss_loss_t = self.slotLossFn(ss_logits, label)
                ss_loss = ss_loss + ss_loss_t
                ss_n = ss_n + ssbs
            ss_loss = ss_loss / ss_n

            value_loss = 0.0

            cat_ids = batch['cat_ids']
            cat_attention_mask = batch['cat_attention_mask']
            starts = batch['start_positions']
            ends = batch['end_positions']
            #cat_ids = torch.tensor(
                #[ cat_id.tolist() for cat_id, act_ty, slot_ty in zip(cat_ids, action_type, slot_type) if act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1],
                #device=self.device)
            #cat_attention_mask = torch.tensor(
                #[cat_attn.tolist() for cat_attn, act_ty, slot_ty in zip(cat_attention_mask, action_type, slot_type) if act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1],
                #device=self.device)
            #starts = torch.tensor(
                #[s.tolist() for s, act_ty, slot_ty in zip(starts, action_type, slot_type) if act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1],
                #device=self.device)
            #ends = torch.tensor([e.tolist() for e, act_ty, slot_ty in zip(ends, action_type, slot_type) if act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1],
                                #device=self.device)
            assert (cat_ids.shape[0] == starts.shape[0])
            cat_ids = [ cat_id for cat_id, act_ty, slot_ty in zip(cat_ids, action_type, slot_type) if
             act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1]
            if len(cat_ids) > 0:
                cat_ids = torch.stack(cat_ids, dim = 0)
                cat_attention_mask = [cat_attn for cat_attn, act_ty, slot_ty in zip(cat_attention_mask, action_type, slot_type) if
                           act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1]
                cat_attention_mask = torch.stack(cat_attention_mask, dim=0)
                starts = [s for s, act_ty, slot_ty in zip(starts, action_type, slot_type) if
                           act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1]
                starts = torch.stack(starts, dim=0)
                ends = [e for e, act_ty, slot_ty in zip(ends, action_type, slot_type) if
                           act_ty in ['INFORM', 'AFFIRM', 'SELECT'] and slot_ty == 1]
                ends = torch.stack(ends, dim=0)

                start_logits = None
                end_logits = None

                cat_hidden_state = self.distilbert(input_ids=cat_ids, attention_mask=cat_attention_mask).last_hidden_state
                #cat_hidden_state = torch.cat((x_outputs.last_hidden_state, y_outputs.last_hidden_state), dim=1)
                cat_hidden_state = self.dropout(cat_hidden_state)
                logits = self.slotValueHeader(cat_hidden_state)
                assert (logits.shape[0] == cat_ids.shape[0])
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(dim=-1)
                end_logits = end_logits.squeeze(dim=-1)

                valueLossFn = self.valueLossFn

                start_loss = valueLossFn(start_logits, starts)
                end_loss = valueLossFn(end_logits, ends)
                value_loss = (start_loss + end_loss) / 2 

            total_loss = retrieve_loss + ss_loss + value_loss

        return RetrieveBasedDSTModelOutput(
            loss=total_loss,
            retrieve_loss=retrieve_loss,
            slot_loss=ss_loss,
            value_loss=value_loss
        )

    def encode(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        self.eval()
        with torch.no_grad():
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = self.retrieveHeader(embeddings)
        return embeddings


    def predict(self, input_text):

        tokenized_input_text = common.tokenizer(input_text, return_tensors='pt', padding=True)

        input_ids = tokenized_input_text["input_ids"].to(self.device)
        attention_mask = tokenized_input_text["attention_mask"].to(self.device)

        self.eval()
        with torch.no_grad():
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            x_embeddings = outputs.last_hidden_state[:, 0, :]
            x_embeddings = self.retrieveHeader(x_embeddings)

            #predicte slot
            x_q = x_embeddings.cpu().numpy()
            slotsamples = create_slot_sample_dataloader(x_q, input_text, self.config.k, mode="predict")

            res = defaultdict(list)
            first_res = []
            for ss in slotsamples:
                input_text = ss['input_text']
                slot_desc = ss["pred_slotdesc"]

                ss_output = self.distilbert(input_ids=ss['input_ids'],
                                                   attention_mask=ss['input_attention_mask'])
                ss_embeddings = ss_output.last_hidden_state[:, 0, :]
                #ss_embeddings = self.dropout(ss_embeddings)
                ss_logits = self.slotHeader(ss_embeddings) # bs, 2
                if ss['label'] is None:
                    #softmax_ss_logits = nn.functional.softmax(ss_logits, dim=-1)
                    probs, labels = torch.max(ss_logits, dim=-1)
                    predicted_slots = [
                        { "input_text": in_text,
                          "slot": slot,
                          "last_hidden_state": last_hidden_state
                         }
                        for in_text, slot, last_hidden_state, label
                        in zip(input_text, slot_desc, ss_output.last_hidden_state, labels)
                        if label == 1
                    ]
                if not predicted_slots:
                    predicted_slots = [ { "input_text": input_text[0],
                                    "slot": slot_desc[0], "last_hidden_state": ss_output.last_hidden_state[0]} ]
            
            # predict slot value
            for slot in predicted_slots:
                slot_desc = json.loads(slot["slot"])
                act = slot_desc["act"]
                if 'slot_value' in slot_desc.keys():
                    slot_value = slot_desc["slot_value"]
                    res[hash(slot['input_text'])].append(
                        {"slot": slot_desc, 'slot_value': slot_value, "act": act})

                # non-catergorial
                elif act in ["INFORM", "AFFIRM", "SELECT"]:

                    last_hidden_state = slot["last_hidden_state"]
                    #last_hidden_state = self.dropout(last_hidden_state)
                    logits = self.slotValueHeader(last_hidden_state)
                    start_logits, end_logits = logits.split(1, dim=-1)
                    start_logits = start_logits.squeeze(dim=-1)
                    end_logits = end_logits.squeeze(dim=-1)
                    start = torch.argmax(start_logits, dim = -1)
                    end = torch.argmax(end_logits, dim = -1)
                    #convert start, end to start_pos, end_pos
                    itext = slot["input_text"]
                    tokenized_text = common.tokenizer(itext, return_offsets_mapping=True)
                    offset = tokenized_text["offset_mapping"]
                    start = clamp(start,0, sum(tokenized_text['attention_mask'])-1)
                    end = clamp(end, 0, sum(tokenized_text['attention_mask'])-1)
                    start_pos = offset[start][0]
                    end_pos = offset[end][1]
                    if start_pos  == 0 and end_pos ==0:
                        slot_value = 'dontcare'
                    else:
                        slot_value = itext[start_pos:end_pos]
                        if act == 'INFORM':
                            subtext = itext.split("><")
                            start_pos = start_pos - len(subtext[0]) - 2
                            end_pos = end_pos - len(subtext[0]) - 2
                        else:
                            start_pos = None
                            end_pos = None
                    res[hash(itext)].append(
                        {"slot": slot_desc, 'act': act, 'slot_value': slot_value,'start_pos':start_pos, 'end_pos': end_pos})
                else:
                    res[hash(slot["input_text"])].append(
                        {"slot": slot_desc, 'slot_value': None, 'act': act})

            return res

    def retrieve(self, input_text, k):
        tokenized_input_text = common.tokenizer(input_text, return_tensors='pt', padding=True)

        input_ids = tokenized_input_text["input_ids"].to(self.device)
        attention_mask = tokenized_input_text["attention_mask"].to(self.device)

        self.eval()
        with torch.no_grad():
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            x_embeddings = outputs.last_hidden_state[:, 0, :]
            x_embeddings = self.retrieveHeader(x_embeddings)

            # predicte slot
            x_q = x_embeddings.cpu().numpy()
            D, I = common.index.search(x_q, k)
            return D, I

