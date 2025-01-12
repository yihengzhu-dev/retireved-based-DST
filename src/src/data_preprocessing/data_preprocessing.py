import json
from collections import defaultdict

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from src.RBDST.common import common
from .data_tokennizer import tokenizer
from .samples import Samples


def createEamplesFromDialogue(dialogue, intent_slot_descs, slot_categorical_dict):
    """
        create training example from dialogue.
        training example has the following fields:
            input_text: <system utterence in previous turn><user utterence of current turn><intent=description of intent in previous turn>. use as x.
            dialogue_id: ,
            example_id: ,
            slot_type: 0 for none slot, intent only, 1 for categorical slot, 2 for non-categorical slot, 3 for requested slot. use as y.
            slot_desc: slot description from schema, use as y.
            slot_value: slot value for non-categorical slot. use as y.

        input:
              dialogue: dialogue in dialogue file with json format.
              intent_slot_descs: intent_slot_descs is created from schema, more in create_slot_description_from_schema.

        returns: samples.

        """

    samples = Samples()
    slotkeys = defaultdict(list)

    dialogue_id = dialogue['dialogue_id']
    turns = dialogue['turns']

    pre_intents = defaultdict(str)
    system_utterance = ""
    system_utterance_len = len(system_utterance)
    pre_system_offered_intent = defaultdict(str)
    pre_system_offered_slots = defaultdict(lambda: defaultdict(str))
    pre_system_confirm_slots = defaultdict(lambda: defaultdict(str))

    for turn in turns:
        if turn['speaker'] == 'USER':
            user_utterence = turn['utterance']
            #print(user_utterence)
            frames = turn['frames']
            for frame in frames:
                service = frame['service']
                pre_intent = pre_intents[service]
                if pre_intent == None or pre_intent == "":
                    intent_slot_desc = ""
                else:
                    intent_slot_desc = intent_slot_descs[f"{service}-{pre_intent}"]
                input_text = f"<{system_utterance}><{user_utterence}>{intent_slot_desc}"

                #print(input_text)

                active_intent = frame['state']['active_intent']
                #if (active_intent == "" or active_intent is None
                        #or active_intent == "NONE"):
                if active_intent is None or active_intent == "":
                    active_intent = "NONE"

                all_slots_values_of_this_frame = frame['state']['slot_values']
                non_categorical_slots_t = frame['slots']

                non_categorical_slots = {}
                for s in non_categorical_slots_t:
                    non_categorical_slots[s['slot']] = [s["start"] + 3 + system_utterance_len,
                                                        s["exclusive_end"] + 3 + system_utterance_len]

                actions = frame['actions']
                for action in actions:
                    act = action['act']
                    action_type = act
                    if act == 'INFORM_INTENT':
                        active_intent = frame['state']['active_intent']
                        slot_desc_key = f"{service}-{active_intent}-{act}"
                        intent_slot_desc = intent_slot_descs[slot_desc_key]
                        slot_type = 2
                        slot_value = ""
                        samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, slot_value,
                                          slot_desc_key, action_type)
                        slotkeys[hash(input_text)].append(slot_desc_key)
                    elif act == 'REQUEST':
                        slot_name = action['slot']
                        slot_desc_key = f"{service}-{active_intent}-{slot_name}-{act}"
                        intent_slot_desc = intent_slot_descs[slot_desc_key]
                        slot_type = 2
                        slot_value = ""
                        #if slot_categorical_dict[service][slot_name] == True:
                        #slot_type = 0
                        samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, slot_value,
                                          slot_desc_key, action_type)
                        slotkeys[hash(input_text)].append(slot_desc_key)
                    elif act == 'INFORM':
                        slot_name = action['slot']
                        if slot_categorical_dict[service][slot_name] == True:
                            slot_value = all_slots_values_of_this_frame[slot_name]
                            for s_v in slot_value:
                                slot_type = 0
                                slot_desc_key = f"{service}-{active_intent}-{slot_name}-{s_v}-{act}"
                                intent_slot_desc = intent_slot_descs[slot_desc_key]
                                samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, s_v,
                                                  slot_desc_key, action_type)
                                slotkeys[hash(input_text)].append(slot_desc_key)
                        else:
                            slot_type = 1
                            if 'dontcare' in action['values']:
                                slot_value = ""
                            else:
                                slot_value = non_categorical_slots[slot_name]
                            #slot_value = non_categorical_slots[slot_name]
                            slot_desc_key = f"{service}-{active_intent}-{slot_name}-{act}"
                            intent_slot_desc = intent_slot_descs[slot_desc_key]
                            samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, slot_value,
                                              slot_desc_key, action_type)
                            slotkeys[hash(input_text)].append(slot_desc_key)
                    elif act == 'AFFIRM_INTENT':
                        slot_desc_key = f"{service}-{active_intent}-{act}"
                        intent_slot_desc = intent_slot_descs[slot_desc_key]
                        slot_type = 2
                        slot_value = ""
                        samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, slot_value,
                                          slot_desc_key, action_type)
                        slotkeys[hash(input_text)].append(slot_desc_key)
                    elif act == 'NEGATE_INTENT':
                        offered_intent = pre_system_offered_intent[service]
                        slot_desc_key = f"{service}-{offered_intent}-{act}"
                        intent_slot_desc = intent_slot_descs[slot_desc_key]
                        slot_type = 2
                        slot_value = ""
                        samples.addSample(input_text, dialogue_id,  intent_slot_desc, slot_type, slot_value,
                                          slot_desc_key, action_type)
                        slotkeys[hash(input_text)].append(slot_desc_key)

                    elif act == 'AFFIRM':
                        confirm_slots = pre_system_confirm_slots[service]
                        confirm_slots_keys = list(confirm_slots.keys())
                        for confirm_slot_name in confirm_slots_keys:
                            if slot_categorical_dict[service][confirm_slot_name] == True:
                                slot_type = 0
                                confirm_slot_value = all_slots_values_of_this_frame[confirm_slot_name]
                                for v in confirm_slot_value:
                                    slot_desc_key = f"{service}-{active_intent}-{confirm_slot_name}-{v}-{act}"
                                    intent_slot_desc = intent_slot_descs[slot_desc_key]
                                    samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, v,
                                                      slot_desc_key, action_type)
                                    slotkeys[hash(input_text)].append(slot_desc_key)
                            else:
                                confirm_slot_value = confirm_slots[confirm_slot_name]
                                slot_type = 1
                                slot_desc_key = f"{service}-{active_intent}-{confirm_slot_name}-{act}"
                                intent_slot_desc = intent_slot_descs[slot_desc_key]
                                samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type,
                                                  confirm_slot_value,
                                                  slot_desc_key, action_type)
                                slotkeys[hash(input_text)].append(slot_desc_key)

                    elif act == 'SELECT':
                        sel_slot_name = action['slot']
                        if sel_slot_name == "":
                            offered_slots = pre_system_offered_slots[service]
                            offered_slots_keys = list(offered_slots.keys())
                            for offered_slot_name in offered_slots_keys:
                                offered_slot_value = offered_slots[offered_slot_name]

                                if slot_categorical_dict[service][offered_slot_name] == True:
                                    slot_type = 0
                                    slot_desc_key = f"{service}-{active_intent}-{offered_slot_name}-{offered_slot_value}-{act}"
                                    intent_slot_desc = intent_slot_descs[slot_desc_key]
                                else:
                                    slot_type = 1
                                    slot_desc_key = f"{service}-{active_intent}-{offered_slot_name}-{act}"
                                    intent_slot_desc = intent_slot_descs[slot_desc_key]
                                samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type,
                                                  offered_slot_value,
                                                  slot_desc_key, action_type)
                                slotkeys[hash(input_text)].append(slot_desc_key)

                        else:
                            if slot_categorical_dict[service][sel_slot_name] == True:
                                slot_value = all_slots_values_of_this_frame[sel_slot_name]
                                slot_type = 0
                                slot_desc_key = f"{service}-{active_intent}-{sel_slot_name}-{slot_value}-{act}"
                                intent_slot_desc = intent_slot_descs[slot_desc_key]
                                samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, slot_value,
                                                  slot_desc_key, action_type)
                                slotkeys[hash(input_text)].append(slot_desc_key)
                            else:
                                slot_type = 1
                                slot_value = pre_system_offered_slots[service][sel_slot_name]
                                slot_desc_key = f"{service}-{active_intent}-{sel_slot_name}-{act}"
                                intent_slot_desc = intent_slot_descs[slot_desc_key]
                                samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, slot_value,
                                                  slot_desc_key, action_type)
                                slotkeys[hash(input_text)].append(slot_desc_key)
                        #pre_system_offered_slots.clear()
                    elif act == 'REQUEST_ALTS' or act == 'THANK_YOU' or act == 'GOODBYE' or act == 'NEGATE':
                        slot_desc_key = f"{service}-{active_intent}-{act}"
                        intent_slot_desc = intent_slot_descs[slot_desc_key]
                        slot_type = 2
                        slot_value = ""
                        samples.addSample(input_text, dialogue_id, intent_slot_desc, slot_type, slot_value,
                                          slot_desc_key, action_type)
                        slotkeys[hash(input_text)].append(slot_desc_key)

                    else:
                        continue
                pre_intents[service] = active_intent

            pre_system_offered_intent.clear()
            pre_system_offered_slots.clear()
            pre_system_confirm_slots.clear()
        else:
            system_utterance = turn['utterance']
            system_utterance_len = len(system_utterance)
            frames = turn['frames']
            for frame in frames:
                service = frame['service']
                actions = frame['actions']
                non_categorical_slots_st = frame['slots']
                #non_categorical_slot_keys_st = [s['slot'] for s in non_categorical_slots_st]
                non_categorical_slots_s = {}
                for ss in non_categorical_slots_st:
                    non_categorical_slots_s[ss['slot']] = [ss["start"] + 1,
                                                           ss["exclusive_end"] + 1]

                for action in actions:
                    act = action['act']
                    if act == 'OFFER_INTENT':
                        pre_system_offered_intent[service] = action['values'][0]
                    elif act == 'OFFER':
                        offered_slot_name = action['slot']
                        offered_slot_value = action['values'][0]
                        if slot_categorical_dict[service][offered_slot_name] == False:
                            offered_slot_value = non_categorical_slots_s[offered_slot_name]
                        pre_system_offered_slots[service][offered_slot_name] = offered_slot_value
                    elif act == 'CONFIRM':
                        confirm_slot_name = action['slot']
                        confirm_slot_value = action['values'][0]
                        if slot_categorical_dict[service][confirm_slot_name] == False:
                            confirm_slot_value = non_categorical_slots_s[confirm_slot_name]
                        pre_system_confirm_slots[service][confirm_slot_name] = confirm_slot_value
                    else:
                        continue
    return samples, slotkeys

def buildSlotSamples(x_embeddings, input_texts, k, mode='train'):
    bs = x_embeddings.shape[0]
    index = common.index
    slotdescdb = common.slotdescdb
    slotdescs = common.slotdescs

    D, I = index.search(x_embeddings, k)

    samples = defaultdict(list)

    for i in range(bs):

        input_text = input_texts[i]
        pred_slotdescs = [ slotdescdb[j] for j in I[i] ]

        if mode == 'train':
            slotdesckeys = common.slotkeys
            y_slotdescs = [slotdescs[key] for key in slotdesckeys[hash(input_text)]]
            n_y = len(y_slotdescs)
            x_input_text = [input_text for _ in range(n_y)]
            samples["input_text"].extend(x_input_text)
            samples["pred_slotdesc"].extend(y_slotdescs)
            samples["label"].extend([1 for _ in range(n_y)])

            label = [ 1 if pred_slotdesc in y_slotdescs else 0 for pred_slotdesc in pred_slotdescs ]
            samples["label"].extend(label)
        x_input_text = [ input_text for _ in range(k) ]
        samples["input_text"].extend(x_input_text)
        samples["pred_slotdesc"].extend(pred_slotdescs)

    return samples

def tokenize_slot_samples(slot_samples, tokenizer=tokenizer):
    input_text = [ itext + stext for itext, stext in zip(slot_samples["input_text"], slot_samples["pred_slotdesc"]) ]

    return tokenizer(input_text)

def slot_samples_collate(batch, tokenizer=tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = [ b["input_ids"] for b in batch ]
    input_attention_mask = [ b["attention_mask"] for b in batch ]
    input_text = {"input_ids": input_ids, "attention_mask": input_attention_mask}
    input_text = tokenizer.pad(input_text)

    train = True if 'label' in batch[0].keys() else False
    target = [ b["label"] for b in batch ] if train  else None

    intext = [b['input_text'] for b in batch ]
    slottext = [b['pred_slotdesc'] for b in batch ]

    res = {
        "input_ids": torch.tensor(input_text["input_ids"]).to(device),
        "input_attention_mask": torch.tensor(input_text["attention_mask"]).to(device),

        "label": torch.tensor(target).to(device) if target is not None else None,
        "input_text": intext,
        "pred_slotdesc": slottext
    }

    return res


def create_slot_sample_dataloader(x_embeddings, input_texts, k, mode="train"):
    samples =  buildSlotSamples(x_embeddings, input_texts, k, mode)
    dataset = Dataset.from_dict(samples)
    batch_size = 32
    tokenized_dataset = dataset.map(tokenize_slot_samples, batched=True, batch_size=batch_size)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=slot_samples_collate, shuffle=True)

    return dataloader

def load_dialogues(file_path):
    dialogues = []
    for file in file_path:
        with open(file, 'r', encoding='utf-8') as f:
            dialogues.extend(json.load(f))
    return dialogues
