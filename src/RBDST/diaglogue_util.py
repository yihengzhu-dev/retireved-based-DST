import json
from collections import defaultdict

import numpy as np

from src.RBDST.common import common


def create_new_frame():
    new_frame = {'service': "",
                 'slots': [],
                 'state': {
                     "active_intent": "",
                     "slot_values": {},
                     "requested_slots": []
                 }}
    return new_frame


def update_frame_by_slots(slots, pre_turn_ref, slot_dict):
    frame_hyp_dict = defaultdict(lambda: create_new_frame())

    service_in_ref = []
    frame_ref_dict = {}
    if pre_turn_ref is not None:
        service_in_ref = [ frame["service"] for frame in pre_turn_ref["frames"] ]
        frame_ref_dict = { frame["service"]: frame for frame in pre_turn_ref["frames"] }

    for slot in slots:
        act = slot['act']
        inslot = slot['slot']
        service = inslot['service_name']
        intent = inslot['intent_name']
        #print(f"{intent}: {act}")
        frame_ = frame_hyp_dict[service]
        frame_['service'] = service
        if service in service_in_ref:
            frame_['state']['slot_values'] = frame_ref_dict[service]['state']['slot_values']
        if act == 'INFORM_INTENT':
            frame_['state']['active_intent'] = intent
        elif act in ['INFORM', 'AFFIRM', 'SELECT']:

            slot_name = inslot['slot_name']
            slot_value = slot['slot_value']
            frame_['state']['active_intent'] = intent
            ssv = []
            ssv.append(slot_value)
            frame_['state']['slot_values'][slot_name] = ssv
            if slot_dict[service][slot_name] == False and act == 'INFORM':
                start_pos = slot['start_pos']
                end_pos = slot['end_pos']
                frame_['slots'].append({'slot': slot_name, 'start': start_pos, 'exclusive_end': end_pos})
        elif act == 'NEGATE_INTENT' or act == 'NEGATE':
            if frame_['state']['active_intent'] == "":
                frame_['state']['active_intent'] = "NONE"
                print(f"{intent}: {act}")
        elif act == 'REQUEST':

            slot_name = inslot['slot_name']
            frame_['state']['active_intent'] = intent
            frame_['state']["requested_slots"].append(slot_name)
        else:

            frame_['state']['active_intent'] = intent
    return frame_hyp_dict

def predict_dialogue(dialogue, intent_slot_descs, slot_categorical_dict, model):
    dialogue_id = dialogue['dialogue_id']
    pre_turn_ref = None
    system_utterance= ""
    pre_intents = defaultdict(str)

    turns_hyp = []
    dialogue_hyp = { 'dialogue_id': dialogue_id, "turns": turns_hyp }
    service_set = set()

    for turn in dialogue['turns']:

        if turn['speaker'] == 'USER':
            user_utterence = turn['utterance']
            input_text = []
            turn_hyp = {"speaker": "USER", "utterance": user_utterence, "frames": []}
            turns_hyp.append(turn_hyp)
            for frame in turn['frames']:
                service = frame['service']
                pre_intent = pre_intents[service]
                if pre_intent == None or pre_intent == '':
                    pre_intent = 'NONE'

                intent_slot_desc = intent_slot_descs[f"{service}-{pre_intent}"]
                input_text.append(f"<{system_utterance}><{user_utterence}>{intent_slot_desc}")
                active_intent = frame['state']['active_intent']
                pre_intents[service] = active_intent

            slot_hyp = model.predict(input_text)
            slot_hyp = [v for _, vs in slot_hyp.items() for v in vs]
            if not slot_hyp:
                D, I = model.retrieve(input_text, 1)
                D = [score for r in D for score in r]
                I = [ ind for r in I for ind in r ]
                ind = I[np.argmax(D)]
                s_j = json.loads(common.slotdescdb[ind])
                slot_hyp = [ {"act": s_j['act'], "slot": s_j, "slot_value": None, "start_pos": 0, "end_pos": 0} ]
            frames_hyp = update_frame_by_slots(slot_hyp, pre_turn_ref, slot_categorical_dict)
            turn_hyp["frames"] = list(frames_hyp.values())
            service_set.update(set(frames_hyp.keys()))
            pre_turn_ref = turn

        else:
            system_utterance = turn['utterance']
            turn_hyp = {"speaker": "SYSTEM", "utterance": system_utterance}
            turns_hyp.append(turn_hyp)
    dialogue_hyp["services"] = list(service_set)
    return dialogue_hyp