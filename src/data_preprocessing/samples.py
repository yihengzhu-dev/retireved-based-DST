class Samples(object):
    def __init__(self):

        self.slot_desc_keys = []

        self.dialogue_ids = []
        self.example_ids = []
        self.input_texts = []
        self.slot_types = []
        self.slot_descs = []
        self.slot_values = []
        self.action_types = []

    def addSample(self, input_text, dialogue_id=None, slot_desc=None, slot_type=None,
                                                  slot_value=None, slot_desc_keys=None, action_type=None):
        self.dialogue_ids.append(dialogue_id)
        self.input_texts.append(input_text)

        self.slot_types.append(slot_type)
        self.slot_descs.append(slot_desc)

        if not isinstance(slot_value, list):
            slot_value = [0,0]
        self.slot_values.append(slot_value)
        self.slot_desc_keys.append(slot_desc_keys)

        self.action_types.append(action_type)

    def toDict(self):
        return {
            'dialogue_ids': self.dialogue_ids,
            'input_texts': self.input_texts,
            'slot_types': self.slot_types,
            'slot_descs': self.slot_descs,
            'slot_values': self.slot_values,
            'slot_desc_keys': self.slot_desc_keys,
            'action_types': self.action_types
       }

    def getSamples(self, index):
        return [ {"dialogue_id": self.dialogue_ids[ind],
                  "input_text": self.input_texts[ind],
                  "slot_type": self.slot_types[ind],
                  "slot_desc": self.slot_descs[ind],
                  "slot_value": self.slot_values[ind],
                  "slot_desc_key": self.slot_desc_keys[ind],
                  "action_type": self.action_types[ind]
                  }
                 for ind in index ]

    def extendSamples(self, samples):
        self.dialogue_ids.extend(samples.dialogue_ids)
        self.input_texts.extend(samples.input_texts)
        self.slot_types.extend(samples.slot_types)
        self.slot_descs.extend(samples.slot_descs)
        self.slot_values.extend(samples.slot_values)
        self.slot_desc_keys.extend(samples.slot_desc_keys)
        self.action_types.extend(samples.action_types)

