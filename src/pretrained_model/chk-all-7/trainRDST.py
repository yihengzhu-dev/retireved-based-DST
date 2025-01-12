import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict

import faiss
import torch
from transformers import DistilBertConfig, DistilBertModel, get_linear_schedule_with_warmup

from src.data_preprocessing.data_preprocessing import createEamplesFromDialogue, load_dialogues
from src.data_preprocessing.data_tokennizer import createDataLoader, tokenizer
from src.data_preprocessing.samples import Samples
from src.data_preprocessing.schema_util import create_slot_description_from_schema, create_slot_categorical_dict, \
    create_slot_desc_db
from src.RBDST.retrieveBasedDST import RetrieveBasedDST
from src.RBDST.trainerForRDST import TrainArgs, TrainerForRDST

from src.RBDST.common import set_seed, common

set_seed(42)

def train(task):
    dialogue_dir = '../dstc8/train'

    if task == 'single':
        dialogue_file = [os.path.join(dialogue_dir, "dialogues_{:03d}.json".format(i)) for i in range(1, 44) ]
    elif task =='multiple':
        dialogue_file = [ os.path.join(dialogue_dir, "dialogues_{:03d}.json".format(i)) for i in range(44, 128) ]
    else:
        dialogue_file = glob.glob(os.path.join(dialogue_dir, "dialogues_*.json"))

    dialogue = load_dialogues(dialogue_file)

    train_schema_dir = os.path.join('../dstc8/train', 'schema.json')
    slotdescs = create_slot_description_from_schema(train_schema_dir)
    slot_categorical_dict = create_slot_categorical_dict(train_schema_dir)
    slotdescdb = create_slot_desc_db(slotdescs)

    datas = Samples()
    slotkeys = defaultdict(list)
    for dial in dialogue:
        samples, sks = createEamplesFromDialogue(dial, slotdescs, slot_categorical_dict)
        datas.extendSamples(samples)
        for k, v in sks.items():
            slotkeys[k].extend(v)
    # datas = createEamplesFromDialogue(dialogue[0], intent_slot_descs, slot_categorical_dict)

    batch_size = 16
    dataLoader = createDataLoader(datas, batch_size)

    index = faiss.IndexFlatIP(256)
    # model = RetrieveBasedDST.from_pretrained('./pretrained_model/RBDST')
    config = DistilBertConfig()
    config.k = 10
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = RetrieveBasedDST(config)
    model.distilbert.load_state_dict(distilbert.state_dict())

    num_epochs = 10
    num_train_steps = num_epochs * len(dataLoader)

    common.index = index
    common.slotkeys = slotkeys
    common.slotdescs = slotdescs
    common.slotdescdb = slotdescdb
    common.tokenizer = tokenizer

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_train_steps,
                                                   num_training_steps=num_train_steps)
    trainArgs = TrainArgs(dataLoader=dataLoader, model=model, lr_scheduler=lr_scheduler, optimizer=optimizer,
                          num_epochs=num_epochs)
    trainArgs.task = task

    trainer = TrainerForRDST(trainArgs)

    trainer.train()

if __name__ == '__main__':
    #task = 'single'
    #task = 'multiple'
    task = 'all'
    train(task)
