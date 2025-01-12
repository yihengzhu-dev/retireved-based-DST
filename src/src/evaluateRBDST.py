import glob
import json
import os

import faiss
import torch
from faiss import read_index
from tqdm import tqdm

from src.RBDST.common import common
from src.RBDST.diaglogue_util import predict_dialogue
from src.RBDST.retrieveBasedDST import RetrieveBasedDST
from src.data_preprocessing.data_preprocessing import load_dialogues
from src.data_preprocessing.data_tokennizer import tokenizer
from src.data_preprocessing.schema_util import create_slot_description_from_schema, create_slot_categorical_dict, \
    create_slot_desc_db, update_slotDB_index
from src.evaluate.evaluate import evaluationOption, evaluate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def generate_hyp_dialogue():
    dialogue_dir = '../dstc8/test'

    evaluationOption.dstc8_data_dir = "../dstc8"
    evaluationOption.eval_set = "test"
    evaluationOption.prediction_dir = "../evaluation/prediction"
    evaluationOption.output_metric_file = os.path.join("../evaluation/metrics", "metrics.txt")

    evaluationOption.joint_acc_across_turn = False
    evaluationOption.use_fuzzy_match = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RetrieveBasedDST.from_pretrained("./pretrained_model/chk-all-7")
    model.to(device)

    test_schema_dir = os.path.join('../dstc8/test', 'schema.json')
    slotdescs = create_slot_description_from_schema(test_schema_dir)
    slot_categorical_dict = create_slot_categorical_dict(test_schema_dir)
    slotdescdb = create_slot_desc_db(slotdescs)

    index = faiss.IndexFlatIP(256)

    common.slotdescs = slotdescs
    common.slotdescdb = slotdescdb
    common.tokenizer = tokenizer
    common.index = index

    update_slotDB_index(model)

    #progress_bar = tqdm(range(34))
    dialogue_file = [os.path.join(dialogue_dir, "dialogues_{:03d}.json".format(i)) for i in range(1, 35)]
    for idx, dial in enumerate(dialogue_file):
        dial = json.load(open(dial,"r", encoding="utf-8"))
        dialogue_hyp = []
        for d in dial:
            dial_hyp = predict_dialogue(d, slotdescs, slot_categorical_dict, model)
            dialogue_hyp.append(dial_hyp)
        json.dump(dialogue_hyp, open(os.path.join(evaluationOption.prediction_dir, "dialogue_hyp_all-{:03d}.json".format(idx+1)), "w"))
        #progress_bar.update(1)

def evaluate_hyp_dialogue():
    dialogue_dir = '../dstc8/test'

    evaluationOption.dstc8_data_dir = "../dstc8"
    evaluationOption.eval_set = "test"
    evaluationOption.prediction_dir = "../evaluation/prediction"
    evaluationOption.output_metric_file = os.path.join("../evaluation/metrics", "metrics.txt")

    evaluationOption.joint_acc_across_turn = False
    evaluationOption.use_fuzzy_match = True
    evaluate()
if __name__ == '__main__':

    generate_hyp_dialogue()
    evaluate_hyp_dialogue()





