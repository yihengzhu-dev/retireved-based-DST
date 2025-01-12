import os

import faiss
from faiss import write_index

from src.RBDST.common import common
from src.data_preprocessing.schema_util import update_slotDB_index

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from tqdm.auto import tqdm

class TrainArgs(object):
    def __init__(self, model=None,dataLoader=None, lr = 5e-2, lr_scheduler=None, optimizer=None,
                 num_epochs=1):
        self.model = model
        self.dataLoader = dataLoader
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.optimizer = optimizer
        self.num_epochs = num_epochs

class TrainerForRDST:
    def __init__(self, trainArgs):
        self.TrainArgs = trainArgs

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataLoader = self.TrainArgs.dataLoader
        model = self.TrainArgs.model
        model.to(device)
        num_epochs = self.TrainArgs.num_epochs
        num_train_steps = num_epochs * len(dataLoader)
        lr_scheduler = self.TrainArgs.lr_scheduler
        optimizer = self.TrainArgs.optimizer
        task = self.TrainArgs.task

        progress_bar = tqdm(range(num_train_steps))

        index_dir = "index"
        mode_dir = "pretrained_model"
        model.train()
        for epoch in range(num_epochs):

            for batch_idx, batch in enumerate(dataLoader):
                output = model(batch)
                loss = output.loss
                loss.backward()

                print(f"loss: {loss}, retrieve loss: {output.retrieve_loss}, value loss: {output.value_loss}, slot_loss: {output.slot_loss})")

                #print(f"layer0 grad: {model.distilbert.transformer.layer[0].attention.q_lin.weight.grad.mean()}")
                #print(f"retrieve grad: {model.retrieveHeader.weight.grad.mean()}")
                #print(f"value grad: {model.slotValueHeader.weight.grad.mean()}")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del output
                del loss
                torch.cuda.empty_cache()
                progress_bar.update(1)
            update_slotDB_index(model)
            # save index and model after every epoch
            index_file = f"{index_dir}/chk-{task}-{epoch}.index"
            write_index(common.index, index_file)
            model_path = f"{mode_dir}/chk-{task}-{epoch}"
            model.save_pretrained(model_path)