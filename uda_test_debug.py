import os
import pandas as pd
import numpy as np
# torch
import torch
from torch import Generator
# local
from uda_data import *
from uda_model import *
from uda_optimizer import *
from uda_trainer import *

### global var
DEVICE = torch.device('mps')
RANDOM_GENERATOR = Generator().manual_seed(42)
DATASET_NAME = 'uda_imdb_data_128'
PATH = '/Users/lunli/Library/CloudStorage/GoogleDrive-yaojn19880525@gmail.com/My Drive/Colab Notebooks/'
# configs
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 2
USE_LORA = False
BATCH_SIZE = {
    'sup' : 8,
    'unsup' : 24,
    'valid' : 16
}


if __name__ == '__main__':

    ### Data Loading
    # sup
    train_sup_data = torch.load(os.path.join(PATH, f'data/{DATASET_NAME}/train_sup_data.pt'))
    # unsup
    train_unsup_data = torch.load(os.path.join(PATH, f'data/{DATASET_NAME}/train_unsup_data.pt'))
    train_unsup_data = TensorDataset(
    torch.cat([train_unsup_data.tensors[0][:20000], train_unsup_data.tensors[0][-20000:]]),
    torch.cat([train_unsup_data.tensors[1][:20000], train_unsup_data.tensors[1][-20000:]]),
    torch.cat([train_unsup_data.tensors[2][:20000], train_unsup_data.tensors[2][-20000:]]),
    torch.cat([train_unsup_data.tensors[3][:20000], train_unsup_data.tensors[3][-20000:]]))
    # valid
    valid_data = torch.load(os.path.join(PATH, f'data/{DATASET_NAME}/val_data.pt'))
    # dataloader
    train_sup_dataloader = DataLoader(train_sup_data, sampler=RandomSampler(train_sup_data, generator=RANDOM_GENERATOR), batch_size=BATCH_SIZE['sup'])
    train_unsup_dataloader = DataLoader(train_unsup_data, sampler=RandomSampler(train_unsup_data, generator=RANDOM_GENERATOR), batch_size=BATCH_SIZE['unsup'])
    valid_dataloader = DataLoader(valid_data, sampler=RandomSampler(valid_data, generator=RANDOM_GENERATOR), batch_size=BATCH_SIZE['valid'])
    # organize the container
    datamodeler = {
        DataLoaderType.TRAINING: train_sup_dataloader,
        DataLoaderType.VALIDATION: valid_dataloader,
        DataLoaderType.TRAINING_UNLABELED: train_unsup_dataloader}
    print('1. DataLoader is ready. \n')

    ### load model and tokenizer
    tokenizer, model = load_bert_model(MODEL_NAME, num_labels=NUM_LABELS, device=DEVICE)
    print(f'2. Model and Tokenizer {MODEL_NAME} is Loaded. \n')

    ### load loss function for sup/unsup
    loss_sup = get_loss_functions(LossFuncType.CROSS_ENTROPY, reduce='none')
    loss_unsup = get_loss_functions(LossFuncType.KL_DIV, reduce='none')
    loss_dict = {'sup':loss_sup, 'unsup':loss_unsup}
    print('3. Loss Function Loaded. \n')

    ### Optimizer Set up    
    if not USE_LORA:
        optimizer = AdamNLP.newNLPAdam(model, {'embeddings':True, 'encoder': 9}, lr = 2e-4)
        model = optimizer.get_model_transformed()
    else:
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["query", "key", "value"], r=1, lora_alpha=1, lora_dropout=0.1)
        optimizer = AdamNLP.newNLPAdam_LORA(model, lora_config)
        model = optimizer.get_model_transformed()
    print(f'4. Optimizer is set up. (Lora is {USE_LORA}).')

    ### Start Training
    trainer = TrainerUDA(model, datamodeler, loss_dict, optimizer, report_freq=2, device=DEVICE)
    trainer.train(2, schedule_type = SchedulerType.INVERSE_SQRT, save_model_freq=-1)



