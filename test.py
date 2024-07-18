import os
import pandas as pd
import numpy as np
from nlpaug.util import Action
from nlpaug.flow import Pipeline
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from transformers import BertConfig, BertTokenizerFast, BertModel

# MY_DEVICE = torch.device('mps')
# DATASET_NAME = './NlpAnalytics/data/dummy_data/amazon_train.csv'
# TRANSLATION_MODEL_1 = 'Helsinki-NLP/opus-mt-en-fr'
# TRANSLATION_MODEL_2 = 'Helsinki-NLP/opus-mt-fr-en'

# class TextGenerator(Pipeline):
    
#     def __init__(self, flow=None, name='Sometimes_Pipeline', aug_p=0.8, verbose=0):
#         Pipeline.__init__(self, name=name, action=Action.SOMETIMES,
#                           flow=flow, aug_p=aug_p, include_detail=False, verbose=verbose)

#     def augment(self, data, n=1):
#         augmented_results = [self._augment(data) for _ in range(n)]
#         augmented_results = [r for sub_results in augmented_results for r in sub_results if len(r) > 0]
#         df_res = pd.DataFrame(columns=['org', 'aug1', 'aug2'])
#         df_res['org'] = data
#         df_res['aug1'] = augmented_results[0]
#         df_res['aug2'] = augmented_results[1]
#         return df_res

#     def draw(self):
#         return self.aug_p > self.prob()


if __name__ == '__main__':

    # # load dataset
    # df = pd.read_csv(DATASET_NAME)
    # df = df[['id', 'text']]

    # # load data generator
    # data_generator = TextGenerator([
    #     naw.RandomWordAug(),
    #     naw.BackTranslationAug(
    #         from_model_name=TRANSLATION_MODEL_1, 
    #         to_model_name=TRANSLATION_MODEL_2,
    #         device=MY_DEVICE)]
    # )

    # # data augmentation
    # res = data_generator.augment(df.text.values.tolist()[:100], n = 2)
    # res.to_csv('test_result.csv', index=False)

    # ### load transformer model as benchmark
    # model_st = SentenceTransformer('all-mpnet-base-v2').to(torch.device('mps'))

    # train_dataset = Dataset.from_dict({
    # "sentence1": ["It's nice weather outside today.", "He drove to work.", "He drove to work.", "He drove to work."],
    # "sentence2": ["It's so sunny.", "She walked to the store.", "She walked to the store.", "She walked to the store."],
    # "label": [1, 0, 0, 0],
    # })
    # loss = losses.ContrastiveLoss(model_st)

    # trainer = SentenceTransformerTrainer(
    #     model=model_st,
    #     train_dataset=train_dataset,
    #     loss=loss,
    # )
    # trainer.train()
    
    student_model = SentenceTransformer("microsoft/mpnet-base").to(torch.device('mps'))
    teacher_model = SentenceTransformer("all-mpnet-base-v2").to(torch.device('mps'))
    train_dataset = Dataset.from_dict({
        "english": ["The first sentence",  "The second sentence", "The third sentence",  "The fourth sentence"],
        "french": ["La première phrase",  "La deuxième phrase", "La troisième phrase",  "La quatrième phrase"],
    })

    def compute_labels(batch):
        return {
            "label": teacher_model.encode(batch["english"])
        }

    train_dataset = train_dataset.map(compute_labels, batched=True)
    loss = losses.MSELoss(student_model)

    trainer = SentenceTransformerTrainer(
        model=student_model,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
