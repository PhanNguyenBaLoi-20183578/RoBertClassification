import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from gensim.utils import simple_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import  AutoTokenizer, AutoModel, logging

import warnings
warnings.filterwarnings("ignore")

logging.set_verbosity_error()





class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x
class_names = ['ham','spam']
model = torch.load('./phobert_pre.pth',map_location=torch.device('cpu'))

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
def infer(text, _tokenizer=tokenizer, max_len=120):
    encoded_review = _tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    #print(model(input_ids, attention_mask))
    output = model(input_ids, attention_mask)
    _, y_pred = torch.max(output, dim=1)

    return y_pred
#infer("test test teserrr hello",tokenizer)
