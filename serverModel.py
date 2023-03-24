
import torch
import torch.nn as nn
from transformers import  AutoTokenizer, AutoModel, logging
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
from flask import Flask, request
import json
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

app = Flask(__name__) 
@app.route('/arraysum', methods = ['POST'])
def classification(): 
    data = request.get_json()
    classifi=infer(data['content'])
    print(classifi)
    # Return data in json format 
    return json.dumps({"result":classifi.item()})

if __name__ == "__main__": 
    app.run(port=5000)