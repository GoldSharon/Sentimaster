from flask import Flask, render_template, request
import tensorflow as tf
from Fine_Tuning.GPT_2 import GPTModel
from Fine_Tuning.GPT_2 import Utils 
import torch
import tiktoken
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


## GPT 124M


GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length": 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : True,
    'n_layers' :12
}

tokenizer = tiktoken.get_encoding("gpt2")

device = "cpu"


gpt = GPTModel(GPT_CONFIG_124M)

gt = Utils()


## Loading Weights

checkpoint = torch.load("model_and_optimizer.pth");
gpt.load_state_dict(checkpoint['model_state_dict']);
optimizer = torch.optim.Adam(gpt.parameters(),lr=0.0004,weight_decay=0.1);
optimizer.load_state_dict(checkpoint['optimizer_state_dict']);
gpt.train();

## Classification

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "Positive" if predicted_label == 1 else "Negative"

## Server

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result")
def result():
    flag = request.args.get('flag')
    return render_template("result.html", flag=flag)

@app.route("/submit", methods=['POST'])
def predict():
    review = request.form.get('Check', "")
    if not review:
        return render_template("index.html", error="Input cannot be empty")
    else:
        result = classify_review(review, gpt, tokenizer, device, max_length=40)
        flag = '1' if result == "Positive" else '0'
        return render_template("result.html", flag=flag)

if __name__ == "__main__":
    app.run(debug=True)
