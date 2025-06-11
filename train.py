import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, TrainingArguments, Trainer


# Load your CSV with 'text', 'latitude', 'longitude'
df = pd.read_csv("full_markup_v2.csv")[['description', 'lat', 'lng']].dropna()

# Split
train_df, val_df = train_test_split(df, test_size=0.01, random_state=42)

# Convert to HF Datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["description"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True, num_proc=20)
val_dataset = val_dataset.map(tokenize, batched=True)

def add_labels(example):
    example["labels"] = torch.tensor([example["lat"], example["lng"]], dtype=torch.float)
    return example

train_dataset = train_dataset.map(add_labels, num_proc=20)
val_dataset = val_dataset.map(add_labels)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


class BoundedLatLonRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        latlon = self.mlp(x)
        lat = latlon[:, 0] * 90     # [-1, 1] → [-90, 90]
        lon = latlon[:, 1] * 180    # [-1, 1] → [-180, 180]
        return torch.stack([lat, lon], dim=1)

class BertForLatLonCosineLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = BoundedLatLonRegressor(config.hidden_size)
        self.init_weights()

    def latlon_to_xyz(self, latlon):
        lat_rad = torch.deg2rad(latlon[:, 0])
        lon_rad = torch.deg2rad(latlon[:, 1])
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return torch.stack([x, y, z], dim=1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        latlon_pred = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            pred_xyz = self.latlon_to_xyz(latlon_pred)
            true_xyz = self.latlon_to_xyz(labels)
            cosine_loss = 1 - (pred_xyz * true_xyz).sum(dim=1)
            loss = cosine_loss.mean()

        return {"loss": loss, "logits": latlon_pred}


training_args = TrainingArguments(
    output_dir="./latlon_model_v5",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps = 1000,
    num_train_epochs=5,
    weight_decay=0.0001,
    logging_dir="./logs/bert_v4",
    logging_steps=50,
    report_to=["tensorboard"],
)

model = BertForLatLonCosineLoss.from_pretrained("bert-base-uncased")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
