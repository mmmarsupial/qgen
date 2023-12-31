# External imports
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import nltk

# Internal imports
from prepare_data import DataPrep
from metric import CalculateMetric

# Variables
model_name = "t5-small"
paths = {"train": "./claim_datasets/train_short.csv",
         "validation": "./claim_datasets/valid_short.csv",
         "test": "./claim_datasets/test_short.csv"
         }

# Prepare data
data_prep = DataPrep(model_name)
train_data = data_prep.prepare_data(paths["train"])
validation_data = data_prep.prepare_data(paths["validation"])

# Model, tokenizer, metric and collator initiation
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
metric = CalculateMetric(model_name, metric="rouge")
data_collator = DataCollatorForSeq2Seq(tokenizer)

# Training constants
max_length_in = 200
max_length_out = 100
output_dir = "./checkpoints/" + model_name + "-fine-tuned"
learning_rate = 0.001
batch_size = 32

# Training arguments
args = Seq2SeqTrainingArguments(
    learning_rate=learning_rate,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=10,
    predict_with_generate=True,
    #fp16=True,
    load_best_model_at_end=True,
    output_dir=output_dir
)

# Trainer 
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=metric.compute_metrics()
)

# Train and save model
trainer.train()
trainer.save_model("./checkpoints/" + model_name + "-fine-tuned")



