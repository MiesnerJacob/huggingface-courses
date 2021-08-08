# Chapter 3 - Fine-tuning a pretrained model

## Introduction

This chapter covers:

1. How to prepare a large dataset from the Huggingface Hub
2. How to use the Trainer API to fine-tune a model
3. How to use a custom training loop
4. How to leverage the Huggingface Accelerate library to easily run a custom training loop on any distributed setup

## Processing the data

**Load Dataset from the Hub**

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

**Preprocessing a dataset**

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

**Dynamic Padding**

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True, collate_fn=data_collator)
```



## Fine-tuning a model with the Trainer API

🤗 Transformers provides a `Trainer` class to help you fine-tune any of the pretrained models it provides on your dataset. 

The first step before we can define our `Trainer` is to define a `TrainingArguments` class that will contain all the hyperparameters the `Trainer` will use for training and evaluation. 

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

The second step is to define our model and what head to use in conjunction with that model.

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Next we instantiate the trainer class and call the train function! (Compute metrics lets the trainer know which metric to calc during eval of your validation set.)

```python
from transformers import Trainer

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```

Finally once the model is trained we can use it to predict, and use argmax to find our class predictions.

```
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=-1)
```



## A full training

### Prep data

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Postprocessing tokeinzed dataset for training
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Define dataloaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Define model 
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

#Instantiate optimizer
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Instantiate learning rate scheduler (optional)
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Make sure we are using GPU
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop!
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
# Eval loop!
from datasets import load_metric

metric= load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

### Supercharge your training loop with 🤗 Accelerate (Training on multiple GPUs or TPUs)

In order to train on multiple GPUs or TPUs we need to make only a few adjustments to our code above:

```python
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training loop!
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
# Eval loop!
from datasets import load_metric

metric= load_metric("glue", "mrpc")
model.eval()

eval_dataloader = accelerate.prepare(eval_dataloader)
for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=accelerate.gather(predictions),
                     references=accelerate.gather(batch["labels"]))

metric.compute()
```

⚠️ In order to benefit from the speed-up offered by Cloud TPUs, Huggingface recommends padding your samples to a fixed length with the `padding="max_length"` and `max_length` arguments of the tokenizer.

If you want to try this in a Notebook (for instance, to test it with TPUs on Colab), just paste the code in a `training_function` and run a last cell with:

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```



## Fine-tuning. Check!

To recap, this chapter covered:

- Datasets in the [Hub](https://huggingface.co/datasets)
- How to load and preprocess datasets, including using dynamic padding and collators
- Implement your own fine-tuning and evaluation of a model
- Implement a lower-level training loop
- Use 🤗 Accelerate to easily adapt your training loop so it works for multiple GPUs or TPUs

