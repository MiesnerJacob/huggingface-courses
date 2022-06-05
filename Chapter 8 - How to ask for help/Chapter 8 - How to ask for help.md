# How to ask for Help

## Introduction

This section will cover what to do when you run into issues while developing with HF Transformers and how to ask for help. This section will also go over how to create issues when you identify bugs in an HF code repo.

None of the suggestions here are specific to HF but can be applied across almost any open source project.

## What to do when you get an error

In order to create some errors we will first copy an example model repo to our HF Hub account:

```bas
!huggingface-cli login
```

```python
from distutils.dir_util import copy_tree
from huggingface_hub import Repository, snapshot_download, create_repo, get_full_repo_name


def copy_repository_template():
    # Clone the repo and extract the local path
    template_repo_id = "lewtun/distilbert-base-uncased-finetuned-squad-d5716d28"
    commit_hash = "be3eaffc28669d7932492681cd5f3e8905e358b4"
    template_repo_dir = snapshot_download(template_repo_id, revision=commit_hash)
    # Create an empty repo on the Hub
    model_name = template_repo_id.split("/")[1]
    create_repo(model_name, exist_ok=True)
    # Clone the empty repo
    new_repo_id = get_full_repo_name(model_name)
    new_repo_dir = model_name
    repo = Repository(local_dir=new_repo_dir, clone_from=new_repo_id)
    # Copy files
    copy_tree(template_repo_dir, new_repo_dir)
    # Push to Hub
    repo.push_to_hub()
    
copy_repository_template()
```



### Debugging the pipeline from 🤗 Transformers

Let's try to load a model via pipelines:

```python
from transformers import pipeline

model_checkpoint = get_full_repo_name("distillbert-base-uncased-finetuned-squad-d5716d28")
reader = pipeline("question-answering", model=model_checkpoint)

"""
OSError: Can't load config for 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28'. Make sure that:

- 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28' is a correct model identifier listed on 'https://huggingface.co/models'

- or 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28' is the correct path to a directory containing a config.json file
"""
```

You can see the error above gives us some clues about what may be wrong. The first order of business when encountering an error is to read the traceback in full to understand the genesis of the error and what is causing it. 

So first to solve this error we look at the first potential problem, that we have the wrong model id. We look in HF Hub and notice we accidentally have 2 Ls in distilbert when there should only be one lol.

So naturally we fix the spelling and retry loading the model into the QA pipeline:

```python
model_checkpoint = get_full_repo_name("distilbert-base-uncased-finetuned-squad-d5716d28")
reader = pipeline("question-answering", model=model_checkpoint)

"""
OSError: Can't load config for 'lewtun/distilbert-base-uncased-finetuned-squad-d5716d28'. Make sure that:

- 'lewtun/distilbert-base-uncased-finetuned-squad-d5716d28' is a correct model identifier listed on 'https://huggingface.co/models'

- or 'lewtun/distilbert-base-uncased-finetuned-squad-d5716d28' is the correct path to a directory containing a config.json file
"""
```

Another error! Don't throw your laptop out the window we can solve this. Because we already solved the first potential problem let's look at the second one. 

Let's print out the files in our repo to see if there is a config file:

```python
from huggingface_hub import list_repo_files

list_repo_files(repo_id=model_checkpoint)

['.gitattributes', 'README.md', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json', 'training_args.bin', 'vocab.txt']
```

There is no config file! We have two options, first we could ask our colleague to upload the missing config file in their model repo, or we could just take the config file from the pre-trained version of this model.

Let's just take the pre-trained config and upload it ourselves:

```python
from transformers import AutoConfig

pretrained_checkpoint = "distilbert-base-uncased"
config = AutoConfig.from_pretrained(pretrained_checkpoint)

config.push_to_hub(model_checkpoint, commit_message="Add config.json")
```

Now let's try using the model again:

```python
reader = pipeline("question-answering", model=model_checkpoint, revision="main")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text
given a question. An example of a question answering dataset is the SQuAD
dataset, which is entirely based on that task. If you would like to fine-tune a
model on a SQuAD task, you may leverage the
examples/pytorch/question-answering/run_squad.py script.

🤗 Transformers is interoperable with the PyTorch, TensorFlow, and JAX
frameworks, so you can use your favourite tools for a wide variety of tasks!
"""

question = "What is extractive question answering?"
reader(question=question, context=context)

{'score': 0.38669535517692566,
 'start': 34,
 'end': 95,
 'answer': 'the task of extracting an answer from a text given a question'}
```

Aha, it worked!!! Make sure to carefully read your traceback errors and look up the chain if you need more context around the final error!

### Debugging the forward pass of your model

Although pipelines are great for standing up a model for prediction quickly, sometimes we may want more control over the framework. For example, we may have a use case where we want to access our model logits. 

Let's try to access our output logits on a QA task:

```python
import torch

inputs = tokenizer(question, context, add_special_tokens=True)
input_ids = inputs["input_ids"][0]
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits
# Get the most likely beginning of answer with the argmax of the score
answer_start = torch.argmax(answer_start_scores)
# Get the most likely end of answer with the argmax of the score
answer_end = torch.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {question}")
print(f"Answer: {answer}")

"""
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/var/folders/28/k4cy5q7s2hs92xq7_h89_vgm0000gn/T/ipykernel_75743/2725838073.py in <module>
      1 inputs = tokenizer(question, text, add_special_tokens=True)
      2 input_ids = inputs["input_ids"]
----> 3 outputs = model(**inputs)
      4 answer_start_scores = outputs.start_logits
      5 answer_end_scores = outputs.end_logits

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, start_positions, end_positions, output_attentions, output_hidden_states, return_dict)
    723         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    724
--> 725         distilbert_output = self.distilbert(
    726             input_ids=input_ids,
    727             attention_mask=attention_mask,

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/miniconda3/envs/huggingface/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    471             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    472         elif input_ids is not None:
--> 473             input_shape = input_ids.size()
    474         elif inputs_embeds is not None:
    475             input_shape = inputs_embeds.size()[:-1]

AttributeError: 'list' object has no attribute 'size'
"""
```

Oh no we are getting an error! If we look at the traceback the error is coming from trying to use the size method on a python list. It is likely that we are passing the wrong data type to the method using the size method. If we change our tokenizer to return PyTorch tensors it should fix our problem:

```python
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"][0]
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits
# Get the most likely beginning of answer with the argmax of the score
answer_start = torch.argmax(answer_start_scores)
# Get the most likely end of answer with the argmax of the score
answer_end = torch.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)
print(f"Question: {question}")
print(f"Answer: {answer}")

"""
Question: Which frameworks can I use?
Answer: pytorch, tensorflow, and jax
"""
```

It worked! When running into issues like this it is always great to look at StackOverflow or the HF forums and see if anyone has run into the same problem.



## Asking for help on the forums

The huggingface forums can be accessed here: https://discuss.huggingface.co/
The forums are broken down into categories such as beginner, intermediate, research, courses, etc. A new topic can be created by clicking the +topic button. When writing an issue make sure:

- Your title is descriptive of the issue

- Your body has sufficient background to reproduce an error if needed

- You tag the issue with appropriate labels

- Use \``` to paste code snippets

- Provide entire traceback

- Provide code to reproduce the error

  

## Debugging the training pipeline

There is a lot that happens within the trainer.train() method.  It converts datasets to dataloaders, so the problem could be something wrong in your dataset, or some issue when trying to batch elements of the datasets together. Then it takes a batch of data and feeds it to the model, so the problem could be in the model code. After that, it computes the gradients and performs the optimization step, so the problem could also be in your optimizer. And even if everything goes well for training, something could still go wrong during the evaluation if there is a problem with your metric.

The best course of action is to go through each step manually and see where things have gone wrong. Here is our example code and error below:

```python
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = load_metric("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["validation_matched"],
    compute_metrics=compute_metrics,
)
trainer.train()

'ValueError: You have to specify either input_ids or inputs_embeds'
```



### Check your data

So lets pull up an example of our data and see whats up with the input_ids field:

```python
trainer.train_dataset[0]
{'hypothesis': 'Product and geography are what make cream skimming work. ',
 'idx': 0,
 'label': 1,
 'premise': 'Conceptually cream skimming has two basic dimensions - product and geography.'}
```

It's not there! Well if we look at our old code we accidentally passed on the raw data to our trainer and not the tokenized datasets!!! Let's adjust that:

```python
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = load_metric("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
)
trainer.train()

'ValueError: expected sequence of length 43 at dim 1 (got 37)'
```

We get another error, this time it has to do with the data collation (we can identify this by looking at the traceback in full.

Next, we decode our input Ids just to do a sanity check:

```python
tokenizer.decode(trainer.train_dataset[0]["input_ids"])
'[CLS] conceptually cream skimming has two basic dimensions - product and geography. [SEP] product and geography are what make cream skimming work. [SEP]'
```

Yup looks good there. Next lets check our mask ids:

```python
trainer.train_dataset[0]["attention_mask"]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

len(trainer.train_dataset[0]["attention_mask"]) == len(
    trainer.train_dataset[0]["input_ids"]
)
True
```

Looks like the attention masks are all turned on as we used no padding, so goof there. Also our attention masks match the length of our input ids.

Lastly, let check our labels:

```python
# Check one example
trainer.train_dataset[0]["label"]
1

# Check all unique values
trainer.train_dataset.features["label"].names
['entailment', 'neutral', 'contradiction']
```

Distilbert doesn't use token type ids but if you were using a model that utilizes that input it should be checked as well. 

In this sub-section, we checked the training data, but in practice, it is good to do these checks on your validation and test sets as well.

### From datasets to dataloaders

From previous steps we are left with this traceback:

```python
~/git/transformers/src/transformers/data/data_collator.py in torch_default_data_collator(features)
    105                 batch[k] = torch.stack([f[k] for f in features])
    106             else:
--> 107                 batch[k] = torch.tensor([f[k] for f in features])
    108 
    109     return batch

ValueError: expected sequence of length 45 at dim 1 (got 76)
```

We can see that the error is coming from our data collator and has to do with the input size of the sequence. Let's look at what collator our model is using:

```python
data_collator = trainer.get_train_dataloader().collate_fn
data_collator

<function transformers.data.data_collator.default_data_collator(features: List[InputDataClass], return_tensors='pt') -> Dict[str, Any]>
```

It looks like the model is using a default collator. This collator does not pad examples, since we have examples of different lengths this is the source of our error. So let's go back to our training script and define our data collator with padding and pass that to our Trainer definition:

```python
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = load_metric("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```

This has solved the error, but we have another one now lol:

```python
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

We will look at how to solve this CUDA error below.

### Going through the model

Something important to note is that CUDA error break your kernel so you will need to restart it to continue debugging.

CUDA errors are notoriously difficult to debug. So how do we debug those errors? The answer is easy: we don’t. Unless your CUDA error is an out-of-memory error (which means there is not enough memory in your GPU), you should always go back to the CPU to debug it.

So to continue debugging our model we will switch to cpu and pass a batch to the model for inference:

```python
outputs = trainer.model.cpu()(**batch)

~/.pyenv/versions/3.7.9/envs/base/lib/python3.7/site-packages/torch/nn/functional.py in nll_loss(input, target, weight, size_average, ignore_index, reduce, reduction)
   2386         )
   2387     if dim == 2:
-> 2388         ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
   2389     elif dim == 4:
   2390         ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)

IndexError: Target 2 is out of bounds.
```

So it looks like the error is coming from the loss computation. It is receiving a target value of 2 which is out of bounds. Since we are performing an NLI task we expect 3 different target variables. To fix this we must define the number of expected labels when instantiating our model.

```python
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = load_metric("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

At this point, we can rerun all our batches through the model on both CPU and GPU to confirm everything is working fine.

### Performing one optimization step

Let's try computing some gradients and performing an optimization step:

```python
loss = outputs.loss
loss.backward()
```

Its rare to get an error here but if you do switch to CPU and try again for a more clear error message. Now let's perform the optimization step:

```python
trainer.create_optimizer()
trainer.optimizer.step()
```

If you use a default optimizer you likely won't run into problems here, if you use a custom optimizer it is more likely. If you run into an error here switch to CPU and run again for a more clear error message.

### Dealing with CUDA OOM errors

Out of memory errors occur when there is too much data being stored on the GPU hardware being used, it doesn't have to do with an error in your code. When you run into an OOM error it is good to make sure you only have 1 model on the GPU and are not storing any unnecessary data there. If your problems persist you can try decreasing your batch sizes to manage the amount of data on the hardware at any given point. If the OOM errors still persist, you can try using a smaller model.

### Evaluating the model

After fixing all the rrors above we are left with another on the training execution:

```
trainer.train()

TypeError: only size-1 arrays can be converted to Python scalars
```

After investigation of the traceback you will notice this is happening during evaluation. We can see this here:

```python
~/git/datasets/src/datasets/metric.py in add_batch(self, predictions, references)
    431         """
    432         batch = {"predictions": predictions, "references": references}
--> 433         batch = self.info.features.encode_batch(batch)
    434         if self.writer is None:
    435             self._init_writer()
```

This is happening in metric.py which is what our compute metric function utilizes. Let's try to use that method directly on CPU:

```python
predictions = outputs.logits.cpu().numpy()
labels = batch["labels"].cpu().numpy()

compute_metrics((predictions, labels))
TypeError: only size-1 arrays can be converted to Python scalars
```

We get the same error! Let's look at the inputs:

```python
print(predictions.shape, labels.shape)

'((8, 3), (8,))'
```

It looks like our preds have 3 cols, this is because they are still logits and haven't been converted to predictions!!!

To fix this we can add an argmax to our compute metrics function:

```python
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# Fixed!
compute_metrics((predictions, labels))
{'accuracy': 0.625}
```

We have fixed all the issues with our training script, for reference here is what it looks like:

```python
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

args = TrainingArguments(
    f"distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = load_metric("glue", "mnli")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```

Although we have fixed all the errors in our code, sometimes a model may not perform well at all, in the next section we will go over techniques to stem this issue.

## Debugging silent errors during training

Getting bad results from a model with no errors is one of the hardest things to debug in machine learning, it requires deep analysis and tehcnical understanding of the components at play.

### Check your data (again!)

Often times a model giving bad results is due to the input data not representing what we think it is. Some critical questions to ask yourself include:

- Is the decoded data understandable?
- Do you agree with the labels?
- Is there one label that’s more common than the others?
- What should the loss/metric be if the model predicted a random answer/always the same answer?

Make sure your data is represented in the fashion you want, your labels line up, and most importantly your data is representative of the population you will be running inference on.

### Overfit your model on one batch

When you are sure your data is not the issue you can see if your model is capable of learning from it with a simple test.

Usually, you want to avoid overfitting your model, but overfitting on a single batch can let you know if your model is able to learn from the data and actually sees improvement. This helps you check if the model you are using can perform the task you are asking it to do.

```python
# Model Training loop on single batch 20 epochs
for batch in trainer.get_train_dataloader():
    break

batch = {k: v.to(device) for k, v in batch.items()}
trainer.create_optimizer()

for _ in range(20):
    outputs = trainer.model(**batch)
    loss = outputs.loss
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
   
# Eval
with torch.no_grad():
    outputs = trainer.model(**batch)
preds = outputs.logits
labels = batch["labels"]

compute_metrics((preds.cpu().numpy(), labels.cpu().numpy()))
{'accuracy': 1.0}
```

As we can see above the model is almost perfectly fit to our training examples. If your model does not get almost perfect results on the train set in a test like this then it likely means you are posing the task or the data incorrectly.

### Don't tune anything until you have a first baseline

Hyperparameter tuning has a reputation for being one of the most difficult parts of a machine learning project. In reality, the default parameters defined by the HF trainer object will likely give you good results. Use the results of that default training as a baseline before testing all combinations of hyperparams. Don't compare thousands of different options, but rather tweak a few parameters and see what has the biggest impact.

If you are planning to tweak the model itself make sure to keep it simple and limited to changes you can reasonably justify.

### Ask for help

As always you can ask questions in the forums. A couple of good additional resources for help are:

Here are some additional resources that may prove helpful:

- [“Reproducibility as a vehicle for engineering best practices”](https://docs.google.com/presentation/d/1yHLPvPhUs2KGI5ZWo0sU-PKU3GimAk3iTsI38Z-B5Gw/edit#slide=id.p) by Joel Grus
- [“Checklist for debugging neural networks”](https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21) by Cecelia Shao
- [“How to unit test machine learning code”](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) by Chase Roberts
- [“A Recipe for Training Neural Networks”](http://karpathy.github.io/2019/04/25/recipe/) by Andrej Karpathy



## How to write a good issue

When you encounter a bug in the HF codebase it's best practice to create an issue associated with the bug, as with any open-source code.

### Create a minimum reproducible error

Make sure you have a readable version of the code that can reproduce the error. Accessibility to reproduction is absolutely critical when looking for help./

### Filling out the issue template

When filling out an issue template some best practices include:

- Providing your environment details

  ```python
  transformers-cli env
  
  '''Copy-and-paste the text below in your GitHub issue and FILL OUT the two last points.'''
  
  - `transformers` version: 4.12.0.dev0
  - Platform: Linux-5.10.61-1-MANJARO-x86_64-with-arch-Manjaro-Linux
  - Python version: 3.7.9
  - PyTorch version (GPU?): 1.8.1+cu111 (True)
  - Tensorflow version (GPU?): 2.5.0 (True)
  - Flax version (CPU?/GPU?/TPU?): 0.3.4 (cpu)
  - Jax version: 0.2.13
  - JaxLib version: 0.1.65
  - Using GPU in script?: <fill in>
  - Using distributed or parallel set-up in script?: <fill in>
  ```

- Tagging relevant stakeholders can be helpful

- Paste a reproducible example using ```python to wrap your code

- Always include your full traceback error

- If you can't provide a minimum reproducible error explain what you have done in clearly defined steps

- Describe the expected behavior you are looking for!!!

## Part 2 completed!

