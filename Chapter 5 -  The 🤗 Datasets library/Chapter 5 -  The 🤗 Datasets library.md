# Chapter 5 - The 🤗 Datasets library

## Introduction

Chapter 3 went over the basics of the dataset library including:

- Loading a dataset from the Huggingface Hub
- Process the data with Dataset.map()
- Load and compute metrics

This chapter will give more details into the datasets library that willl prep us for advanced tokenization and fine tuning in Chapter 6 and Chapter 7!

## What if my dataset isn't on the Hub?

This library allows importing data from many different file types including:

```python
# CSV
csv_dataset = load_dataset('csv', data_files='my_file.csv')

# Text Files
text_dataset = load_dataset('txt', data_files='my_file.txt')

# JSON
json_dataset = load_dataset('json', data_files='my_file.json')

# Parquet
parquet_dataset = load_dataset('parquet', data_files='my_file.parquet')

# Important to note you can provide a single path, list of file paths, or a dict that splits in your data_files argument for train/test etc. like below (you can pass compressed files in any of these formats as well):
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}

## You can use sep=':' arguement for splitting csv data
## You can provide a remote url to this method to load data!
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```



## Time to slice and dice

### Shuffle and Split

```python
# Load Dataset
from datasets import load_dataset

squad = load_dataset("squad", split="train")

# Shuffle
squad_shuffled = squad.shuffle(seed=0)

#Split
dataset = squad_shuffled.train_test_split(test_size=0.2)
```



### Select and filter

```python
# Selecting specific indicies
indicies = [0.10.20.40.80]
examples = sqwuad.select(indicies)

# Selecting random examples
sample = squad.shuffle().select(range(10))

# Selecting examples using filtering
squad_filtered = squad.filter(lambda x: x['title'].startswith("M"))

# Selecting non-null examples using filtering
squad_no_null = squad.filter(lambda x: x["title"] is not None)
```



### Rename, remove, and flatten

```python
# Rename columns
squad.rename_column("context", "passages")

# Remove columns
squad.remove_columns(["id","title"])

# Flatten dataset with nested features
squad.flatten()
```



### Map

```python
# Use map to apply function to each example in dataset
def lowercase_title(example):
  return {"title": example["title"].lower()}

squad_lowercase = squad.map(lowercase_title)

# Feed batches of rows to map method (much faster for tokenization)
squad_lowercase = squad.map(lowercase_title, batched=True, batch_size=500)

# Create a new column using map
def compute_title_length(example):
    return {"title_length": len(example["title"].split())}
  
squad = squad.map(compute_title_length)

# Tokenizing using fast tokenization
fast_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)


def fast_tokenize_function(examples):
    return fast_tokenizer(examples["title"], truncation=True)


tokenized_dataset = squad.map(fast_tokenize_function, batched=True, num_proc=8)
```

### From Datasets to DataFrames and back

```python
# Changing dataset from datasets to Pandas DataFrame
## Method 1
dataset.set_format("pandas")
df = dataset[:]

## Method 2
df = dataset.to_pandas()

df.head()

# Dont forget to rest format after so we can tokenize!!!
## Reset back to Arrow Format
dataset.reset_format()
```

### Creating a validation split

```python
# Here we split our training data to get an additional validation split
squad_clean = squad["train"].train_test_split(train_size=0.8, seed=42)

# Rename the default "test" split to "validation"
squad_clean["validation"] = squad_clean.pop("test")

# Add the "test" set to our `DatasetDict`
squad_clean["test"] = squad["test"]
```



### Saving a dataset

```python
# Saving datasets of different types
## Arrow and Parquet are good for big data
## Arrow for data used a lot, Parquet better for long term optimized storage

# Arrow
Dataset.save_to_disk("path")

# CSV
raw_datasets = load_dataset("allocine")

for split, dataset in raw_datasets.items():
  dataset.to_csv(f"my-dataset-{split}.csv", index=None)

# JSON
for split, dataset in raw_datasets.items():
  dataset.to_json(f"my-dataset-{split}.jsonl", index=None)

# Parquet
for split, dataset in raw_datasets.items():
  dataset.to_parquet(f"my-dataset-{split}.parquet", index=None)
```

```python
# Loading saved datasets with splits
## Same snnytax for all file formats
data_files = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)
```



## Big data? 🤗 Datasets to the rescue!

It is hard to work with gigantic datasets on your local machine if you try to load it all into ram at 1 time. This is where Datasets comes in handy, it allows you to stream data so that you can look at a slidingh window of the dataset and not load more than you need!

### Loading Pile Dataset

THe pile is a multi-domain dataset which has 14gb train splits that contains 815GB data in total!

```python
from datasets import load_dataset

# This takes a few minutes to run, so go grab a tea or coffee while you wait :)
data_files = "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
pubmed_dataset
```

Check how much ram this taskes:

```python
import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
```

```python
RAM used: 5678.33 MB
```

Streaming datasets

```python
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
```

Iterate through the dataset

```python
next(iter(pubmed_dataset_streamed))
```

Tokenize streamed data

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
next(iter(tokenized_dataset))
```

Shuffle a batch of data

```python
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
next(iter(shuffled_dataset))
```

```python
# Skip the first 1,000 examples and include the rest in the training set
train_dataset = shuffled_dataset.skip(1000)
# Take the first 1,000 examples for the validation set
validation_dataset = shuffled_dataset.take(1000)
```

Ciombine two datasets to iterate over in tandem

```python
from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
list(islice(combined_dataset, 2))
```

Loading all datasets in the pile via streaming:

```python
base_url = "https://mystic.the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
next(iter(pile_dataset["train"]))
```



## Creating your own dataset

Get Dataset from HF Hub

```python
from huggingface_hub import list_datasets

all_datasets = list_datasets()
print(f"Number of datasets on Hub: {len(all_datasets)}")
print(all_datasets[0])
```

Login from notebook to HF Hub

```py
from huggingface_hub import notebook_login

notebook_login()
```

Create Repo an upload data

```python
from huggingface_hub import create_repo

repo_url = create_repo(name="github-issues", repo_type="dataset")
repo_url

from huggingface_hub import Repository

repo = Repository(local_dir="github-issues", clone_from=repo_url)
!cp datasets-issues-with-comments.jsonl github-issues/

repo.lfs_track("*.jsonl")
repo.push_to_hub()

# Load Dataset
remote_dataset = load_dataset("miesnerjacob/github-issues", split="train")
remote_dataset
```



## Semantic search with FAISS

Method to get embeddings via CLS pooling and sentence transformers model

```python
# Turn text into embeddings
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
```

Get FAISS index using HF method!

```python
# Add FAISS index to dataset
embeddings_dataset.add_faiss_index(column="embeddings")
```

Encode question we want to answer

```python
# Encode question
question = "How can I a dataset to a pandas DataFrame?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape
```

Find answers

```python
# Lookup on FIASS index for most similar examples
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
```



## 🤗 Datasets, check!

This chapter covered the following:

- Load datasets from anywhere, be it the Hugging Face Hub, your laptop, or a remote server at your company.
- Wrangle your data using a mix of the `Dataset.map()` and `Dataset.filter()` functions.
- Quickly switch between data formats like Pandas and NumPy using `Dataset.set_format()`.
- Create your very own dataset and push it to the Hugging Face Hub.
- Embed your documents using a Transformer model and build a semantic search engine using FAISS.
