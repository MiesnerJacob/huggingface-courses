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



## Creating your own dataset



## Semantic search with FAISS



## 🤗 Datasets, check!

