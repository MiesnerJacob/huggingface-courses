# Chapter 2 - Using 🤗 Transformers



## Introduction

In [Chapter 1](https://huggingface.co/course/chapter1), we used Transformer models for different tasks using the high-level `pipeline` API. Although this API is powerful and convenient, it’s important to understand how it works under the hood so we have the flexibility to solve other problems.

In this chapter, you will learn:

- How to use tokenizers and models to replicate the `pipeline` API’s behavior
- How to load and save models and tokenizers
- Different tokenization approaches, such as word-based, character-based, and subword-based
- How to handle multiple sentences of varying lengths

## Behind the Pipeline

Steps to reproduct pipeline API behavior:

1. Load tokenizer using AutoTokenizer and pass raw inputs (include padding & truncation, also specify return tensor type (Pytorch, Tensorflow, Numpy))
   1. tokenized text will contain input ids and attention masks for each sequence in the raw input
2. Load Model and pass tokeized inputs to it (output = model(**inputs))
   1. Output if passed to model loaded with AutoModel will be hidden state with shape 0="batch_size", 1="sequence_length", 2="hidden_size"
   2. Output if passed to task specific AutoModel (i.e. AutoModelForSequenceClassification) will include model head and will return logits
3. Convert logit outputs from your model to probabilities (using softmax in this case)
4. Map probabilities to labels (model.config,id2label)

## Models

There are multiple model classes but using AutoModel works seamlessly across architecture types

There are a few different ways to load a model:

1. Load config into model class (only instantiates architecture & randomly initializes weights)
2. Load config and weights using .from_pretrained()
   1. from_pretrained` method won’t re-download them, it will be in the cache folder, which defaults to *~/.cache/huggingface/transformers*

Saving a model is just as easy:

1. model.save_pretrained("directory_on_my_computer")
2. Two main files will always be saved:
   1. config.json (attributes necessary to build the model architecture)
   2. *pytorch_model.bin* (known as the *state dictionary*; it contains all your model’s weights)

While the model accepts a lot of different arguments, only the input IDs are necessary. We’ll get into what the other arguments do and when they are required later.

## Tokenizers

Tokenizers are one of the core components of the NLP pipeline. They serve one purpose: to translate text into data that can be processed by the model. Models can only process numbers, so tokenizers need to convert our text inputs to numerical data. A vocabulary is defined by the total number of independent tokens that we have in our corpus. Each word gets assigned an ID, starting from 0 and going up to the size of the vocabulary. The model uses these IDs to identify each word. 

Tokenization steps:

1. Split text into tokens
2. Produce Input IDs (with other inputs i.e. token types, attention masks)
3. Add special token IDs
4. Decode tokens returned by model

Tokenization methods include but are not limited to:

1. Word-based tokenizers![word_based_tokenization](https://huggingface.co/course/static/chapter2/word_based_tokenization.png)
   1. Produce large vocabularies ("dog" & "dogs" will be independent tokens, model interprets them separately)
   2. Can produce a decent amount of unknown "[UNK]" out of vocab tokens. One way to reduce the amount of unknown tokens is to go one level deeper, using a *character-based* tokenizer.
2. Character-based tokenizers![character_based_tokenization](https://huggingface.co/course/static/chapter2/character_based_tokenization.png)
   1. The vocabulary is much smaller. (Only so many possible chars)
   2. There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.
   3. Produces many more tokens for model to evaluate! Because each word is being split into chars
   4. Tokens can carry less meaning because one char doesn't hold as much info as a word. (This can very across languages in Chinese, for example, each character carries more information than a character in a Latin language.)
3. Subword-based tokenizers![bpe_subword](https://huggingface.co/course/static/chapter2/bpe_subword.png)
   1. This allows us to have relatively good coverage with small vocabularies, and close to no unknown tokens.
   2. Happy medium between word-based and char-based tokenization
   3. This approach is especially useful in agglutinative languages such as Turkish, where you can form (almost) arbitrarily long complex words by stringing together subwords.
4. Unsurprisingly, there are many more techniques out there. To name a few:

- Byte-level BPE, as used in GPT-2
- WordPiece, as used in BERT
- SentencePiece or Unigram, as used in several multilingual models

## Handling multiple sequences

- Models expect a batch of inputs so you must pass tensors as a list of lists. If you havre a single input you should add a dimension to the tensor of input ids before passing to the model to avoid it failing
- When passing multiple sequences the inputs must be padded to be of equal length
- Attention masks are boolean indicators [0,1] that let the model know which tokens to ingore in attention layers (like padding tokens)
- If you dont adjust attention masks then you will receive a differenty output for your sentence with and without padding (set padding=True in tokenizer and this will be done automatically)
- With Transformer models, there is a limit to the lengths of the sequences we can pass the models. Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences. There are two solutions to this problem:
  - Use a model with a longer supported sequence length.
  - Truncate your sequences.

## Putting it all together

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Basic usage completed!

Great job following the course up to here! To recap, in this chapter you:

- Learned the basic building blocks of a Transformer model.
- Learned what makes up a tokenization pipeline.
- Saw how to use a Transformer model in practice.
- Learned how to leverage a tokenizer to convert text to tensors that are understandable by the model.
- Set up a tokenizer and a model together to get from text to predictions.
- Learned the limitations of input IDs, and learned about attention masks.
- Played around with versatile and configurable tokenizer methods.