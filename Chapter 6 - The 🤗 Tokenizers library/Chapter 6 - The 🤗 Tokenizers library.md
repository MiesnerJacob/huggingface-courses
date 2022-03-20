

# Chapter 6 - The 🤗 Tokenizers library

## Introduction

This chapter will go over how to train a tokenizer from scratch on a text corpus. It will cover:

* How to train a tokenizer similar to ones given by a pre-trained checkpoint on a new text corpus
* The special features of Fast Tokenizers
* The differences between the three main sub-word tokenizer algos used in NLP today
* How to build a tokenizer from scratch

## Training a new tokenizer from an old one

Training tokenizers is deterministic unlike model training which requires a random seed to produce the same results.

A tokenizer wil NOT BE suitable if it is trained on data disimilar to the training data for the model. for example:

- Difference leanugagte
- New chars
- New domain
- New Style

To train tokenizer:

1. Gather text corpus
2. Choose architecture
3. train
4. save!

### Assembling a Corpus

First we must load our data

```python
# Load some traning data
from datasets import load_daatset

raw_datasets = load_datraset('code_search_net','python')

```

Next we create an iterator out of the dataset so we can train in batches

```python
# Method 1

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

# Method 2

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
        
# Run func
training_corpus = get_training_corpus()
```

Next we will look at training an existing tokenizer, essentially fine-tuning

```python
# Import tokenizer
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Train using iterator created above
# This only works with fast tokenizers (written in rust in Tokenizers library)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```

Saving your tokenizer

```python
# save to local

tokenizer.pretrained('my-tokeinzer')

# Save to HF Hub after log in

tokenizer.push_to_hub('my-tokenizer')
```



## Fast tokenizers' special powers

Slow tokenizers are provided by the vanilla transformers library, while fast tokenizers are provided by HF tokenizers library (written in Rust). The difference in performance of these will become more apparent the bigger your dataset is. 

Faswt tokenizers are automatically used when loading a tokenizer if it is available, although you can toggle between them using the "use_fast" parameter in the .from_pretrained() method.

The incremental benefit of fast tokenizers is primarily in its ability to batch process in parellel due to its Rust backing.

### Batch Encoding

The output of a tokenizer isn’t a simple Python dictionary; what we get is actually a special `BatchEncoding` object. This object comes with some additional attributes and methods outline below

```python
# First lets load a tokenizer and encode some text
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)

# Lets look at the output object type
print(type(encoding))

<class 'transformers.tokenization_utils_base.BatchEncoding'>

# See if encoding is was generated using fast tokenizer
encoding.is_fast

# print tokens
encoding.tokens()

# print word ids
# Keeps track of which words the tokens belong to
encoding.word_ids()

# Mappuing tokens to original text
start, end = encoding.word_to_chars(3)

print(example[start:end])

'Sylvain'
```



## Inside the `token-classification` pipeline

Token classification apply NER to tokens, but post-processing must be done to join the tokens together to get a clean output of entities and their classifications not just the tokens. This post-processing can be done using aggregation techniques withing the pipeline definition for a token-classification pipeline.

The beggining to end process for NER looks like:

Raw Text -> tokens -> Input_ids -> Model -> Logits -> softmax -> predictions -> map to config.id_2_label -> Mapped tokens -> group mapped tokens

Let's look at the code step by step starting with tokenization and modeling

```python
# Load tokenizer and model and get logit outputs
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

# Look at logit shapes
print(inputs["input_ids"].shape)
print(outputs.logits.shape)

# Here we see we have one piece of text mapped to 19 tokens and logits for the 9 labels
torch.Size([1, 19])
torch.Size([1, 19, 9])

```

Next we can apply softmax to our output logits to get predictions for each token and what label they correspond to

```python
# softmax and print output label predictions
import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)

[0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 0, 8, 0, 0]
```

Next lets take a look at what these labels mean

```python
# The B prefix is used to indicate a token is the first token for a certain prediction
# The I prefix is used to indicate the token lives inside an entity
# The O label (stands for "outside") says this token does not belong to an entity
# this is used to be able to aggregate the tokens together

print(model.config.id2label)

{0: 'O',
 1: 'B-MISC',
 2: 'I-MISC',
 3: 'B-PER',
 4: 'I-PER',
 5: 'B-ORG',
 6: 'I-ORG',
 7: 'B-LOC',
 8: 'I-LOC'}
```

Next we can take our outputs and format them to get ready for aggregation

```python
# This method uses offset mapping to allow us to see where the token lived in the original text
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

        
print(results)

[{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S', 'start': 11, 'end': 12},
 {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl', 'start': 12, 'end': 14},
 {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va', 'start': 14, 'end': 16},
 {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in', 'start': 16, 'end': 18},
 {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu', 'start': 33, 'end': 35},
 {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging', 'start': 35, 'end': 40},
 {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face', 'start': 41, 'end': 45},
 {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

Lastly we can skip the method above and just go straight to grouping the tokens into their common entities

```python
# Taking model output and grouping tokens into aggregated NER output
import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Grab all the tokens labeled with I-label
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # The score is the mean of all the scores of the tokens in that grouped entity
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)

[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
 {'entity_group': 'ORG', 'score': 0.97960204, 'word': 'Hugging Face', 'start': 33, 'end': 45},
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```



## Fast tokenizers in the QA pipeline

It its important to note that QA tokens are created to look like such: [CLS] question [SEP] context [SEP]

First lets look how to use the pipeline API fior question-answering:

```python
# Encode our question and answer and pass them to the pipeline API
from transformers import pipeline

question_answerer = pipeline("question-answering")
context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)

{'score': 0.97773,
 'start': 78,
 'end': 105,
 'answer': 'Jax, PyTorch and TensorFlow'}
```

Let's do the same thing as obive but this time using a specific model and not the pipeline

```python
# Here we tokenize our inputs and pass them to the model which gives us our output logits
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# The model output gives us logits for the start and end of the answer
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

torch.Size([1, 66]) torch.Size([1, 66])
```

Now lets generate our predictions

```python
#First we must mask the question tokens

import torch

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000

# Next we could just take softmax but we may wend up with an end token that is before the start token LOL
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

# First we calculate all possible products
scores = start_probabilities[:, None] * end_probabilities[None, :]

# Then we use the .trui function from torch to return the upper triangular part of the 2D tensor passed as an argument, so it will do that masking for us
scores = torch.triu(scores)

# Now we get the index of the maxes
max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])

# Then we get our indexes for the answer using the offset mapping
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_char]

# Lastly we format our results and print it out
result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)

{'answer': 'Jax, PyTorch and TensorFlow',
 'start': 78,
 'end': 105,
 'score': 0.97773}
```



### Handling long contexts

The question-answer pipeline has a max token length of 384, so we have to take a different approach when using large piece of text as context

```python
# We could just truncate our context but we lose information
inputs = tokenizer(question, long_context, max_length=384, truncation="only_second")
print(tokenizer.decode(inputs["input_ids"]))

# What we can do instead is split our text into small overlapping chunks so we alwyas retain the answer in its full form within one of the chunks
sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
    
'[CLS] This sentence is not [SEP]'
'[CLS] is not too long [SEP]'
'[CLS] too long but we [SEP]'
'[CLS] but we are going [SEP]'
'[CLS] are going to split [SEP]'
'[CLS] to split it anyway [SEP]'
'[CLS] it anyway. [SEP]'

# This feature shows us which which sentence these chunks are mapped to
# We only used one sentence
print(inputs["overflow_to_sample_mapping"])

[0, 0, 0, 0, 0, 0, 0]

# Heres what the output would look like with muttiple sentences
sentences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]
inputs = tokenizer(
    sentences, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

print(inputs["overflow_to_sample_mapping"])

[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

# Here is our tokenization method with all relevant parameters applied
inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

# This output will have a size of (# of examples, # ofg tokens)

_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")
print(inputs["input_ids"].shape)

# Next we pass them to our model
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

# Then mask tokens that are part of the question itself
sequence_ids = inputs.sequence_ids()
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
# Mask all the [PAD] tokens
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

start_logits[mask] = -10000
end_logits[mask] = -10000

# We then apply softmax to our logits
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

# And find tokens that are candidates for answers (making sure start is before end predicted)
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[0]
    end_idx = idx % scores.shape[0]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)

[(0, 18, 0.33867), (173, 184, 0.97149)]

# Finally we format our output
for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)
```



## Normalization and pre-tokenization

### Normalization

Normalization -> Pre-tokenization -> Tokenizaation Model -> Postprocessor (Special chars)

Normalization is general cleanup on text before tokeinzatrion (whitespace removal, lowercasing, removing accents, etc.)

You can acess a tokenizer like below:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
Copied
<class 'tokenizers.Tokenizer'>
```

This backend tokenizer has a normalizer method to test out normalization by itself:

```python
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

'hello how are u?'
```

### Pre-tokenization

​	Pretokenization can modify text- like replacing spaces withunderscores, or split text into tokens.



You can access pretokenization via the backend tokenizer attribute of a tokenizer:

```python
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
```

You can see the pretokenization step provides us with the offset mapping.

Different pretokenizers ghave different tules, and will therefore segment and modify the text differently:

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

[('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)),
 ('?', (19, 20))]
 
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

[('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
```

### SentencePiece

A tokenization algo, looks at text in unicode, replaces spaces with "_", if used in conjunction with Unigram it doesnt even need a pre-tokenization step.

### Algorithm Overview



| Model         | BPE                                                          | WordPiece                                                    | Unigram                                                      |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Training      | Starts from a small vocabulary and learns rules to merge tokens | Starts from a small vocabulary and learns rules to merge tokens | Starts from a large vocabulary and learns rules to remove tokens |
| Training step | Merges the tokens corresponding to the most common pair      | Merges the tokens corresponding to the pair with the best score based on the frequency of the pair, privileging pairs where each individual token is less frequent | Removes all the tokens in the vocabulary that will minimize the loss computed on the whole corpus |
| Learns        | Merge rules and a vocabulary                                 | Just a vocabulary                                            | A vocabulary with a score for each token                     |
| Encoding      | Splits a word into characters and applies the merges learned during training | Finds the longest subword starting from the beginning that is in the vocabulary, then does the same for the rest of the word | Finds the most likely split into tokens, using the scores learned during training |



## Byte-Pair Encoding Tokenization

### Training Algorithm

BPE training starts by getting a corpus of words in your text. (raw frequency of complete words post normalization and pre-tokenization).

The vocabulary starts as each unique char in the training text. This is why some algos are bad at recognizing chars like emojis because they were never included in the tokenization vocavulary and are turned into unknown tokens.

After getting the base vocal (unique chars), the algo learns "merges" unitl you reach your desired vocabulary size. These merges are rules to combine two elements of the existing vocabulary based on their frequencies within the training corpus. 

### Tokenization Algorithm

1. Normalization
2. Pre-tokenization
3. Splitting the words into individual characters
4. Applying the merge rules learned in order on those splits

### Implementing BPE

```python
# Define Courpus
corpus = [
    "This is the Hugging Face course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# pre-tokeinzation
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# Get word corpus
print(word_freqs)

defaultdict(int, {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1,
    'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1,
    'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1,
    'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1})

# Get base vocab
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

print(alphabet)

[ ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's',
  't', 'u', 'v', 'w', 'y', 'z', 'Ġ']

# Add special tokens
vocab = ["<|endoftext|>"] + alphabet.copy()

# Calculate merges / merge rules
## Split words into chars
splits = {word: [c for c in word] for word in word_freqs.keys()}

## Func to get merge freqs
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs
  
pair_freqs = compute_pair_freqs(splits)

## Get best mertge pair candidate
for i, key in enumerate(pair_freqs.keys()):
  print(f"{key}: {pair_freqs[key]}")
  if i >= 5:
    break
    
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)

## Add merge pair to vocab
merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

#Loop rto get the rest
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

# Run tokenization!
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
  
  tokenize("This is not a token.")

```

## WordPiece Tokenization

### Training Algorithm

⚠️ Google never open-sourced its implementation of the training algorithm of WordPiece, so what follows is Huggingface's best guess based on the published literature. It may not be 100% accurate.

We starts by splitting all words into chars, with each char contained after the first letter of a word with the prefix "##".

Next we create merge rules like BPE but use a different method for selection of the "best" merge rule:

score=(freq_of_pair)/(freq_of_first_element×freq_of_second_element)

This method minimizes creating merges of vocab items that appear frequently alone.

Note that the "##" chars are removed when merging,

### Tokenization Algorithm

WordPiece does not store moerge rules only the final vocab. Wordpiece reads each word from left to right and finds the longest subword in the vocab for the first letter and starts splitting there, once done it moves onto the next letter not contained in the first token.

Wordpiece will mark whole words it does not recgonize as unknown chars vs BPE which only does this at the individual character level.

### Implementing WordPiece

```python
# Define training corpus
corpus = [
    "This is the Hugging Face course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# Pretokenization
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# Word frequencies
print(word_freqs)

defaultdict(
    int, {'This': 3, 'is': 2, 'the': 1, 'Hugging': 1, 'Face': 1, 'Course': 1, '.': 4, 'chapter': 1, 'about': 1,
    'tokenization': 1, 'section': 1, 'shows': 1, 'several': 1, 'tokenizer': 1, 'algorithms': 1, 'Hopefully': 1,
    ',': 1, 'you': 1, 'will': 1, 'be': 1, 'able': 1, 'to': 1, 'understand': 1, 'how': 1, 'they': 1, 'are': 1,
    'trained': 1, 'and': 1, 'generate': 1, 'tokens': 1})

# Chars for baseline vocab
alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()

print(alphabet)

['##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##r', '##s',
 '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u',
 'w', 'y']

# Add special chars
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()


# split each word, with all the letters that are not the first prefixed by ##
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
}

# Create merge rules
## Compute pair scores
def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores
  
pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f"{key}: {pair_scores[key]}")
    if i >= 5:
        break

## Get best pair
best_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        best_pair = pair
        max_score = score

print(best_pair, max_score)
        
('a', '##b') 0.2

## Append to vocab
vocab.append("ab")

## Merge pairs in splits dict
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
  
splits = merge_pair("a", "##b", splits)
splits["about"]

## Calculate rest of merge rules
vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)

# Encode a text
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens
  
print(encode_word("Hugging"))
print(encode_word("HOgging"))

['Hugg', '##i', '##n', '##g']
['[UNK]']

# Tokenize a text using our tokenizer!
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])
  
tokenize("This is the Hugging Face course!")

['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'c', '##o', '##u', '##r', '##s',
 '##e', '[UNK]']
```

## Unigram tokenization

### Training Algorithm

Unigram starts with a large vocab and removes tokens until it reaches the desired size.

Unigram calculates a loss over the corpus given the vocab and calculates removing which token would result in the small increase inn the loss. Tokens that have the lowest overall effect on the loss are removed.

The hyperparam *p* is used to indicate what percent of symbols associated with the lowest loss increase to remove in each iteration.

### Tokenization Algorithm

First all possible substring from training corpus are created.

Next, the probability of each token is calculated (# of occurrences/ total token occurences). Then, aggregate probabilities for each possible tokenization of a word is calculated. We can use the Verbiti algorithm to produces these results more efficiently using a graph bases methodology.

### Back to Training

Next we calculate the loss for each of our tokens using the unigram model and remove the one that increases the loss the least.

### Implementing Unigram

```python
# Define training corpus
corpus = [
    "This is the Hugging Face course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# Get tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

# Apply pretokenization and get word freuqencies
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs

# Get all subwords
char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # Loop through the subwords of length at least 2
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

# Sort subwords by frequency
sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
sorted_subwords[:10]

[('▁t', 7), ('is', 5), ('er', 5), ('▁a', 5), ('▁to', 4), ('to', 4), ('en', 4), ('▁T', 3), ('▁Th', 3), ('▁Thi', 3)]

# Select most freq subwords to get startting vocab
token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}

# Calculate token loss
from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

# Get best segmentations for each word

def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score
  
print(encode_word("Hopefully", model))
print(encode_word("This", model))

(['H', 'o', 'p', 'e', 'f', 'u', 'll', 'y'], 41.5157494601402)
(['This'], 6.288267030694535)

# Compute model loss

def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss
  
compute_loss(model)
413.10377642940875

# Compute model loss after removing token
import copy


def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores
  
scores = compute_scores(model)
print(scores["ll"])
print(scores["his"])
6.376412403623874
0.0

# Get final vocab

percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    # Remove percent_to_remove tokens with the lowest scores.
    for i in range(int(len(model) * percent_to_remove)):
        _ = token_freqs.pop(sorted_scores[i][0])

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

# Tokenize using our tokenizer!!!
def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


tokenize("This is the Hugging Face course.", model)

['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁', 'c', 'ou', 'r', 's', 'e', '.']
```



## Building a tokenizer, block by block

As we have seen up until now, tokenization can be broken down into the following steps:

- Normalization (text cleanup, removing blanks, removing access, unicode normalization)
- Pre-tokenization (splitting into words)
- Tokenization model (produce tokens)
- Post-processing (special tokens, adding atttention masks, and token type ids, etc.)

The Tokenizer class has multiple submodeules that act as its buiding blocks:

- Normalizers (https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.normalizers)
- Pre-tokenizers (https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.pre_tokenizers)
- Models (BPE, Wordpiece, Unigram, etc.) (https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.models)
- trainers (https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.trainers)
- Post_processors (https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.processors)
- decoders (https://huggingface.co/docs/tokenizers/python/latest/components.html#decoders)

### Acquiring a corpus

We will use a small subset of Wiki data for the pruposes of training tokenizers:

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
```



### Building a WordPiece tokenizer from scratch (BERT, such as DistilBERT, MobileBERT, Funnel Transformers, and MPNET)

To builda tokenizer we must instantiate a Tokenizer model class and set its components (normalizer, pre_tokenizer, post_processor, decoder)

```python
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

# Instantiate tokenizer
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Set normalizer
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

## Test normalizer
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
'hello how are u?'

# Set pre-tokenizer
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)

## Test pre-tokenizer
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
[('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre', (14, 17)),
 ('-', (17, 18)), ('tokenizer', (18, 27)), ('.', (27, 28))]

# Set trainer, with special tokens
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# Train tokenizer!
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Set post-processor
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# Set decoder
tokenizer.decoder = decoders.WordPiece(prefix="##")

# Save tokenizer
tokenizer.save("tokenizer.json")

# Run tokenization!!!!
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)
['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '...', '[SEP]', 'on', 'a', 'pair', 'of', 'sentences', '.', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Turn our tokenizer into a fast tokenizer
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
```



### Building a BPE tokenizer from scratch (GPT, GPT-2, RoBERTa, BART, and DeBERTa)

Let's follow the same steps to build a BPE tokenizer:

```python
# Instantiate model class
tokenizer = Tokenizer(models.BPE())

# No normalization for GPT-2

# Set pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Test pre-tokenizer
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")
[('Let', (0, 3)), ("'s", (3, 5)), ('Ġtest', (5, 10)), ('Ġpre', (10, 14)), ('-', (14, 15)),
 ('tokenization', (15, 27)), ('!', (27, 28))]

# Set trainer
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])

# Train tokenizer!
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Set post-processor
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# Set decoder
tokenizer.decoder = decoders.ByteLevel()

# Wrap in fast tokenizer
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
```



### Building a Unigram tokenizer from scratch (AlBERT, T5, mBART, Big Bird, and XLNet)

Next, let's do the same for Unigram tokenizer:

```python
# Instantiate model class
tokenizer = Tokenizer(models.Unigram())

# Set normalizer
from tokenizers import Regex

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)


# Set pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

## Test pre-tokenizer
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!")
[("▁Let's", (0, 5)), ('▁test', (5, 10)), ('▁the', (10, 14)), ('▁pre-tokenizer!', (14, 29))]

# Set trainer
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)

# Train!
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Set postp=processor
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)

# Set decoder
tokenizer.decoder = decoders.Metaspace()

# Wrap in Fast tokenizer
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="left",
)
```



## Tokenizers check!

This chapter covered the following:

- Train a new tokenizer using an old one as a template
- Understand offset mapping
- Know the differences between BPE, WordPiece, and Unigram tokenization algos
- Be able to mix and match block using the 🤗Tokenizers to build your own tokenizer
- Use that tokenizer from previous bullet in the 🤗 Transformers library