# Chapter 1 - Transformer Models

## Introduction

This course will teach you about natural language processing (NLP) using libraries from the [Hugging Face](https://huggingface.co/) ecosystem — [🤗 Transformers](https://github.com/huggingface/transformers), [🤗 Datasets](https://github.com/huggingface/datasets), [🤗 Tokenizers](https://github.com/huggingface/tokenizers), and [🤗 Accelerate](https://github.com/huggingface/accelerate) — as well as the [Hugging Face Hub](https://huggingface.co/models). It’s completely free and without ads.

## Natural Lanugage Processing

NLP is a field of linguistics and machine learning focused on understanding everything related to human language. The aim of NLP tasks is not only to understand single words individually, but to be able to understand the context of those words. Examples of NLP tasks include classifying whole sentences, classifying each word in a sentence, generating text content, extracting an answer from a text, generating a new sentence from an input text.

## Transformers - What can they do? (Pipelines)

There are three main steps involved when you pass some text to a pipeline:

1. The text is preprocessed into a format the model can understand.
2. The preprocessed inputs are passed to the model.
3. The predictions of the model are post-processed, so you can make sense of them.

Some of the currently available pipelines are:

- `feature-extraction` (get the vector representation of a text)
- `fill-mask`
- `ner` (named entity recognition)
- `question-answering`
- `sentiment-analysis`
- `summarization`
- `text-generation`
- `translation`
- `zero-shot-classification`

## How do Transformers work?

The model is primarily composed of two blocks:

- **Encoder (left)**: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
- **Decoder (right)**: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

A key feature of Transformer models is that they are built with special layers called *attention layers*. 

## Encoder Models

Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called *auto-encoding models*.

The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.

## Decoder Models

Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called *auto-regressive models*.

The pretraining of decoder models usually revolves around predicting the next word in the sentence.

## Sequence-to-sequence models

Encoder-decoder models (also called *sequence-to-sequence models*) use both parts of the Transformer architecture. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.

The pretraining of these models can be done using the objectives of encoder or decoder models, but usually involves something a bit more complex. For instance, T5 is pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces.

## Bias and Limitations

If your intent is to use a pretrained model or a fine-tuned version in production, please be aware that, while these models are powerful tools, they come with limitations. The biggest of these is that, to enable pretraining on large amounts of data, researchers often scrape all the content they can find, taking the best as well as the worst of what is available on the internet. Fine-tuning the model on your data won’t make this intrinsic bias the pre-trained model has learned disappear.

## Summary

| Model           | Examples                                   | Tasks                                                        |
| --------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Encoder         | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Sentence classification, named entity recognition, extractive question answering |
| Decoder         | CTRL, GPT, GPT-2, Transformer XL           | Text generation                                              |
| Encoder-decoder | BART, T5, Marian, mBART                    | Summarization, translation, generative question answering    |