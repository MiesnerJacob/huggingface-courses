# Chapter 4 - Sharing models and tokenizers

## The Hugging Face Hub

the Huggingface hub is a model repository for all kinds of Deep Learning models (not limited to NLP models). There are SOTA models, fine-tnued models, and any model the open source community wants to contribute. There are currently over 10K publically available models on the Hugging Face Hub (badass). Sharing a model on the hub automatically deploys a hosted Inference API for that model so that anyone can quickly test the model and see if its right for them!

## Using pretained models

**Check to see which tasks your chosen model can perform! It may not have a pretrained head for all tasks.**

Using pipeline

```python
from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
```

Instantiaing directly

```python
from transformers import CamembertTokenizer, CamembertForMaskedLM 

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
```

Using the Auto* classes (architecture-agnostic loading)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM 

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

## Sharing pretrained models

Can view docs on sharing models here:  https://huggingface.co/transformers/model_sharing.html

Need to login to get auth token on local machine: 

```bash
transformers-cli login

huggingface-cli login
```

You should be prompted for your username and password.

### **There are three ways to go about creating new model repositories:**

- **Using the `push_to_hub` API:**

  ```python
  # Organization and use_auth_token params are optional
  tokenizer.push_to_hub(
      "dummy-model", organization="huggingface", use_auth_token="<TOKEN>"
  )
  
  model.push_to_hub(
      "dummy-model", organization="huggingface", use_auth_token="<TOKEN>"
  )
  ```

- **Using the `transformers` CLI:**

  ```
  usage: huggingface-cli <command> [<args>]
  
  positional arguments:
    {login,whoami,logout,repo,lfs-enable-largefiles,lfs-multipart-upload}
                          huggingface-cli command helpers
      login               Log in using the same credentials as on huggingface.co
      whoami              Find out which huggingface.co account you are logged
                          in as.
      logout              Log out
      repo                {create, ls-files} Commands to interact with your
                          huggingface.co repos.
      lfs-enable-largefiles
                          Configure your repository to enable upload of files >
                          5GB.
      lfs-multipart-upload
                          Command will get called by git-lfs, do not call it
                          directly.
  
  optional arguments:
    -h, --help            show this help message and exit
  ```

  Create repo example:

  ```bash\
  huggingface-cli repo create dummy-model --organization huggingface
  ```

- **Using the web interface:**

  To create a new repository, visit [huggingface.co/new](https://huggingface.co/new)

## Building a model card

The model card usually starts with a very brief, high-level overview of what the model is for, followed by additional details in the following sections:

- Model description
- Intended uses & limitations
- How to use
- Limitations and bias
- Training data
- Training procedure
- Evaluation results

Creating the model card is done through the *README.md* file, which is a Markdown file.

Check out the following for a few examples of well-crafted model cards:

- [`bert-base-cased`](https://huggingface.co/bert-base-cased)
- [`gpt2`](https://huggingface.co/gpt2)
- [`distilbert`](https://huggingface.co/distilbert-base-uncased)

The categories a model belongs to in the hub are identified according to the metadata you add in the model card header. For example:

```
---
language: fr
license: mit
datasets:
- oscar
---
```

Can access full model card specififcation example here: https://raw.githubusercontent.com/huggingface/huggingface_hub/main/modelcard.md

## Part 1 completed!

This is the end of the first part of the course!