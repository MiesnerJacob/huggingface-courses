# Building and Sharing Demos

## Introduction to Gradio

This chapter will cover building GUI applications for your machine learning models!!! 

We will be using the Gradio library to build, customize, and share web-based demos for machine learning models!

Examples of Gradio applications include:

- sketch recognition
- question answering
- image background removal

This chapter will include both concepts and applications so we will learn and then put them into practice.



## Building tour first demo

Gradio can be run almost anywhere including an IDE, ipython notebook, Google Colab, etc. You can install Gradio like so:

```python
pip install gradio
```

Let's create a "Hello World" application with Gradio to understand the components at play:

```python
import gradio as gr

def greet(name):
  return "Hello " + name

demo = gr.Inferface(fn=greet, 
                    inputs="text",
                    outputs="text")

demo.launch()
```

- First, we create our function for the application to use, typically this would be a function to call an ML model and make a prediction.
- Next, we instantiate an Interface object with the function and the inputs and output types as arguments.
- Lastly, we launch the app!

If you run this app in a notebook it will display the GUI there, if you run it in a script it will show up at **[http://localhost:7860](http://localhost:7860/)**

If we want we can explicitly define our input or output objects more clearly like so:

```python
import gradio as gr


def greet(name):
    return "Hello " + name


# We instantiate the Textbox class
textbox = gr.Textbox(label="Type your name here:", placeholder="John Doe", lines=2)

gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()
```

To build an app that uses a model to make preidctions we just need to load in our model and create a predict function to pass to our Interface object. This is shown below:

```python
from transformers import pipeline
import gradio as gr

model = pipeline("text-generation")

def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion
    
    import gradio as gr

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
```



## Understanding the Interface class

In order to create a Gradio Interface, you must specify a function, input, and output.

For a complete list of components that can be used as inputs or outputs, [see the Gradio docs.

First lets build an example using audio:

```python
# Imports
import numpy as np
import gradio as gr

# Function to reverse input audio
def reverse_audio(audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

# Create audio input
mic = gr.Audio(source="microphone", type="numpy", label="Speak here...")

# Create and launch interface
gr.Interface(reverse_audio, mic, "audio").launch()
```

Next we build an interface with multiple inputs (also applies to outputs). In order to do this we just have to pass a list of components to the inputs variable of the Interface object as shown below:

```python
# Imports
import numpy as np
import gradio as gr

# Set note options
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Define funcntion to execute
def generate_tone(note, octave, duration):
    sr = 48000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return (sr, audio)

# Define interface with multiple inputs and launch
gr.Interface(
    generate_tone,
    [
        gr.Dropdown(notes, type="index"),
        gr.Slider(minimum=4, maximum=6, step=1),
        gr.Textbox(type="number", value=1, label="Duration in seconds"),
    ],
    "audio",
).launch()
```

You can customize the behavior of `launch()` through different parameters:

- `inline` - whether to display the interface inline on Python notebooks.
- `inbrowser` - whether to automatically launch the interface in a new tab on the default browser.
- `share` - whether to create a publicly shareable link from your computer for the interface. Kind of like a Google Drive link!

We can also build apps with optional inputs, or make one require like so:

```python
# Imports
from transformers import pipeline
import gradio as gr

# Instantiate speech recognition pipeline via Huggingface
model = pipeline("automatic-speech-recognition")


# Function to execute
def transcribe_audio(mic=None, file=None):
    if mic is not None:
        audio = mic
    elif file is not None:
        audio = file
    else:
        return "You must either provide a mic recording or a file"
    transcription = model(audio)["text"]
    return transcription

# Create and launch Interface with optional set to True on inputs objects
gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(source="microphone", type="filepath", optional=True),
        gr.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs="text",
).launch()
```



## Sharing demos with others

Gradio demos can be shared in two ways: using a ***temporary share link\*** or ***permanent hosting on Spaces\***.

To add some additional context to your application the Interface class accepts the following params:

- title
- description
- article
- theme
- examples
- Live

Here is an example of how we use some of those options above:

```python
title = "Ask Rick a Question"
description = """
The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

article = "Check out [the original Rick and Morty Bot](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) that this demo is based off of."

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[["What are you doing?"], ["Where should we time travel to?"]],
).launch()
```

### Sharing your demo with temporary links

To get a temporary sharable link use:

```python
gr.Interface(classify_image, "image", "label").launch(share=True)
```

This will give you a link that will be up for 72 and hosted on your local machine (so your device must be on for it to work).

### Hosting your demo on Hugging Face Spaces

To host an app on HF Hub we must build a project with a directory that looks like

```
-project
--app.py
--requirements.txt
--README.md
--model.h5 () Optional
```

Let's build a Pictionary demo we can host on the HF hub!

```python
# Imports
from pathlib import Path
import torch
import gradio as gr
from torch import nn

# Load in names of classes to predict
LABELS = Path("class_names.txt").read_text().splitlines()

# Define model architecture
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)

# Load model weights
state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# Function to run inference, return preds
def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}
    
# Instantiate gradio interface
interface = gr.Interface(
    predict,
    inputs="sketchpad",
    outputs="label",
    theme="huggingface",
    title="Sketch Recognition",
    description="Who wants to play Pictionary? Draw a common object like a shovel or a laptop, and the algorithm will guess in real time!",
    article="<p style='text-align: center'>Sketch Recognition | Demo Model</p>",
    live=True,
)

# Launch gradio app with sharable link
interface.launch(share=True)
```

Notice the `live=True` parameter in `Interface`, which means that the sketch demo makes a prediction every time someone draws on the sketchpad (no submit button!).



## Integrations with Huggingface Hub

HF and Gradio have an integration that allow you to load model straight from the HF Hub! All you have to do is provide the "hugging face/" prefix when defining the model in your interface.

```python
# Import
import gradio as gr

# Metadata
title = "GPT-J-6B"
description = "Gradio Demo for GPT-J 6B, a transformer model trained using Ben Wang's Mesh Transformer JAX. 'GPT-J' refers to the class of model, while '6B' represents the number of trainable parameters. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/kingoflolz/mesh-transformer-jax' target='_blank'>GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model</a></p>"
examples = [
    ["The tower is 324 metres (1,063 ft) tall,"],
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

# Create interface with model from HF Hub & Launch!
gr.Interface.load(
    "huggingface/EleutherAI/gpt-j-6B",
    inputs=gr.Textbox(lines=5, label="Input Text"),
    title=title,
    description=description,
    article=article,
    examples=examples,
    enable_queue=True,
).launch()
```

Loading a model in this fashion allows the model to be run via Huggingface's inference API and not locally, this is huge for larger models that are hard to stuff into the RAM on your local machine.

You can also load a space someone else has created with one line of code!

```python
# Use the spaces/ prefix!
gr.Interface.load("spaces/abidlabs/remove-bg").launch()
```

You can override arguements of an existing space as well:

```python
gr.Interface.load(
    "spaces/abidlabs/remove-bg", inputs="webcam", title="Remove your webcam background!"
).launch()
```



## Advanced Interface Features

### Using state to persist data

Gradio supports what are called *session states* which allow you to store and reference multiple data inputs from a user within a page load, this is particularly useful for use cases like chatbots where you may want to reference those multiple inputs.

To store data in a session state, you need to do three things:

1. Pass in an *extra parameter* into your function, which represents the state of the interface.
2. At the end of the function, return the updated value of the state as an *extra return value*.
3. Add the ‘state’ input and ‘state’ output components when creating your `Interface`.

Here is an example:

```python
# Imports
import random
import gradio as gr


# Pass in state return state
def chat(message, history):
    history = history or []
    if message.startswith("How many"):
        response = random.randint(1, 10)
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, response))
    return history, history

# Add state components to input and output
iface = gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
    allow_screenshot=False,
    allow_flagging="never",
)
iface.launch()
```



### Using interpretation to understand predictions

Gradio includes some functionality to interpret predictions by seeing what parts of the input are responsible for the output. All you have to do is set the interpretation argument when defining your interface:

```python
# Imports
import requests
import tensorflow as tf
import gradio as gr

# Load model
inception_net = tf.keras.applications.MobileNetV2()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Function for inference and preds
def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}

# Define image size and label properties
image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)

# Create title
title = "Gradio Image Classifiction + Interpretation Example"

# Create interface with interpretation option, and launch!
gr.Interface(
    fn=classify_image, inputs=image, outputs=label, interpretation="default", title=title
).launch()
```

Gradio also supports shap interpretation, you can read about it [here](https://christophm.github.io/interpretable-ml-book/shap.html). You can also pass your own custom interpretation function if you want!



## Introduction to Blocks

Gradio supports both:

- Interface: a high-level API for building ML apps by defining just model, inputs and outputs
- Blocks: a low-level API that allows you to have full control over the application allowing you to build complex multi-step applications.

### Why Blocks 🧱?

Interface is great for quickly standing up an app, but when you want the flexibility to change the app at a granular level Blocks is more appropriate because it allows you to:

- Group together multiple demos as tabs in a web application
- Change the layout of your demo
- Multi-step inference (i.e. output of one model becomes an input to another), and more granular control over data flows in general
- Change a component's properties

### Creating a simple demo using Blocks

Let's start off with an example and explain the components:

```python
import gradio as gr


def flip_text(x):
    return x[::-1]


demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
    # Flip Text!
    Start typing below to see the output.
    """
    )
    input = gr.Textbox(placeholder="Flip this text")
    output = gr.Textbox()

    input.change(fn=flip_text, inputs=input, outputs=output)

demo.launch()
```

In this example we:

1. Combine multiple components with gr.Blocks()

2. You can run regular python functions with Blocks, and write your functions anywhere in the code

3. You can assign events to any Blocks component. To see a list of events that each component supports, see the Gradio [documentation](https://www.gradio.app/docs/).

4. Gradio auto-recognizes which components should be interactive, but you can set it explicitly like 

   ```python
   gr.Textbox(placeholder="Flip this text", interactive=True)
   ```

### Customizing the layout of your demo

By default, `Blocks` renders the components that you create vertically in one column. You can change that by creating additional columns `with gradio.Column():` or rows `with gradio.Row():` and creating components within those contexts. You can also create tabs for your demo by using the `with gradio.Tabs()`. Here's an example:

```python
# Imports
import numpy as np
import gradio as gr

# Instantiate Blocks object
demo = gr.Blocks()

# Function for flipping text
def flip_text(x):
    return x[::-1]

# Function for flipping image
def flip_image(x):
    return np.fliplr(x)

# Application layout definition
with demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tabs():
        with gr.TabItem("Flip Text"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")
        with gr.TabItem("Flip Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

    # Events definitions
    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

# Launch!
demo.launch()
```

### Exploring events and state

There are different event components for different inputs, for example, a textbox input has both a change() and submit() event. More complex components have more events associated with them. You can attach functions to these events by defining the functions, inputs, and outputs like so:

```python
# Imports
import gradio as gr

# Load model
api = gr.Interface.load("huggingface/EleutherAI/gpt-j-6B")

# Define function to run
def complete_with_gpt(text):
    # Use the last 50 characters of the text as context
    return text[:-50] + api(text[-50:])

# Create app layout via Blocks
with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Type here and press enter...", lines=4)
    btn = gr.Button("Generate")

    # Assign event action to the button
    btn.click(complete_with_gpt, textbox, textbox)

# Luanch!
demo.launch()
```

### Creating multi-step demos

In some cases, you may want a multi-step demo where the output of one model is the input for another. We will illustrate this with a speech-to-text application that passes the output to a sentiment analysis model:

```python
# Imports
from transformers import pipeline
import gradio as gr

# Instantiate HF pipelines
asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")

# Function for speech to text inference
def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

# Function for speech to sentiment classification inference
def text_to_sentiment(text):
    return classifier(text)[0]["label"]

# Instantiate Blocks
demo = gr.Blocks()

# Create application layout
with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    # Define button actions
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

# Launch application
demo.launch()
```



### Updating Component Properties

We have gone over how to create events to update the values of another component. In this section we will go over th how to change the visibility of a component given the output of another. You can do this by returning a component class’s `update()` method instead of a regular return value from your function.

```python
# Imports
import gradio as gr

# Function for changing box size
def change_textbox(choice):
    if choice == "short":
        return gr.Textbox.update(lines=2, visible=True)
    elif choice == "long":
        return gr.Textbox.update(lines=8, visible=True)
    else:
        # Set visible to false if 'short' or 'long' are not picked
        return gr.Textbox.update(visible=False)

# Define app layout
with gr.Blocks() as block:
  	# Create radio button
    radio = gr.Radio(
        ["short", "long", "none"], label="What kind of essay would you like to write?"
    )
    # Define text box as interactive
    text = gr.Textbox(lines=2, interactive=True)

    # Set event action on radio button
    radio.change(fn=change_textbox, inputs=radio, outputs=text)
    
    # Launch!
    block.launch()
```



## Gradio, check!

In this chapter we learned:

- How to create Gradio demos with the high-level `Interface` API, and how to configure different input and output modalities.
- Different ways to share Gradio demos, through temporary links and hosting on [Hugging Face Spaces](https://huggingface.co/spaces).
- How to integrate Gradio demos with models and Spaces on the Hugging Face Hub.
- Advanced features like storing state in a demo or interpreting predictions.
- How to have full control of the data flow and layout of your demo with Gradio Blocks.