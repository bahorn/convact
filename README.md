# ConvAct

Listens to conversations, emitted events based on asking an LLM if the event is
relevant.

Using ollama (with llama3) and whisper.

## Setup

Some variant of the following to get the dependencies:

```
virtualenv -p python3 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run ollama in a seperate terminal:
```
ollama pull llama3
ollama serve
```


Then you can run the code from your virtualenv:
```
python3 convact
```

Only tested on MacOS, with python installed via homebrew.
