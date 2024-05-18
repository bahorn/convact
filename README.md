# ConvAct

Listens to conversations, emitting events based on asking an LLM if the
conversation is convering relevant things.

Using ollama (with llama3) and whisper, with wake words deciding if a call to
the LLM is needed.

Not that great yet, need to improve how the audio is batched.

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

Then you can run the code from your virtualenv with the following to use a file:

```
python3 convact prerecorded ./samples/config.json PATH_TO_FILE
```

or the following to use your microphone:
```
python3 convact realtime ./samples/config.json
```

Both support `--model` and `--ollama` to set the model and ollama endpoint.

Only tested on MacOS, on a M2 MacBook Air, with python installed via homebrew.
Running everything locally.


## License

MIT
