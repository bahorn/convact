import click
import json
from core import start
from defaults import DEFAULT_MODEL, DEFAULT_OLLAMA


# Use a file for transcription, mainly for testing
@click.command()
@click.option('--model', default=DEFAULT_MODEL)
@click.option('--ollama', default=DEFAULT_OLLAMA)
@click.argument('config')
@click.argument('filename')
def prerecorded(model, ollama, config, filename):
    with open(config) as f:
        start(json.load(f), model, ollama, filename=filename)


# Collect the transcription at runtime
@click.command()
@click.option('--model', default=DEFAULT_MODEL)
@click.option('--ollama', default=DEFAULT_OLLAMA)
@click.argument('config')
def realtime(model, ollama, config):
    with open(config) as f:
        start(json.load(f), model, ollama, filename=None)


@click.group()
def main():
    pass


main.add_command(prerecorded)
main.add_command(realtime)


if __name__ == "__main__":
    main()
