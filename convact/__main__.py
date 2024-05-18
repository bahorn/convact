import click
from commands.realtime import realtime as realtime_func
from commands.prerecorded import prerecorded as prerecorded_func

from defaults import DEFAULT_MODEL


# Use a file for transcription, mainly for testing
@click.command()
@click.option('--model', default=DEFAULT_MODEL)
@click.argument('filename')
def prerecorded(model, filename):
    prerecorded_func(filename, model)


# Collect the transcription at runtime
@click.command()
@click.option('--model', default=DEFAULT_MODEL)
def realtime(model):
    realtime_func(model)


@click.group()
def main():
    pass


main.add_command(prerecorded)
main.add_command(realtime)


if __name__ == "__main__":
    main()
