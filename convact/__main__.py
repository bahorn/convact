import multiprocessing as mp
import click
from core import audio_thread, work_thread, action_thread
from defaults import DEFAULT_MODEL


@click.command()
@click.option('--model', default=DEFAULT_MODEL)
def main(model):
    transcription_queue = mp.Queue()
    action_queue = mp.Queue()
    audio = mp.Process(target=audio_thread, args=(transcription_queue, ))
    audio.start()

    work = mp.Process(
        target=work_thread,
        args=(
            'http://localhost:11434',
            model,
            transcription_queue,
            action_queue
        )
    )
    work.start()

    action = mp.Process(
        target=action_thread,
        args=(action_queue, )
    )
    action.start()

    audio.join()
    work.join()
    action.join()


if __name__ == "__main__":
    main()