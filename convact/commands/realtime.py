import multiprocessing as mp
from audio.thread import audio_realtime_thread
from core import work_thread, action_thread


def realtime(model):
    transcription_queue = mp.Queue()
    action_queue = mp.Queue()
    audio = mp.Process(
        target=audio_realtime_thread,
        args=(transcription_queue, )
    )
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
