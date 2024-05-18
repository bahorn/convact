from events.core import ConvActions
from events.events import CatEvent
from audio.clean import clean_audio
from audio.transcription import Transcription


class EventEmitter:
    """
    Pushes events to a queue.
    """

    def __init__(self, queue):
        self._queue = queue

    def emit(self, event, data):
        self._queue.put((event, data))


def work_thread(host, model, transcription_queue, action_queue):
    """
    Transcribes a audio it reads from a queue and processes it using the event
    emitters.
    """

    transcriber = Transcription()
    emitters = [CatEvent(host, model)]
    event_emitter = EventEmitter(action_queue)
    ca = ConvActions(event_emitter, emitters)

    while True:
        try:
            joined = transcription_queue.get()
            new = clean_audio(joined)
            transcription = transcriber.get_transcription(new)

            if transcription:
                print('TRANSCRIPTION: ', transcription)
                ca.add_segment(transcription.split(' '))

        except KeyboardInterrupt:
            break


def action_thread(queue):
    """
    Act on the events.
    """

    while True:
        try:
            action = queue.get()
            print('EVENT: ', action)
        except KeyboardInterrupt:
            break
