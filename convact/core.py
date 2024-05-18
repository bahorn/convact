import pyaudio
import whisper
import numpy as np
from convact import ConvActions
from events import CatEvent
import librosa

from defaults import CHANNELS, RATE, CHUNK, BATCH, FILTER_PROB

FORMAT = pyaudio.paInt16


class EventEmitter:
    """
    Pushes events to a queue.
    """

    def __init__(self, queue):
        self._queue = queue

    def emit(self, event, data):
        self._queue.put((event, data))


class Transcription:
    """
    Wrapper around whisper to perform the transcription we need.
    """

    def __init__(self, model_name='small.en', filter_prob=FILTER_PROB):
        self._filter_prob = filter_prob
        self._model = whisper.load_model(model_name)

    def get_transcription(self, joined):
        res = self._model.transcribe(
            joined,
            language='en',
            no_speech_threshold=self._filter_prob
        )

        if len(res['segments']) == 0:
            return None

        message = ''

        for segment in res['segments']:
            if segment['no_speech_prob'] > self._filter_prob:
                continue
            message += segment['text']

        return message.strip()


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

            # just trying to remove silence for performance reasons.
            new, _ = librosa.effects.trim(
                y=joined,
                frame_length=RATE,
                top_db=40
            )

            transcription = transcriber.get_transcription(new)

            if transcription:
                print('TRANSCRIPTION: ', transcription)
                ca.add_segment(transcription.split(' '))

        except KeyboardInterrupt:
            break


def audio_thread(queue):
    """
    Captures audio in batches and pushes it to a queue.
    """
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    batch = []

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=True)
            data = np.frombuffer(data, np.int16) \
                .flatten() \
                .astype(np.float32)
            # some examples use the peak, but we want to be able to detect
            # silence, so we need to do this.
            data /= 32768.0

            batch.append(data)

            if len(batch) < BATCH:
                continue

            queue.put(np.concatenate(batch, axis=0))

            batch = []
        except KeyboardInterrupt:
            break

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()


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
