import pyaudio
import librosa
import numpy as np

from taskman import Task, Message

from defaults import CHANNELS, RATE, CHUNK, BATCH

FORMAT = pyaudio.paInt16


class RealtimeTask(Task):
    def on_message(self, message: Message):
        if message.name() != 'start_realtime':
            return

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

                self._out_queue.put(
                    Message('audio', np.concatenate(batch, axis=0))
                )

                batch = []
            except KeyboardInterrupt:
                break

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class PrerecordedTask(Task):
    def on_message(self, message: Message):
        if message.name() != 'start_prerecorded':
            return

        filename = message.data()['filename']
        # already had librosa as a dependency, might as well use it for file
        # reading.
        y, _ = librosa.load(filename, sr=RATE, dtype=np.float32)
        # convert to a batch
        for chunk in chunks(y, CHUNK * BATCH):
            self._out_queue.put(Message('audio', chunk))
