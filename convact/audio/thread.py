import pyaudio
import numpy as np

from defaults import CHANNELS, RATE, CHUNK, BATCH

FORMAT = pyaudio.paInt16


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
