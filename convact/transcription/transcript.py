from defaults import BUFFER_SIZE


class RecentTranscript:
    """
    Manage a small transcript of the latest words.
    """

    def __init__(self, buffer_size=BUFFER_SIZE):
        self._buffer = []
        self._buffer_size = buffer_size

    def add_segment(self, segment):
        self._buffer += segment
        self._buffer = self._buffer[-self._buffer_size:]

    def get(self):
        return ' '.join(self._buffer)

    def clear(self):
        self._buffer = []
