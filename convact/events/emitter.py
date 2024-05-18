class EventEmitter:
    """
    Pushes events to a queue.
    """

    def __init__(self, queue):
        self._queue = queue

    def emit(self, event, data):
        self._queue.put((event, data))
