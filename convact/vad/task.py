from taskman import Task, Message


class VADTask(Task):
    """
    Filtering out silence using a VAD model.

    Using Silero VAD.
    """

    def setup(self, data):
        self._model = None

    def on_message(self, message: Message):
        pass
