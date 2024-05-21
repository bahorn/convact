from taskman import Task, Message
from pipelines.basic_llm_classifier import BasicLLMBooleanClassifer
from llmoutput import normalize
from defaults import DECAY_STEPS, DEFAULT_OLLAMA, DEFAULT_MODEL
from taskman import TaskManager, TaskDescription
from audio.thread import PrerecordedTask, RealtimeTask
from transcription.task import TranscriptionTask
from transcription.transcript import RecentTranscript


class TranscriptSummerizeAndForwardTask(Task):
    """
    Applys corrections and summerizes the recent transcript and forwards it to,
    the recognizers.
    """

    def setup(self, data):
        self._transcript = RecentTranscript()
        self._summmaries = []
        self._to_forward = data['to_forward']

    def on_message(self, message: Message):
        d = message.data()
        if not d:
            return

        new = d['transcript'].split(' ')
        self._transcript.add_segment(new)

        print(new)

        # if we just generated a summary, forward it to the recognizers.

        for dest in self._to_forward:
            new_message = Message(dest, message.data())
            self.send(new_message)


class RecognizerTask(Task):
    def setup(self, data):
        self._transcript = RecentTranscript()
        self._decay_steps = data['decay_steps'] \
            if 'decay_steps' in data else DECAY_STEPS
        self._steps = 0

        host = data['host'] if 'host' in data else DEFAULT_OLLAMA
        model = data['model'] if 'model' in data else DEFAULT_MODEL

        self._classifer = BasicLLMBooleanClassifer(host, model)
        self._classifer.train(data['queries'])

        self._wakewords = set(normalize(word) for word in data['wakewords'])

    def active(self):
        return self._steps > 0

    def decay(self):
        self._steps = self._steps - 1 if self._steps > 0 else 0

    def on_message(self, message: Message):
        d = message.data()
        if not d:
            return

        new = d['transcript'].split(' ')

        self._transcript.add_segment(new)

        # We use wake words to avoid calling the LLM too much.

        for word in new:
            w = normalize(word)
            if w in self._wakewords:
                self._steps = self._decay_steps

        self.decay()

        if not self.active():
            return

        # Try and classify
        transcript = self._transcript.get()

        if self._classifer.run(transcript):
            data = {
                'source': self._name,
                'transcript': transcript
            }
            self._steps = 0
            return self.send(Message('action', data))

        return


class ActionTask(Task):
    def on_message(self, message: Message):
        print(message.name(), message.data())


def start(config, model, ollama_endpoint, filename=None):
    """
    Start the program
    """

    manager = TaskManager()

    # VAD

    # the transcription task
    manager.register(
        TaskDescription('transcription', TranscriptionTask, ['audio'])
    )

    # action task
    manager.register(
        TaskDescription('action', ActionTask, ['action'])
    )

    # register the recognizers
    recognizer_list = []

    for recognizer in config['recognizers']:
        conf = {
            'model': model,
            'host': ollama_endpoint,
            'wakewords': recognizer['wakewords'],
            'queries': recognizer['queries']
        }
        name = recognizer['name']
        manager.register(
            TaskDescription(name, RecognizerTask, [name], conf)
        )
        recognizer_list.append(name)

    # now the forwarders
    manager.register(
        TaskDescription(
            'transcription_forward',
            TranscriptSummerizeAndForwardTask,
            ['transcription'],
            {'to_forward': recognizer_list}
        )
    )

    # get the audio and start
    if filename is not None:
        manager.register(
            TaskDescription('audio', PrerecordedTask, ['start_prerecorded'])
        )
        manager.start('start_prerecorded', {'filename': filename})
    else:
        manager.register(
            TaskDescription('audio', RealtimeTask, ['start_realtime'])
        )
        manager.start('start_realtime', {})
