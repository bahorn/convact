from datetime import datetime, timedelta
from taskman import Task, Message, SchedulerTask
from pipelines.basic_llm_classifier import BasicLLMBooleanClassifer
from pipelines.summerization import SummerizationWithLLM
from llmoutput import normalize
from defaults import DEFAULT_OLLAMA, DEFAULT_MODEL
from taskman import TaskManager, TaskDescription
from audio.thread import PrerecordedTask, RealtimeTask
from transcription.task import TranscriptionTask
from transcription.transcript import RecentTranscript
# from vad.task import VADTask


class TranscriptSummerizeAndForwardTask(Task):
    """
    Applys corrections and summerizes the recent transcript and forwards it to,
    the recognizers.
    """

    def setup(self, data):
        self._transcript = RecentTranscript()
        self._summaries = []
        self._to_forward = data['to_forward']
        self._model = SummerizationWithLLM(data['host'], data['model'])

        self._last_updated = datetime.fromtimestamp(0)
        self._last_forwarded = datetime.fromtimestamp(0)

    def on_message(self, message: Message):
        d = message.data()

        if message.name() == 'transcription':
            new = d['transcript'].split(' ')
            self._transcript.add_segment(new)

            self._last_updated = datetime.now()

            print(new)
            return

        if message.name() == 'forward_summary':
            if self._last_forwarded > self._last_updated:
                return

            # generate a summary, then forward.
            # todo: implement logic to stop this from forwarding if there is no
            # new messages.
            summary = self._model.run(self._transcript.get())
            self._summaries.append(summary)
            self._summaries[-5:]
            for dest in self._to_forward:
                new_message = Message(
                    dest, {'transcript': self._summaries[-1]}
                )
                self.send(new_message)

            self._last_forwarded = datetime.now()
            return


class RecognizerTask(Task):
    def setup(self, data):
        host = data['host'] if 'host' in data else DEFAULT_OLLAMA
        model = data['model'] if 'model' in data else DEFAULT_MODEL

        self._classifer = BasicLLMBooleanClassifer(host, model)
        self._classifer.train(data['queries'])

        self._wakewords = set(normalize(word) for word in data['wakewords'])
        self._last_found = datetime.fromtimestamp(0)

    def on_message(self, message: Message):
        # cool off period
        if self._last_found + timedelta(seconds=10) > datetime.now():
            return

        d = message.data()
        if not d:
            return

        transcript = d['transcript']

        # We use wake words to avoid calling the LLM too much.
        found = False

        for word in transcript.split(' '):
            w = normalize(word)
            if w in self._wakewords:
                found = True

        if not found:
            return

        # Try and classify
        print('here', transcript)
        if self._classifer.run(transcript):
            print('found')
            data = {
                'source': self._name,
                'transcript': transcript
            }
            self._steps = 0
            self._last_found = datetime.now()
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

    manager.register(
        TaskDescription('sched', SchedulerTask, ['start_sched'], {
            'sched': [{'time': 5, 'message_name': 'forward_summary'}]}
        )
    )
    manager.send_message(Message('start_sched', {}))

    # VAD
    # manager.register(
    #    TaskDescription('vad', VADTask, ['vad'])
    # )

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
            [
                'transcription',
                'forward_summary'
            ],
            {
                'to_forward': recognizer_list,
                'host': ollama_endpoint,
                'model': model

            }
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
