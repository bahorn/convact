from taskman import Task, Message
from transcription.audiotranscription import Transcription
from audio.clean import clean_audio


class TranscriptionTask(Task):
    """
    Transcribe a segment of audio
    """

    def setup(self, data):
        self._transcriber = Transcription()

    def on_message(self, message: Message):
        if message.name() != 'audio':
            return

        cleaned = clean_audio(message.data())
        transcription = self._transcriber.get_transcription(cleaned)

        if transcription:
            self.send(
                Message('transcription', {'transcript': transcription})
            )
