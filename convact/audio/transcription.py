from defaults import FILTER_PROB


class Transcription:
    """
    Wrapper around whisper to perform the transcription we need.
    """

    def __init__(self, model_name='small.en', filter_prob=FILTER_PROB):
        self._filter_prob = filter_prob
        # only import here to keep things running smoothly
        import whisper
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
