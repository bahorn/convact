from ollama import Client
from llmoutput import map_llm_response_to_binary, normalize, BooleanResponse
from transcript import RecentTranscript
from defaults import DECAY_STEPS


class Event:
    def __init__(self, event):
        self._event = event


class LLMAskEventEmitter:
    """
    Determines if an event should be emitted based on:
    * If it is active based on the appearance of a wakeword.
    * Asking an LLM with a specific prompt with it applies to the transcript.
    """

    QUERIES = []
    WAKEWORDS = []
    POSTFIX = "Answer with just a yes or a no, and NEVER elaborate."

    def __init__(self, host, model, decay_steps=DECAY_STEPS):
        self._steps = 0
        self._decay_steps = decay_steps
        self._client = Client(host=host)
        self._model = model

    def wakewords(self, word):
        if normalize(word) in self.WAKEWORDS:
            self._active = True
            self._steps = self._decay_steps
            return True
        return False

    def active(self):
        return self._steps > 0

    def decay(self):
        self._steps = self._steps - 1 if self._steps > 0 else 0

    def event(self, idx, query, transcript):
        name = self.__class__.__name__
        event = {
            'query_idx': idx,
            'query': query,
            'transcript': transcript
        }
        self._steps = 0
        return (name, event)

    def filter_events(self, events):
        return events

    def try_transcript(self, transcript):
        res = []

        for idx, query in enumerate(self.QUERIES):
            response = self._client.chat(
              model=self._model,
              messages=[
                {"role": "system", "content": query + self.POSTFIX},
                {"role": "user", "content": transcript + '\n'}
              ],
              options={
                  'temperature': 0.0,
              }
            )
            resp = response['message']['content']

            # print(transcript)
            # print(resp)

            if map_llm_response_to_binary(resp) == BooleanResponse.YES:
                res.append(self.event(idx, query, transcript))

        return self.filter_events(res)


class ConvActions:
    def __init__(self, event_emitter, emitters):
        self._transcript = RecentTranscript()
        self._emitters = emitters
        self._event_emitter = event_emitter

    def add_segment(self, segment):
        self._transcript.add_segment(segment)
        for word in segment:
            for emitter in self._emitters:
                if emitter.wakewords(word):
                    emitter.active()

        need_to_clear = False

        for emitter in self._emitters:
            emitter.decay()

            if emitter.active():
                events = emitter.try_transcript(self._transcript.get())
                for event, data in events:
                    self._event_emitter.emit(event, data)
                    need_to_clear = True

        if need_to_clear:
            self._transcript.clear()
