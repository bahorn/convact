from ollama import Client
from llmoutput import map_llm_response_to_binary, normalize, BooleanResponse
from defaults import DECAY_STEPS


class LLMAskEventEmitter:
    """
    Determines if an event should be emitted based on:
    * If it is active based on the appearance of a wakeword.
    * Asking an LLM with a specific prompt with it applies to the transcript.
    """
    NAME = None
    QUERIES = []
    WAKEWORDS = []
    POSTFIX = "Answer with just a yes or a no, and NEVER elaborate."

    def __init__(self, host, model, name=None, decay_steps=DECAY_STEPS):
        self._steps = 0
        self._decay_steps = decay_steps
        self._client = Client(host=host)
        self._model = model

        if not self.NAME or name:
            self._name = self.__class__.__name__ if not name else name
        else:
            self._name = self.NAME

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
        event = {
            'query_idx': idx,
            'query': query,
            'transcript': transcript
        }
        self._steps = 0
        return (self._name, event)

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


def event_from_dict(event):
    class NewEvent(LLMAskEventEmitter):
        QUERIES = event['queries']
        WAKEWORDS = event['wakewords']
        NAME = event['name']

    return NewEvent
