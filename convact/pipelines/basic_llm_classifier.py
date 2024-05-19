from ollama import Client
from llmoutput import BooleanResponse, map_llm_response_to_binary


class BasicLLMBooleanClassifer:
    """
    One-Shot approach to use an LLM to classify, using a threshold of queries
    returning true
    """

    POSTFIX = "Answer with just a yes or a no, and NEVER elaborate"

    def __init__(self, host, model):
        self._queries = []
        self._threshold = 1
        self._client = Client(host=host)
        self._model = model

    def train(self, queries: list[str], threshold: int = 1):
        self._queries = queries
        self._threshold = threshold

    def run(self, body: str) -> bool:
        res = []
        for idx, query in enumerate(self._queries):
            response = self._client.chat(
              model=self._model,
              messages=[
                {"role": "system", "content": query + self.POSTFIX},
                {"role": "user", "content": body + '\n'}
              ],
              options={
                  'temperature': 0.0,
                  'num_predict': 3
              }
            )
            resp = response['message']['content']

            res.append(map_llm_response_to_binary(resp) == BooleanResponse.YES)

        return sum(res) >= self._threshold
