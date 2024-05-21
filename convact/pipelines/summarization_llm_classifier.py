from ollama import Client
from llmoutput import BooleanResponse, map_llm_response_to_binary


class SummarizationLLMBooleanClassifer:
    """
    First asks the body to be summerized, then tries to classify it.
    """

    POSTFIX = "Answer with just a yes or a no, and NEVER elaborate"
    # needs more gaslighting
    SUMMERIZE = "Provide a short summary of the following transcript so it can"
    SUMMERIZE += "be used later"
    SUMMERIZE += "There may be errors, correct those if possible "
    SUMMERIZE += "but do not mention them, just fix."
    SUMMERIZE += "Do not say it can not be summerized, just try"

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

        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SUMMERIZE},
                {"role": "user", "content": body + '\n'}
            ],
            options={
                'temperature': 0.0,
                'num_predict': 100
            }
        )

        summary = response['message']['content']

        print('SUMMARY: ', summary)

        for idx, query in enumerate(self._queries):
            response = self._client.chat(
              model=self._model,
              messages=[
                {"role": "system", "content": query + self.POSTFIX},
                {"role": "user", "content": summary + '\n'}
              ],
              options={
                  'temperature': 0.0,
                  'num_predict': 3
              }
            )
            resp = response['message']['content']

            res.append(map_llm_response_to_binary(resp) == BooleanResponse.YES)

        return sum(res) >= self._threshold
