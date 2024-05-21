"""
Summerization of a text.
"""
from ollama import Client


class SummerizationWithLLM:
    """
    Summerize the text with an LLM.
    """

    PROMPT = "Summarize the following transcript so it can be used later"
    PROMPT += "for classification."
    PROMPT += "There may be errors, correct those if possible "
    PROMPT += "but do not mention them, just fix."
    PROMPT += "Do not say it can not be summerized, just try."

    def __init__(self, host, model):
        self._client = Client(host=host)
        self._model = model

    def run(self, body):
        response = self._client.chat(
          model=self._model,
          messages=[
            {"role": "system", "content": self.PROMPT},
            {"role": "user", "content": body + '\n'}
          ],
          options={
              'temperature': 0.0,
              'num_predict': 100
          }
        )
        return response['message']['content']
