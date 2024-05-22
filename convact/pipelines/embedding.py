import ollama
from scipy.spatial import distance
from defaults import MODEL_EMBEDDINGS


class Embedding:
    def __init__(self, model=MODEL_EMBEDDINGS):
        self._model = model

    def run(self, prompt):
        res = ollama.embeddings(model=self._model, prompt=' '.join(prompt))
        if 'embedding' in res:
            return res['embedding']
        return None

    def dist(self, a, b):
        return distance.cosine(a, b)
