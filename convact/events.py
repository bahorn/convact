from convact import LLMAskEventEmitter


class CatEvent(LLMAskEventEmitter):
    QUERIES = ['Does the following mention a cats name?']
    WAKEWORDS = ['cat', 'cats']
