from transcript import RecentTranscript


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
                event = emitter.try_transcript(self._transcript.get())
                if event:
                    self._event_emitter.emit(event[0], event[1])
                    need_to_clear = True

        if need_to_clear:
            self._transcript.clear()
