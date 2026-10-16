"""One-frame webcam buffer; slow inference drops old frames, never queues latency."""

from threading import Condition, Thread


class LatestCapture:
    def __init__(self, source):
        self.source = source
        self.changed = Condition()
        self.frame = None
        self.sequence = self.consumed = 0
        self.closed = False
        self.thread = Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while True:
                with self.changed:
                    if self.closed:
                        break
                ok, frame = self.source.read()
                if not ok:
                    break
                with self.changed:
                    self.frame = frame
                    self.sequence += 1
                    self.changed.notify_all()
        finally:
            with self.changed:
                self.closed = True
                self.changed.notify_all()

    def read(self):
        with self.changed:
            self.changed.wait_for(
                lambda: self.closed or self.sequence != self.consumed, timeout=10
            )
            if self.sequence == self.consumed:
                return False, None
            self.consumed = self.sequence
            return True, self.frame

    def close(self):
        with self.changed:
            self.closed = True
            self.changed.notify_all()
        self.source.release()
        self.thread.join(timeout=2)
