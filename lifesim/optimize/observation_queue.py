import heapq
from queue import Queue

class Removed():

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, Removed)

class ObservationQueue(Queue):

    def _init(self, maxsize):
        self.queue = []
        self.REMOVED = Removed()
        self.entry_finder = {}

    def _put(self, item, heappush=heapq.heappush):
        item = list(item)
        priority, task = item
        identifier = task[:2]
        if identifier in self.entry_finder:
            previous_item = self.entry_finder[identifier]
            previous_priority, _ = previous_item
            # Remove previous item.
            previous_item[-1] = self.REMOVED
            self.entry_finder[identifier] = item
            heappush(self.queue, item)
        else:
            self.entry_finder[identifier] = item
            heappush(self.queue, item)

    def _qsize(self, len=len):
        return len(self.entry_finder)

    def _get(self, heappop=heapq.heappop):
        """
        The base makes sure this shouldn't be called if `_qsize` is 0.
        """
        while self.queue:
            item = self.queue[0]
            _, task = item
            if task is not self.REMOVED:
                return item
            heappop(self.queue)
        raise KeyError('It should never happen: pop from an empty priority queue')