import itertools
import heapq


class PriorityQueue(object):

    INFINITY = 999999999999999999999999

    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def add_task(self, task, priority=0):
        """
            Add a new task or update the priority of an existing task
        """
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry     # 0 => [10, 1, 0]
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        """
            Mark an existing task as REMOVED.  Raise KeyError if not found.
        """
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        """
            Remove and return the lowest priority task. Raise KeyError if empty.
        """
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def get_priority(self, task):
        return self.entry_finder[task][0]

    def not_empty(self):
        return len(self.entry_finder) > 0

    def contains_task(self, task):
        return task in self.entry_finder
        #try:
            #entry = self.entry_finder[task]
            #return True
        #except:
            #return False
