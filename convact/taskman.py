"""
A simple task manager library for running multiple tasks as seperate processes,
implemented as message handlers.

Meant to make managing some applications a bit easier, allowing debugging, etc.

Not really designed for high performance message passing, just passing some
what regular messages.
"""
import time
import logging
import schedule
from datetime import datetime
from multiprocessing import Queue, Process


logger = logging.getLogger('taskman')
logging.basicConfig(level=logging.INFO)


class Message:
    def __init__(self, name, data):
        self._name = name
        self._data = data
        self._time = datetime.now()

    def name(self):
        return self._name

    def data(self):
        return self._data

    def time(self):
        return self._time


class MessageBus:
    """
    Forwards messages to the right place.
    """

    def __init__(self):
        self._queue_mapping = {}
        self._forwarding_table = {}
        self._out_queue = Queue()

    def register_process_queue(self, name: str) -> Queue:
        """
        Register a set of destinations.
        """
        queue = Queue()
        self._queue_mapping[name] = queue
        return queue

    def get_out_queue(self) -> Queue:
        return self._out_queue

    def register_message_handler(self, message_type: str, process: str):
        if process not in self._queue_mapping:
            raise Exception('Process not registered')
        self._forwarding_table[message_type] = process

    def send(self, message):
        self._out_queue.put(message)

    def step(self):
        event = self._out_queue.get()
        if event.name() in self._forwarding_table:
            dest = self._forwarding_table[event.name()]
            logger.info(f'event: {event.name()} -> {dest}')
            self._queue_mapping[dest].put(event)
        else:
            raise Exception('Event with no known destination')


class Task:
    """
    A task for the system.
    """

    def __init__(self, name, queue_pair, data={}):
        self._name = name
        self._in_queue = queue_pair[0]
        self._out_queue = queue_pair[1]
        # task specific setup
        self.setup(data)

    def setup(self, data):
        pass

    def on_message(self, message: Message):
        return

    def send(self, message):
        if isinstance(message, list):
            for msg in message:
                self._out_queue.put(msg)
        else:
            self._out_queue.put(message)

    def message_handler(self):
        while True:
            try:
                message = self._in_queue.get()
                self.on_message(message)
            except KeyboardInterrupt:
                break


class TaskDescription:
    def __init__(self, name, task, messages_types, config={}):
        self._name = name
        self._task = task
        self._message_types = messages_types
        self._config = config

    def name(self) -> str:
        return self._name

    def message_types(self) -> list[str]:
        return self._message_types

    def get_task(self) -> Task:
        return self._task

    def config(self) -> dict:
        return self._config


def run_task(queue_pair: tuple[Queue, Queue],
             task_description: TaskDescription):
    """
    Runs a task from the description.
    """
    logging.info(f'Starting: {task_description.name()}')
    task = task_description.get_task()(
            task_description.name(), queue_pair, task_description.config()
        )
    task.message_handler()


class TaskManager:
    def __init__(self):
        self._tasks = []
        self._bus = MessageBus()

    def register(self, task: TaskDescription):
        """
        Start a task.
        """
        self._tasks.append(task)

    def send_message(self, message: Message):
        self._bus.send(message)

    def start(self, start_dest='start', start_data={}):
        running = []
        out_queue = self._bus.get_out_queue()
        # setup the buses and pass each one a reference to it.
        for task in self._tasks:
            in_queue = self._bus.register_process_queue(task.name())
            queue = (in_queue, out_queue)
            # register handlers
            for message_type in task.message_types():
                self._bus.register_message_handler(
                    message_type, task.name()
                )
            p = Process(target=run_task, args=(queue, task,))
            p.start()
            running.append(p)

        # Need to send a start event to get the ball rolling
        self.send_message(Message(start_dest, start_data))

        # now listen on the buses and pass messages between them.
        while True:
            try:
                self._bus.step()
            except KeyboardInterrupt:
                break

        for task in running:
            task.join()


class SchedulerTask(Task):
    """
    Send messages based on a schedule.
    """

    def setup(self, data):
        self._data = data['sched']

    def send_message(self, message_name):
        msg = Message(message_name, {})
        self.send(msg)

    def on_message(self, message: Message):
        if message.name() != 'start_sched':
            return

        s = schedule.Scheduler()
        for task in self._data:
            rep_time = task['time']
            message_name = task['message_name']
            s.every(rep_time).seconds.do(
                self.send_message,
                message_name=message_name
            )

        while True:
            s.run_pending()
            time.sleep(1)
