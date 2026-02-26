from multiprocessing import Process, Queue
from nnunet_serve.logging_utils import get_logger
from nnunet_serve.seg_writers import export_predictions

logger = get_logger(__name__)


class ProcessPool:
    """
    A process pool for parallelizing function calls.
    """

    def __init__(self, num_processes: int):
        """
        Initialize the process pool.

        Args:
            num_processes (int): Number of processes to use.
        """
        self.num_processes = num_processes
        self.processes = []
        self.input_queue = Queue()
        self.output_queue = Queue()

        for i in range(self.num_processes):
            logger.info("Starting process %d", i)
            p = Process(
                target=self.run_case, args=(self.input_queue, self.output_queue)
            )
            p.start()
            self.processes.append(p)

    def fn(self, *args, **kwargs):
        """
        Placeholder for the function to be executed in the pool.

        This method must be overridden by subclasses to implement specific
        processing logic.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """

        raise NotImplementedError

    def run_case(self, input_queue: Queue, output_queue: Queue):
        """
        Worker loop that processes cases from the input queue.

        Args:
            input_queue (Queue): Queue containing case dictionaries with
                'identifier', 'args', and 'kwargs'.
            output_queue (Queue): Queue for storing the results of processed cases.
        """
        while True:
            case = input_queue.get()
            if case is None:
                break
            logger.info("Processing case %s", case["identifier"])
            output_queue.put(
                self.fn(
                    case["identifier"],
                    *case["args"],
                    **case["kwargs"],
                )
            )
            logger.info("Finished case %s", case["identifier"])

    def put(self, identifier: int, args: tuple, kwargs: dict):
        """
        Put a case to be processed.

        Args:
            identifier (int): Identifier of the case.
            args (tuple): Positional arguments for the function.
            kwargs (dict): Keyword arguments for the function.
        """
        self.input_queue.put(
            {"identifier": identifier, "args": args, "kwargs": kwargs}
        )

    def get(self):
        """
        Get the result of a processed case.

        Returns:
            Any: Result of the function.
        """
        return self.output_queue.get()

    def close(self):
        """
        Close the process pool.
        """
        for _ in range(self.num_processes):
            self.input_queue.put(None)
        for p in self.processes:
            logger.debug("Joining process %d", p.pid)
            p.join()
        logger.debug("Closing queues")
        self.input_queue.close()
        self.output_queue.close()
        logger.debug("Process pool closed")

    def __del__(self):
        """
        Ensures the process pool is closed when the instance is deleted."""
        try:
            self.close()
        except ValueError:
            logger.debug("Process pool already closed")


class WritingProcessPool(ProcessPool):
    """
    A process pool for parallelizing file writing operations.
    """

    def fn(self, identifier: int, *args, **kwargs):
        """
        Executes export_predictions for a given case.

        Args:
            identifier (int): Identifier of the case.
            *args: Positional arguments for export_predictions.
            **kwargs: Keyword arguments for export_predictions.

        Returns:
            tuple: (identifier, result_of_export_predictions)
        """
        return identifier, export_predictions(*args, **kwargs)
