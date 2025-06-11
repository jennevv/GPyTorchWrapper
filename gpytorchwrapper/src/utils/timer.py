import logging
import time

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self, step_name: str):
        self.t1 = (0.0, 0.0)
        self.t2 = (0.0, 0.0)
        self.step_name = step_name

    @staticmethod
    def _time() -> tuple[float, float]:
        return time.perf_counter(), time.process_time()

    def set_init_time(self):
        self.t1 = self._time()

    def set_final_time(self):
        self.t2 = self._time()

    def real_time(self):
        return self.t2[0] - self.t1[0]

    def cpu_time(self):
        return self.t2[1] - self.t1[1]

    def log_timings(self):
        logger.info(
            f"------------------------------------------\nTIMINGS FOR {self.step_name.upper()}"
        )
        logger.info(f"Real time: {self.real_time():.2f} seconds")
        logger.info(f"CPU time: {self.cpu_time():.2f} seconds")
        logger.info("------------------------------------------\n")
