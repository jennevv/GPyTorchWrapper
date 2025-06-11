import time


class Timer:
    def __init__(self, logger):
        self.t1 = (0., 0.)
        self.t2 = (0., 0.)
        self.logger = logger
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
    def log_timings(self, step_name: str):
        self.logger.info(f"------------------------------------------\nTIMINGS FOR {step_name.upper()}")
        self.logger.info(f"Real time: {self.real_time():.2f} seconds")
        self.logger.info(f"CPU time: {self.cpu_time():.2f} seconds")
        self.logger.info("------------------------------------------\n")
