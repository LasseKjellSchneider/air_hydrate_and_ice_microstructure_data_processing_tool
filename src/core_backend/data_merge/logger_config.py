import logging
import psutil
import time
import os

class PipelineLogger:
    def __init__(self, log_file="debug_data_merge.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("IcePipeline")
        self.logger.setLevel(logging.DEBUG)
        
        # Format: [Zeit] [Level] [RAM-Usage] Nachricht
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [RAM: %(ram_usage)s MB] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File Handler
        fh = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def _get_extra(self):
        """Ermittelt den aktuellen RAM-Verbrauch des Prozesses in MB."""
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        return {'ram_usage': f"{ram_mb:.1f}"}

    def info(self, msg):
        self.logger.info(msg, extra=self._get_extra())

    def warning(self, msg):
        self.logger.warning(msg, extra=self._get_extra())

    def error(self, msg):
        self.logger.error(msg, extra=self._get_extra())

    def debug(self, msg):
        self.logger.debug(msg, extra=self._get_extra())

    # Timing helper used by pipeline stages.
class TimerContext:
    """Kontext-Manager, um die Dauer von Code-Blöcken zu messen."""
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.op_name = operation_name

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"START: {self.op_name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"DONE:  {self.op_name} (Dauer: {elapsed:.3f} Sekunden)")
        else:
            self.logger.error(f"FAIL:  {self.op_name} nach {elapsed:.3f} Sekunden mit Fehler: {exc_val}")