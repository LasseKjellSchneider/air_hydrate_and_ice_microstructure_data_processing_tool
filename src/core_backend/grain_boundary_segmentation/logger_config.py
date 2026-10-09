import logging
import time
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

def setup_logging(log_dir):
    """
    Sets up the basic Python logging (INFO, ERROR, etc.) to console and a general log file.
    """
    if log_dir is None:
        return
        
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "grain_boundary_pipeline.log"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class DebugProfiler:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.log_file_path = None
        self.log_file = None
        self.session_start_time = time.time()  # Track elapsed time across the full session.
        
        if self.output_dir is not None:
            if self.output_dir.suffix == ".txt":
                self.log_file_path = self.output_dir
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.log_file_path = self.output_dir / "debug_grain_boundary.txt"
                
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self.start_time = None
        self.step_name = None
        self.initial_ram = 0.0

        if self.log_file_path is not None:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] \n")
                    f.write("=======================================================\n")
                    f.write(f"=== PROFILING SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write("=======================================================\n")
            except Exception:
                pass

    def _get_ram_gb(self):
        if self.process is not None:
            try:
                return self.process.memory_info().rss / (1024 ** 3)
            except Exception:
                return 0.0
        return 0.0

    def start_step(self, step_name):
        self.step_name = step_name
        self.start_time = time.time()
        self.initial_ram = self._get_ram_gb()
        
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: {self.step_name} | Current RAM: {self.initial_ram:.2f} GB\n"
        self._write_msg(msg)
        return self.start_time, self.initial_ram

    def end_step(self, step_name=None):
        if step_name and step_name != self.step_name:
            self.step_name = step_name
            
        elapsed = time.time() - (self.start_time or time.time())
        current_ram = self._get_ram_gb()
        ram_diff = current_ram - self.initial_ram
        
        sign = "+" if ram_diff >= 0 else ""
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] --> FINISHED: {self.step_name} | Took: {elapsed:.2f}s | RAM Diff: {sign}{ram_diff:.2f} GB | Current RAM: {current_ram:.2f} GB\n"
        self._write_msg(msg)

    def log_step(self, step_name, start_time=None, start_ram=None):
        """Unified wrapper supporting the 2-parameter signature used in sub-modules."""
        if start_time is None:
            return self.start_step(step_name)
        else:
            self.start_time = start_time
            self.initial_ram = start_ram if start_ram is not None else self._get_ram_gb()
            self.end_step(step_name)

    def log_info(self, message):
        msg = f"[{datetime.now().strftime('%H:%M:%S')}]        -> {message}\n"
        self._write_msg(msg)

    def log_msg(self, message):
        """Alias for log_info."""
        self.log_info(message)

    def _write_msg(self, msg):
        if self.log_file_path is not None:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass

    def end_session(self):
        # Record the total duration when the profiling session ends.
        total_time = time.time() - self.session_start_time
        msg = f"\n[{datetime.now().strftime('%H:%M:%S')}] === PROFILING SESSION ENDED (Total Time: {total_time:.2f}s) ===\n\n"
        self._write_msg(msg)

    def close(self):
        """Alias for end_session."""
        self.end_session()