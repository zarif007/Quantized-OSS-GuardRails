import time
import psutil
import os
from threading import Thread

class Profiler:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.latency_sec = 0
        
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0
        self._monitoring = False
        self._monitor_thread = None

    def _monitor_memory(self):
        while self._monitoring:
            # RSS includes mmap and typical RAM usage
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            if mem_mb > self.peak_memory_mb:
                self.peak_memory_mb = mem_mb
            time.sleep(0.01) # Sample every 10ms

    def start(self):
        self.start_time = time.time()
        self.peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        self._monitoring = True
        self._monitor_thread = Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self.end_time = time.time()
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            
        self.latency_sec = self.end_time - self.start_time
        return self.latency_sec, self.peak_memory_mb
