"""
profiling.py — Memory and latency profiling for llama.cpp + Apple Silicon.

Two-phase measurement strategy
-------------------------------
Phase 1 — Model load (ModelProfiler):
  On Apple Silicon, `n_gpu_layers=-1` offloads ALL model weights to the Metal
  GPU via unified memory.  `psutil.rss` only reflects CPU-side anonymous pages
  and is therefore blind to the weights.  The authoritative memory figure is the
  GGUF file size on disk, which equals the weight tensor memory transferred to
  unified memory.  We additionally capture the CPU-side RSS delta around the
  load call to account for KV-cache allocation, llama.cpp bookkeeping, and Python
  interpreter overhead.

  Reported fields (stored once per model run, constant across prompts):
    model_weight_mb   — GGUF file size in MiB  (weight memory in unified/GPU RAM)
    cpu_overhead_mb   — RSS(post-load) – RSS(pre-load)  (CPU-side runtime cost)
    total_memory_mb   — model_weight_mb + cpu_overhead_mb

Phase 2 — Per-prompt inference (InferenceProfiler):
  Measures wall-clock latency only.  Per-inference RSS fluctuations are ≤50 MB
  (KV-cache reuse) and are not reported as a separate memory figure to avoid
  misleading reviewers.
"""

import os
import time
import psutil


class ModelProfiler:
    """
    Wraps model load to capture accurate memory usage.

    Usage
    -----
    profiler = ModelProfiler(model_path)
    profiler.before_load()
    model = LLMGuard(...)          # the actual load call
    profiler.after_load()
    weight_mb, overhead_mb, total_mb = profiler.memory_stats()
    """

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Absolute or relative path to the GGUF weight file.  Used to read
            the on-disk size, which equals the unified-memory footprint for a
            full Metal offload (n_gpu_layers=-1).
        """
        self._model_path = model_path
        self._process = psutil.Process(os.getpid())

        # Populated by before_load() / after_load()
        self._rss_before_mb: float = 0.0
        self._rss_after_mb: float = 0.0

        # Public results (set after after_load())
        self.model_weight_mb: float = 0.0
        self.cpu_overhead_mb: float = 0.0
        self.total_memory_mb: float = 0.0

    # ------------------------------------------------------------------
    # Measurement API
    # ------------------------------------------------------------------

    def before_load(self) -> None:
        """Call immediately before the model load statement."""
        self._rss_before_mb = self._process.memory_info().rss / (1024 * 1024)

    def after_load(self) -> None:
        """Call immediately after the model load statement completes."""
        self._rss_after_mb = self._process.memory_info().rss / (1024 * 1024)

        # Weight memory = GGUF file size (authoritative for Metal GPU offload)
        self.model_weight_mb = os.path.getsize(self._model_path) / (1024 * 1024)

        # CPU-side overhead = RSS delta; clamp to 0 in case of GC noise
        self.cpu_overhead_mb = max(0.0, self._rss_after_mb - self._rss_before_mb)

        self.total_memory_mb = self.model_weight_mb + self.cpu_overhead_mb

    def memory_stats(self) -> tuple:
        """
        Returns
        -------
        (model_weight_mb, cpu_overhead_mb, total_memory_mb)
        """
        return self.model_weight_mb, self.cpu_overhead_mb, self.total_memory_mb

    def summary(self) -> str:
        return (
            f"  Model weights (GGUF / unified mem): {self.model_weight_mb:>8.1f} MB\n"
            f"  CPU-side RSS overhead (post-load):  {self.cpu_overhead_mb:>8.1f} MB\n"
            f"  Total reported memory:              {self.total_memory_mb:>8.1f} MB"
        )


class InferenceProfiler:
    """
    Lightweight per-prompt profiler that measures wall-clock latency only.

    Memory is NOT tracked here — it is constant after model load (weights stay
    in unified memory, KV cache is reused).  Mixing per-inference RSS noise
    into the reported memory figure would be misleading.
    """

    def __init__(self):
        self._t0: float = 0.0
        self.latency_sec: float = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Returns wall-clock latency in seconds."""
        self.latency_sec = time.perf_counter() - self._t0
        return self.latency_sec
