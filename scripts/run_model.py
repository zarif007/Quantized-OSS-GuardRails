import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm

# Add parent dir to path so we can import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import LLMGuard, MODEL_CONFIGS
from evaluation.profiling import ModelProfiler, InferenceProfiler
from scripts.data_loader import get_harmbench, get_xstest


def run_model(model_name: str, dataset_name: str, subset: int = None, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)

    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown quantization model: {model_name}")
        sys.exit(1)

    print(f"\n[{model_name.upper()}] Loading dataset: {dataset_name}...")
    if dataset_name.lower() == 'harmbench':
        df = get_harmbench(subset_size=subset)
    elif dataset_name.lower() == 'xstest':
        df = get_xstest(subset_size=subset)
    else:
        print(f"Error: Unknown dataset: {dataset_name}")
        sys.exit(1)

    df['dataset'] = dataset_name.lower()

    # ------------------------------------------------------------------
    # Phase 1: Load model and capture memory
    # ------------------------------------------------------------------
    # ModelProfiler wraps the load call.  On Apple Silicon with n_gpu_layers=-1,
    # model weights live in Metal/unified memory which psutil.rss cannot see.
    # We therefore use the GGUF file size as the weight memory figure and the
    # RSS delta as the CPU-side overhead.  See evaluation/profiling.py for
    # full methodology notes.
    print(f"\n[{model_name.upper()}] Loading model...")
    try:
        # Resolve the model path before load so ModelProfiler can read the file size
        from huggingface_hub import hf_hub_download
        config = MODEL_CONFIGS[model_name]
        model_path = hf_hub_download(
            repo_id=config['repo'],
            filename=config['filename'],
            cache_dir="./models/weights",
            token=False
        )

        mem_profiler = ModelProfiler(model_path=model_path)
        mem_profiler.before_load()
        model = LLMGuard(quant_level=model_name)
        mem_profiler.after_load()

    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        sys.exit(1)

    model_weight_mb, cpu_overhead_mb, total_memory_mb = mem_profiler.memory_stats()
    print(f"\n[{model_name.upper()}] Memory profile:\n{mem_profiler.summary()}")

    # ------------------------------------------------------------------
    # Phase 2: Run inference — measure latency per prompt
    # ------------------------------------------------------------------
    results = []

    print(f"\n[{model_name.upper()}] Running inference on {dataset_name} ({len(df)} samples)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']

        inf_profiler = InferenceProfiler()
        inf_profiler.start()

        try:
            prediction = model.predict(prompt)
        except Exception as e:
            print(f"Error predicting on {model_name}: {e}")
            prediction = "error"

        latency_sec = inf_profiler.stop()

        results.append({
            "prompt": prompt,
            "ground_truth": row['ground_truth'],
            "dataset": row['dataset'],
            "prediction": prediction,
            "latency_sec": latency_sec,
            # Memory columns — constant for all rows of this model run.
            # model_weight_mb: GGUF file size = weight memory in unified/GPU RAM.
            # cpu_overhead_mb: RSS delta captured during model load (CPU-side).
            # total_memory_mb: sum of the above — the figure reported in tables.
            "model_weight_mb": model_weight_mb,
            "cpu_overhead_mb": cpu_overhead_mb,
            "total_memory_mb": total_memory_mb,
        })

    # Save individual model predictions
    output_file = os.path.join(output_dir, f"predictions_{model_name}_{dataset_name.lower()}.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\n[{model_name.upper()}] Saved predictions to {output_file}")

    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific Llama Guard 3 model on a dataset")
    parser.add_argument("--model", required=True, help="Quantization level to test (e.g., q8, q6, q4)")
    parser.add_argument("--dataset", required=True, help="Dataset to evaluate on (e.g., harmbench, xstest)")
    parser.add_argument("--subset", type=int, default=None, help="Number of prompts to evaluate (useful for testing)")

    args = parser.parse_args()
    run_model(model_name=args.model, dataset_name=args.dataset, subset=args.subset)
