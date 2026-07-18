import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm

# Add parent dir to path so we can import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import LLMGuard, MODEL_CONFIGS
from evaluation.profiling import Profiler
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
    
    print(f"\n[{model_name.upper()}] Loading model...")
    try:
        model = LLMGuard(quant_level=model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        sys.exit(1)
        
    results = []
    
    print(f"\n[{model_name.upper()}] Running inference on {dataset_name} ({len(df)} samples)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        
        profiler = Profiler()
        profiler.start()
        
        try:
            prediction = model.predict(prompt)
        except Exception as e:
            print(f"Error predicting on {model_name}: {e}")
            prediction = "error"
            
        latency_sec, peak_mem_mb = profiler.stop()
        
        results.append({
            "prompt": prompt,
            "ground_truth": row['ground_truth'],
            "dataset": row['dataset'],
            "prediction": prediction,
            "latency_sec": latency_sec,
            "peak_memory_mb": peak_mem_mb
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
