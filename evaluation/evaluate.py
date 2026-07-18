import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_metrics
from models.llm_loader import MODEL_CONFIGS

def aggregate_predictions(results_dir="results"):
    """Loads all prediction CSVs and groups them by model"""
    all_files = glob.glob(os.path.join(results_dir, "predictions_*.csv"))
    if not all_files:
        print(f"No prediction files found in {results_dir}.")
        return None
        
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        # Extract model from filename: predictions_{model}_{dataset}.csv
        basename = os.path.basename(f)
        parts = basename.replace(".csv", "").split("_")
        model_name = parts[1]
        df['model'] = model_name
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def compute_all_metrics(combined_df):
    """Computes metrics per model and returns a summary dataframe"""
    all_metrics = []
    models = combined_df['model'].unique()
    
    for m in models:
        model_df = combined_df[combined_df['model'] == m]
        metrics = calculate_metrics(model_df)
        metrics['model'] = m
        # Add model size explicitly from config
        metrics['size_gb'] = MODEL_CONFIGS.get(m, {}).get('size_gb', 0.0)
        all_metrics.append(metrics)
        
    summary_df = pd.DataFrame(all_metrics)
    
    # Sort by size or conceptually q8 -> q3
    quant_order = {"q8": 1, "q6": 2, "q5": 3, "q4": 4, "q3": 5}
    summary_df['order'] = summary_df['model'].map(lambda x: quant_order.get(x, 99))
    summary_df = summary_df.sort_values(by='order').drop(columns=['order'])
    
    return summary_df

def generate_tables(summary_df, output_dir="paper/tables"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Table 1: Model Information | Quantization | File Size | Weight Mem | CPU Overhead | Total Mem |
    # peak_memory_gb = model_weight_gb + cpu_overhead_gb (total reported memory)
    t1_cols = ['model', 'size_gb', 'model_weight_gb', 'cpu_overhead_gb', 'peak_memory_gb']
    # Gracefully drop columns missing from old runs
    t1_cols = [c for c in t1_cols if c in summary_df.columns]
    t1 = summary_df[t1_cols].copy()
    t1.columns = [{
        'model': 'Quantization',
        'size_gb': 'File Size (GB)',
        'model_weight_gb': 'Weight Memory (GB)',
        'cpu_overhead_gb': 'CPU Overhead (GB)',
        'peak_memory_gb': 'Total Memory (GB)',
    }.get(c, c) for c in t1_cols]
    t1.to_csv(os.path.join(output_dir, "table1_model_info.csv"), index=False)
    
    # Table 2: Safety | Model | Safety Rate | Recall | F1 |
    t2 = summary_df[['model', 'safety_rate', 'recall', 'safety_f1']].copy()
    t2.columns = ['Model', 'Safety Rate', 'Recall', 'F1']
    t2.to_csv(os.path.join(output_dir, "table2_safety.csv"), index=False)
    
    # Table 3: Usefulness | Model | False Positive | Accuracy |
    t3 = summary_df[['model', 'false_positive_rate', 'accuracy']].copy()
    t3.columns = ['Model', 'False Positive Rate', 'Accuracy']
    t3.to_csv(os.path.join(output_dir, "table3_usefulness.csv"), index=False)
    
    # Table 4: Efficiency | Model | Latency | Total Memory | Throughput |
    # Total Memory = model weight (GGUF size) + CPU-side RSS overhead at load time
    t4_cols = ['model', 'avg_latency_sec', 'model_weight_gb', 'cpu_overhead_gb', 'peak_memory_gb', 'throughput']
    t4_cols = [c for c in t4_cols if c in summary_df.columns]
    t4 = summary_df[t4_cols].copy()
    t4.columns = [{
        'model': 'Model',
        'avg_latency_sec': 'Latency (s)',
        'model_weight_gb': 'Weight Memory (GB)',
        'cpu_overhead_gb': 'CPU Overhead (GB)',
        'peak_memory_gb': 'Total Memory (GB)',
        'throughput': 'Throughput (prompts/s)',
    }.get(c, c) for c in t4_cols]
    t4.to_csv(os.path.join(output_dir, "table4_efficiency.csv"), index=False)
    
    # Table 5: Overall
    t5 = summary_df.copy()
    t5.to_csv(os.path.join(output_dir, "table5_overall.csv"), index=False)
    
    print(f"Tables saved to {output_dir}")

def generate_plots(summary_df, output_dir="paper/figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Bar chart: Safety
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x="model", y="safety_f1", palette="Blues_d")
    plt.title("Safety F1 across models")
    plt.ylabel("Safety F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_bar_safety.png"))
    plt.close()
    
    # 2. Bar chart: Latency
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x="model", y="avg_latency_sec", palette="Reds_d")
    plt.title("Average Latency across models")
    plt.ylabel("Latency (sec/prompt)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_bar_latency.png"))
    plt.close()
    
    # 3. Line plot: Quantization vs Safety
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary_df, x="model", y="safety_f1", marker="o", color="blue")
    plt.title("Quantization vs Safety (F1)")
    plt.ylabel("Safety F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_line_safety.png"))
    plt.close()
    
    # 4. Line plot: Quantization vs Memory
    # peak_memory_gb = GGUF weight size + CPU-side RSS overhead (Metal unified memory aware)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary_df, x="model", y="peak_memory_gb", marker="o", color="green")
    plt.title("Quantization vs Total Memory (Weights + CPU Overhead)")
    plt.ylabel("Total Memory (GB)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_line_memory.png"))
    plt.close()
    
    # 5. Line plot: Quantization vs Latency
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary_df, x="model", y="avg_latency_sec", marker="o", color="red")
    plt.title("Quantization vs Latency")
    plt.ylabel("Latency (sec/prompt)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_line_latency.png"))
    plt.close()
    
    # Optional 6: Radar Chart
    try:
        categories = ['Safety F1', 'Usefulness F1', 'Throughput', 'Inv Memory']
        N = len(categories)
        
        # Normalize variables for radar chart (0 to 1 scale)
        radar_df = summary_df[['model', 'safety_f1', 'usefulness_f1', 'throughput', 'peak_memory_gb']].copy()
        
        # Normalize throughput
        t_max = radar_df['throughput'].max()
        radar_df['Throughput'] = radar_df['throughput'] / t_max if t_max > 0 else 0
        
        # Invert and normalize memory (lower memory = higher score)
        m_max = radar_df['peak_memory_gb'].max()
        radar_df['Inv Memory'] = 1.0 - (radar_df['peak_memory_gb'] / m_max) if m_max > 0 else 0
        
        radar_df = radar_df.rename(columns={'safety_f1': 'Safety F1', 'usefulness_f1': 'Usefulness F1'})
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        for i, row in radar_df.iterrows():
            values = row[['Safety F1', 'Usefulness F1', 'Throughput', 'Inv Memory']].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['model'])
            ax.fill(angles, values, alpha=0.1)
            
        plt.xticks(angles[:-1], categories)
        plt.title('Performance Radar Chart')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig6_radar.png"))
        plt.close()
    except Exception as e:
        print(f"Skipping radar chart: {e}")
    
    print(f"Figures saved to {output_dir}")

if __name__ == "__main__":
    combined_df = aggregate_predictions()
    if combined_df is not None and not combined_df.empty:
        summary_df = compute_all_metrics(combined_df)
        
        # Save overarching summary to CSV
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv("results/summary_metrics.csv", index=False)
        print("Summary metrics saved to results/summary_metrics.csv")
        
        generate_tables(summary_df)
        generate_plots(summary_df)
    else:
        print("No data available to evaluate.")
