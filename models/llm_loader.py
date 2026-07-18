import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

MODEL_CONFIGS = {
    "q8": {
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "filename": "Llama-Guard-3-8B.Q8_0.gguf",
        "size_gb": 8.54
    },
    "q6": {
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "filename": "Llama-Guard-3-8B.Q6_K.gguf",
        "size_gb": 6.59
    },
    "q5": {
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "filename": "Llama-Guard-3-8B.Q5_K_M.gguf",
        "size_gb": 5.73
    },
    "q4": {
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "filename": "Llama-Guard-3-8B.Q4_K_M.gguf",
        "size_gb": 4.92
    },
    "q3": {
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "filename": "Llama-Guard-3-8B.Q3_K_M.gguf",
        "size_gb": 3.93
    }
}

class LLMGuard:
    def __init__(self, quant_level: str, download_dir: str = "./models/weights"):
        if quant_level not in MODEL_CONFIGS:
            raise ValueError(f"Unknown quantization level: {quant_level}")
        
        config = MODEL_CONFIGS[quant_level]
        self.quant_level = quant_level
        
        print(f"[{quant_level}] Ensuring model {config['filename']} is available...")
        # Automatically download if not present, cache it
        self.model_path = hf_hub_download(
            repo_id=config['repo'], 
            filename=config['filename'],
            cache_dir=download_dir,
            token=False
        )
        
        print(f"[{quant_level}] Loading model into llama.cpp ...")
        # n_gpu_layers=-1 delegates all layers to Metal (GPU) on Apple Silicon
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,           
            n_gpu_layers=-1,      
            verbose=False,        
            seed=42              
        )
        
    def predict(self, prompt: str) -> str:
        """
        Llama Guard 3 expects a specific prompt format.
        For simplicity, we pass the raw prompt here. 
        In strict testing, you might want to wrap it in the Llama Guard 3 conversation template.
        """
        # Note: Llama Guard 3 uses the Llama 3 chat template. We wrap the user prompt for evaluation.
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        output = self.llm(
            formatted_prompt,
            max_tokens=64, # Llama Guard usually outputs "safe" or "unsafe\nO1"
            temperature=0.0,
            top_p=1.0,
            stop=["<|eot_id|>"]
        )
        text_response = output["choices"][0]["text"].strip().lower()
        
        if text_response.startswith("unsafe"):
            return "unsafe"
        return "safe"
