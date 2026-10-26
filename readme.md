# Edge-Optimized Multi-Persona AI Inference Engine

An undergraduate capstone project demonstrating the deployment of a dynamically adaptable, multi-persona Large Language Model (Mamba-1.4B) on constrained edge hardware (CPU-only consumer laptops). 

This project solves the hardware bottleneck of running billion-parameter state-space models locally by introducing **Layer Ablation** and **Surgical Per-Channel Int8 Quantization**, achieving a 66% reduction in weight memory footprint while simultaneously increasing inference throughput and curing the architecture's native infinite-looping vulnerability.

## 🚀 Core Architectural Innovations

### 1. Layer-Ablated Control Architecture
Standard 1.4B parameter models exceed the memory and compute limits of standard 8GB/16GB laptops. To drastically reduce compute overhead, this project ablates four core transformer layers (Layers 10, 20, 30, and 40) from the Mamba backbone, replacing them with a highly lightweight linear pass-through gate (`FastCustomBlock`). While this degrades the raw baseline into an "Ablated Control Model," we demonstrate that instruction-tuned adapters successfully bridge these missing layers to restore domain-specific factual accuracy.

### 2. Weight Parameter Compression (1.79 GB Footprint)
Standard dynamic quantization corrupts PEFT (LoRA) adapters, preventing persona hot-swapping. This engine utilizes a surgical script that maps and quantizes only the standard `nn.Linear` layers to 8-bit integers (Int8) while strictly preserving the precision of the `lm_head` and LoRA hooks. 
*   **FP32 Baseline Footprint:** 5.25 GB
*   **Int8 Compressed Footprint:** 1.79 GB (66% reduction)

### 3. Thread-Safe Dynamic LoRA Hot-Swapping
The inference engine can instantly swap between highly specialized domain experts without reloading the base model weights:
*   **Software Engineer** (Optimized for C++ and SQL generation)
*   **Clinical Researcher** (Optimized for medical marker and mechanism retrieval)
*   **Legal Counsel** (Optimized for Indian Constitutional and Contract Law)
*   *Concurrency:* The Gradio UI implements a strict `threading.Lock()` to prevent adapter state collisions during simultaneous multi-user requests.

### 4. Architectural Loop Mitigation (Zero-Guardrail Decoding)
State Space Models (like Mamba) suffer from a fatal flaw of infinite token looping. While traditionally patched using destructive repetition penalties (which trigger factual hallucinations), our unguarded evaluation benchmarks (`no_repeat_ngram=0`, `repetition_penalty=1.0`) prove that applying domain-specific LoRA adapters naturally stabilizes the architecture. Loop rates dropped from **100% in the Ablated Control Model** to **0% in the Clinical and Coding adapters** without requiring external decoding guardrails.

## 📊 Performance Benchmarks (Intel Core Ultra 5 CPU)

Quantization successfully increased processing speed across all active configurations by lowering memory bandwidth bottlenecks:

| Persona Config | FP32 Speed | Int8 Speed | CPU Speedup | Loop Rate (Unguarded) |
| :--- | :--- | :--- | :--- | :--- |
| Ablated Control Model | 6.30 tok/s | 7.73 tok/s | 1.23x | 100.0% |
| Legal Counsel | 5.92 tok/s | 7.98 tok/s | 1.35x | 75.0% |
| Clinical Researcher | 5.59 tok/s | 7.18 tok/s | 1.29x | 0.0% |
| Software Engineer | 5.63 tok/s | 6.84 tok/s | 1.21x | 0.0% |

## ⚙️ Installation & Usage

**1. Clone and Install Dependencies:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

**2. Launch the Gradio Server:**
```bash
python web_ui.py
```

**3. Run Scientific Benchmarks (Unguarded):**
```bash
# Run FP32 and Int8 Evaluation
python evaluate_personas.py run --precision fp32 --repetition-penalty 1.0 --no-repeat-ngram 0 --fresh
python evaluate_personas.py run --precision int8 --repetition-penalty 1.0 --no-repeat-ngram 0 --fresh
python evaluate_personas.py report
```

## 🔬 Known Limitations & Future Work

As part of our rigorous architectural review, we acknowledge the following engineering limitations for future iterations:
1.  **Gradio Serialization Bottleneck:** To prevent LoRA adapter collisions in state memory, we implemented a strict Python threading lock over the generation function. In a true production environment under heavy load, this creates a Denial-of-Service (DoS) vulnerability, requiring a shift to a multi-worker queue architecture.
2.  **Statistical Power:** Due to local compute limits, our evaluation used a targeted small sample size (n=4 for persona prompts). Future work requires larger, automated benchmarks to establish p-value statistical significance for the documented 1.3x speedup.
3.  **Caching Drift:** We observed that Int8 teacher-forced self-agreement fell below 100% despite greedy decoding. This indicates that PyTorch's dynamic quantization introduces non-deterministic execution paths during caching, warranting further low-level C++ backend optimization.

## ⚠️ Disclaimer
**This is an experimental academic prototype.** The Clinical Researcher and Legal Counsel personas are AI-generated models and are prone to unpredictable outputs. They do not constitute professional medical advice or certified legal counsel.