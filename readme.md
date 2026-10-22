# 🐍 Mamba-1.4B Edge-Computing Multi-Persona Router

This repository contains the final capstone implementation of a locally-hosted, edge-optimized AI routing architecture. It leverages a customized **Mamba-1.4B** state-space model capable of dynamically hot-swapping between specialized domain expert personas (LoRA adapters) in real-time on consumer-grade CPU hardware.

## 🚀 Key Architectural Achievements

* **Surgical Per-Channel Int8 Quantization:** Successfully compressed the 5.25 GB Mamba-1.4B FP32 footprint down to **0.64 GB**, enabling fluid execution on a 16GB RAM laptop without memory swapping. Bypassed PEFT wrapper limitations by selectively targeting `nn.Linear` layers with `default_per_channel_weight_observer` to eliminate mathematical activation noise and preserve logic generation (e.g., complex SQL queries).
* **Dynamic Persona Hot-Swapping:** Implemented seamless sub-second switching between PEFT (LoRA) adapters. The model dynamically adopts one of three expert brains:
  * 💻 **Software Engineer:** Code generation and algorithm analysis.
  * 🩺 **Clinical Researcher:** Medical diagnostics and pharmacological knowledge.
  * ⚖️ **Legal Counsel:** Constitutional law and Indian penal code expertise.
* **Custom Attention Injections:** Grafted a hybrid `FastCustomBlock` into specific layers (10, 20, 30, 40) of the Mamba backbone to stabilize reasoning tasks.
* **Blazing CPU Inference:** Optimized thread utilization and applied greedy decoding (`repetition_penalty=1.15`) to eliminate infinite Mamba looping, achieving generation speeds exceeding **10.5 tokens/second** on an Intel Core Ultra 5.

## 📁 Repository Structure

* `web_ui.py`: The main Gradio production application. Features the complete quantized inference engine, dynamic persona routing, and latency telemetry.
* `evaluate_personas.py`: The scientific benchmarking script used to evaluate the model across FP32 and Int8 precisions.
* `eval_results/`: Contains the raw execution data (`int8.json`) and the generated metrics (`comparison_2.md` and `grading.csv`) proving a 1.33x speedup and intact factual fidelity.
* `version_2_artifacts/`: Directory for the base PEFT adapters and hybrid attention weights.
* `persona_adapter_*/`: Domain-specific LoRA weights for the distinct expert personas.

## 🛠️ Usage

**1. Launch the Web UI:**
To boot the production interface with surgical Int8 quantization and real-time hot-swapping:
```powershell
python web_ui.py
```
*The app will automatically launch in your browser at `http://127.0.0.1:7860`.*

**2. Run the Evaluation Suite:**
To independently verify the benchmark metrics and quantization fidelity:
```powershell
# Run the Int8 benchmark with loop-killing penalties
python evaluate_personas.py run --precision int8 --repetition-penalty 1.15 --fresh

# Generate the markdown report
python evaluate_personas.py report
```

## 📊 Evaluation Highlights

* **Speedup:** 1.33x inference speed increase (from ~5.9 tok/s in FP32 to >7.8 tok/s in Int8 evaluation, peaking at >10.5 tok/s in dynamic CPU production).
* **Fidelity:** Per-channel quantization stabilized logic outputs, allowing the Int8 model to maintain an ~85% teacher-forced agreement with the FP32 baseline while preventing code hallucinations.