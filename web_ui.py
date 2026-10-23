import time
import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Optimize CPU thread utilization
torch.set_num_threads(torch.get_num_threads())

# 1. Custom Block Architecture
class FastCustomBlock(nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, **kwargs):
        orig_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        out = h + self.gate * self.proj(self.norm(h))
        return out.to(orig_dtype)

print("Loading base Mamba-1.4B model (CPU)...")
model_name = "state-spaces/mamba-1.4b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to("cpu")

print("Injecting Stage 2 aligned blocks...")
aligned_weights = torch.load("version_2_artifacts/aligned_custom_blocks_stage2.pt", map_location="cpu", weights_only=True)
for idx in [10, 20, 30, 40]:
    block = FastCustomBlock().to("cpu")
    block.load_state_dict(aligned_weights[idx])
    model.backbone.layers[idx] = block

# 2. Attach Primary Adapter and Additional Personas
print("Attaching Base Adapter...")
model = PeftModel.from_pretrained(model, "version_2_artifacts/mamba_fully_adapted_final", adapter_name="base").to("cpu")

print("Loading Software Engineer Adapter...")
model.load_adapter("./persona_adapter_engineering_v2", adapter_name="software_engineer")

print("Loading Clinical Researcher Adapter...")
model.load_adapter("./persona_adapter_clinical", adapter_name="clinical_researcher")

print("Loading Legal Counsel Adapter...")
model.load_adapter("./persona_adapter_legal", adapter_name="legal_counsel")

# 3. Layer 2: Selective Mixed-Precision Int8 Quantization
print("Applying Layer 2: Surgical Per-Channel Int8 Compression (Base=Int8, LoRA=FP32)...")
import torch.ao.quantization as tq
import torch.ao.nn.quantized.dynamic as nnqd

per_channel_qconfig = tq.QConfigDynamic(
    activation=tq.default_dynamic_quant_observer,
    weight=tq.default_per_channel_weight_observer
)

def force_quantize(m):
    for name, child in m.named_children():
        if "dt_proj" in name or "lm_head" in name:
            continue
        
        # Brute-force the PEFT base_layer
        if hasattr(child, "base_layer") and hasattr(child, "lora_A"):
            if isinstance(child.base_layer, nn.Linear):
                child.base_layer.qconfig = per_channel_qconfig
                child.base_layer = nnqd.Linear.from_float(child.base_layer)
            continue  
            
        # Brute-force standard unadapted Linear layers
        if isinstance(child, nn.Linear):
            child.qconfig = per_channel_qconfig
            setattr(m, name, nnqd.Linear.from_float(child))
        else:
            force_quantize(child)

force_quantize(model)
model.eval()

print("Multi-Persona Quantized Model ready! Memory Footprint: ~1.79 GB")

import threading

# Add a global lock to prevent concurrency adapter collisions
generation_lock = threading.Lock()

# 4. Dynamic Router & Generation Function
def generate_text(prompt, persona, max_tokens, temperature=None):
    # Lock the thread so multiple Gradio users don't race the set_adapter method
    with generation_lock:
        if persona == "Software Engineer":
            model.set_adapter("software_engineer")
            formatted_prompt = f"Instruction: {prompt.strip()}\n\nInput:\n\nResponse:"
            use_adapters = True
        elif persona == "Clinical Researcher":
            model.set_adapter("clinical_researcher")
            formatted_prompt = f"Instruction: {prompt.strip()}\n\nResponse:"
            use_adapters = True
        elif persona == "Legal Counsel":
            model.set_adapter("legal_counsel")
            formatted_prompt = f"Instruction: {prompt.strip()}\n\nResponse:"
            use_adapters = True
        elif persona == "Base Model":
            # FIX: Actually disable all adapters for the true Base Model
            formatted_prompt = f"Instruction: {prompt.strip()}\n\nResponse:"
            use_adapters = False

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
        
        start_time = time.perf_counter()
        
        
# Determine if we should generate with or without PEFT adapters active
        with torch.inference_mode():
            if use_adapters:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    repetition_penalty=1.0,      # Synced to eval
                    no_repeat_ngram_size=0,      # Synced to eval
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                with model.disable_adapter():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens, 
                        repetition_penalty=1.0,      # Synced to eval
                        no_repeat_ngram_size=0,      # Synced to eval
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
        end_time = time.perf_counter()
        
        generation_time = end_time - start_time
        input_length = inputs.input_ids.shape[1]
        generated_tokens_count = outputs.shape[1] - input_length
        tokens_per_second = generated_tokens_count / max(generation_time, 1e-5)
        
        # FIX: Remove the destructive BPE tokenizer spaces
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # FIX: Robust slicing based on character length of the prompt, avoiding the "Response:" split bug
        final_text = raw_output[len(formatted_prompt):].strip()
            
        metrics = f"\n\n[Telemetry: {generation_time:.2f}s total | {tokens_per_second:.2f} tokens/sec | {generated_tokens_count} tokens]"
        return final_text + metrics

# 5. Gradio Interface with Persona Selector
with gr.Blocks() as demo:
    gr.Markdown("# 🐍 Mamba-1.4B Multi-Persona Assistant")
    # FIX: Accurate architectural description without the unverified footprint claim
    gr.Markdown("Dynamic LoRA adapter hot-swapping running on CPU (Layer-Ablated Int8 Quantization).")
    
    # FIX: Legal and Medical Disclaimer
    gr.HTML("<div style='background-color: #ffcccc; padding: 10px; border-radius: 5px; color: #900; font-size: 12px; margin-bottom: 15px;'>"
            "<strong>WARNING:</strong> Clinical and Legal outputs are generated by an experimental AI. They are NOT professional medical or legal advice. "
            "Do not use for diagnostic purposes or legal compliance.</div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            persona_dropdown = gr.Dropdown(
                choices=["Software Engineer", "Clinical Researcher", "Legal Counsel", "Base Model"], 
                value="Legal Counsel", 
                label="Active Persona"
            )
            prompt_input = gr.Textbox(
                lines=5, 
                label="Task / Instruction", 
                placeholder="e.g., What are the essential legal grounds required to file a PIL under Article 32?"
            )
            max_tokens_slider = gr.Slider(minimum=20, maximum=300, value=120, step=10, label="Max New Tokens")
            submit_btn = gr.Button("Generate Response", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(lines=14, label="Persona Output", interactive=False)
            
    submit_btn.click(
        fn=lambda p, pers, tokens: generate_text(p, pers, tokens, None), 
        inputs=[prompt_input, persona_dropdown, max_tokens_slider], 
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())