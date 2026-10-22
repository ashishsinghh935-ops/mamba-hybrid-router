import time
import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Optimize CPU thread utilization
torch.set_num_threads(8)

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
# Quantize all heavy base Mamba linear layers while keeping LoRA adapters in FP32
print("Applying Layer 2: Selective Mixed-Precision Int8 Quantization (Base=Int8, LoRA=FP32)...")
def quantize_base_layers_only(module):
    for name, child in module.named_children():
        if name in ["lora_A", "lora_B"]:
            continue  # Keep LoRA adapter weights in FP32 for PEFT compatibility
        if isinstance(child, nn.Linear):
            setattr(
                module, 
                name, 
                torch.quantization.quantize_dynamic(child, {nn.Linear}, dtype=torch.qint8)
            )
        else:
            quantize_base_layers_only(child)

quantize_base_layers_only(model)
model.eval()

print("Multi-Persona Quantized Model ready!")

# 4. Dynamic Router & Generation Function
def generate_text(prompt, persona, max_tokens, temperature):
    if persona == "Software Engineer":
        model.set_adapter("software_engineer")
        formatted_prompt = f"Instruction: {prompt.strip()}\n\nInput:\n\nResponse:"
    elif persona == "Clinical Researcher":
        model.set_adapter("clinical_researcher")
        formatted_prompt = f"Instruction: {prompt.strip()}\n\nResponse:"
    elif persona == "Legal Counsel":
        model.set_adapter("legal_counsel")
        formatted_prompt = f"Instruction: {prompt.strip()}\n\nResponse:"
    else:
        model.set_adapter("base")
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
    
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=temperature, 
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.perf_counter()
    
    generation_time = end_time - start_time
    input_length = inputs.input_ids.shape[1]
    generated_tokens_count = outputs.shape[1] - input_length
    tokens_per_second = generated_tokens_count / max(generation_time, 1e-5)
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Response:" in raw_output:
        final_text = raw_output.split("Response:")[-1].strip()
    else:
        final_text = raw_output
        
    metrics = f"\n\n[Telemetry: {generation_time:.2f}s total | {tokens_per_second:.2f} tokens/sec | {generated_tokens_count} tokens]"
    return final_text + metrics

# 5. Gradio Interface with Persona Selector
with gr.Blocks() as demo:
    gr.Markdown("# 🐍 Mamba-1.4B Multi-Persona Assistant")
    gr.Markdown("Dynamic LoRA adapter hot-swapping running on CPU (Selective Mixed-Precision: Int8 Base + FP32 LoRA).")
    
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
            with gr.Row():
                max_tokens_slider = gr.Slider(minimum=20, maximum=300, value=150, step=10, label="Max New Tokens")
                temp_slider = gr.Slider(minimum=0.1, maximum=1.2, value=0.4, step=0.1, label="Temperature")
            
            submit_btn = gr.Button("Generate Response", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(lines=14, label="Persona Output", interactive=False)
            
    submit_btn.click(
        fn=generate_text, 
        inputs=[prompt_input, persona_dropdown, max_tokens_slider, temp_slider], 
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())