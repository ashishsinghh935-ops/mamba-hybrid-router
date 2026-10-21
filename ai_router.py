import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

print("Booting AI Router Backend...")
model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32 if device == "cpu" else torch.float16, 
    device_map=device
)

class CustomAttentionBlock(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        return attn_output + residual

# 1. Graft the empty blocks into the Mamba backbone
for idx in [4, 9, 14, 19]:
    dtype_to_use = torch.float32 if device == "cpu" else torch.float16
    base_model.backbone.layers[idx] = CustomAttentionBlock(hidden_size=768).to(device, dtype=dtype_to_use)

# 2. Load the aligned weights
print("Loading aligned attention blocks...")
aligned_state = torch.load("aligned_custom_blocks.pt", map_location=device, weights_only=True)
for idx in [4, 9, 14, 19]:
    base_model.backbone.layers[idx].load_state_dict(aligned_state[idx])

# 3. Load the domain-specific LoRA adapters
model = PeftModel.from_pretrained(base_model, "./clinical_lora_final", adapter_name="clinical")
model.load_adapter("./legal_lora_final", adapter_name="legal")
model.load_adapter("./code_lora_final_lightning", adapter_name="code")
print("All Personas Loaded. Ready for UI integration.")

def generate_response(persona_name, prompt):
    """Hot-swaps the model and returns the generated text and latency."""
    model.set_adapter(persona_name)
    
    # Format prompts to force direct completion rather than conversational chat
    if persona_name == "code":
        # Force a 4-space indent immediately after the user's function header
        formatted_prompt = prompt.strip() + "\n    "
    elif persona_name == "clinical":
        formatted_prompt = f"Clinical Note: {prompt}\nAssessment: "
    else:
        formatted_prompt = f"Legal Clause: {prompt}\nProvision: "
        
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    start_time = time.time()
    
    # Switched to greedy search (do_sample=False) to force logical structure 
    # instead of probabilistically wandering off-topic.
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    latency = round(time.time() - start_time, 2)
    
    generated_tokens = outputs[0][input_length:]
    assistant_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return assistant_reply, latency