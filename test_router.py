import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import db_manager

print("1. Booting Base Model...")
model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32 if device == "cpu" else torch.float16, 
    device_map=device
)

print("2. Applying the 80/20 Hybrid Graft...")
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

for idx in [4, 9, 14, 19]:
    dtype_to_use = torch.float32 if device == "cpu" else torch.float16
    base_model.backbone.layers[idx] = CustomAttentionBlock(hidden_size=768).to(device, dtype=dtype_to_use)

print("3. Loading All Three Domain Personas into Memory...")
model = PeftModel.from_pretrained(base_model, "./clinical_lora_final", adapter_name="clinical")
model.load_adapter("./legal_lora_final", adapter_name="legal")
model.load_adapter("./code_lora_final_lightning", adapter_name="code")

print("4. Initializing SQLite Database...")
db_manager.init_db()
current_session = "demo_session_01"

def ask_persona(persona_name, prompt):
    model.set_adapter(persona_name)
    print(f"\n[{persona_name.upper()} ACTIVE]")
    
    # Log User Prompt
    db_manager.log_message(current_session, "User", "None", prompt)
    
    inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").to(device)
    
    # Start Latency Timer
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=40, temperature=0.7)
    latency = round(time.time() - start_time, 2)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_reply = response_text.split("Assistant:")[-1].strip()
    
    print(f"{assistant_reply} \n[Latency: {latency}s]")
    
    # Log Assistant Reply
    db_manager.log_message(current_session, "Assistant", persona_name, assistant_reply, latency)

print("\n--- INITIATING 3-WAY HOT-SWAP TEST ---")
ask_persona("clinical", "What type of injury often leads to supracondylar fractures?")
ask_persona("legal", "What is the primary purpose of a liability clause in a contract?")
ask_persona("code", "Write a Python function to reverse a string.")