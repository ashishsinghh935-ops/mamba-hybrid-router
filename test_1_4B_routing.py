import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class CustomAttentionBlock(nn.Module):
    def __init__(self, hidden_size=2048, num_heads=16):
        super().__init__()
        # 1.4B backbone has a hidden size of 2048
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # STAGE 0: The Zero-Init Gate
        # Initializes as a learnable parameter starting at exactly 0.0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, **kwargs):
        # Save the native Mamba residual stream
        residual = hidden_states
        
        # Process through your custom Transformer attention
        normed_states = self.layer_norm(hidden_states)
        attn_output, _ = self.attention(normed_states, normed_states, normed_states)
        
        # Gated Injection: (residual + 0 * attn_output) equals pure Mamba initially
        output = residual + (self.gate * attn_output)
        
        return output

# --- INFERENCE PIPELINE (The part that got deleted) ---
print("Booting 1.4B inference engine...")
model_name = "state-spaces/mamba-1.4b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Inject custom attention blocks
for idx in [10, 20, 30, 40]:
    model.backbone.layers[idx] = CustomAttentionBlock()

print("Injecting trained custom attention weights...")
trained_weights = torch.load("aligned_custom_blocks_1_4B.pt", map_location="cpu", weights_only=True)

for idx in [10, 20, 30, 40]:
    # We use strict=False because the saved weights don't have the new zero-init gate yet
    model.backbone.layers[idx].load_state_dict(trained_weights[idx], strict=False)

print("Architecture successfully loaded. Ready for generation.\n")

prompt = "The future of artificial intelligence routing architectures will rely on"
print(f"Prompt: {prompt}")
print("Generating...")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

print("\nOutput:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))