#!/usr/bin/env python3
"""
eval_compare.py - evaluation harness for the Mamba-1.4B hybrid + persona adapters.

What it measures (all with GREEDY decoding, identical settings for every config):
  1. Speed:        tokens/sec per config and precision (warm-up excluded)
  2. Quantization: FP32 vs Int8 -> perplexity, teacher-forced top-1 agreement,
                   exact-match rate and first-divergence point of greedy outputs
  3. Adapters:     backbone vs hybrid vs each persona, on the SAME prompts
  4. Quality proxies: keyword hit-rate, distinct-2, repetition-loop rate
  5. A side-by-side markdown file plus a CSV for manual correct/incorrect grading

Usage (run each precision as its own process so only one model is in RAM):
    python eval_compare.py run --precision fp32
    python eval_compare.py run --precision int8      # picks up fp32.json automatically
    python eval_compare.py report

Smoke test first:  python eval_compare.py run --precision int8 --limit 1 --max-new-tokens 30 --fresh

Optional perplexity data: put plain-text files in ./eval_data/
    general.txt, software_engineer.txt, clinical_researcher.txt, legal_counsel.txt
(held-out text NOT used in training; a few thousand tokens each is enough).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import PeftModel

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE_MODEL = "state-spaces/mamba-1.4b-hf"
OUT_DIR = Path("eval_results")
PPL_DIR = Path("eval_data")
PERSONAS = ["software_engineer", "clinical_researcher", "legal_counsel"]

# Keywords: a "|" inside one entry means any-of. Hit-rate = fraction of entries found.
PROMPTS = {
    "general": [
        ("Explain the concept of dynamic equilibrium in a basic system.",
         ["forward|reverse|both directions", "equal|same rate|balance"]),
        ("Summarize how photosynthesis works in three sentences.",
         ["light", "carbon dioxide|CO2", "glucose|sugar"]),
    ],
    "software_engineer": [
        ("Write a C++ function to reverse a singly linked list and explain its time complexity.",
         ["next", "O(n)", "reverse"]),
        ("Write a Python function that checks whether a string is a palindrome.",
         ["def", "return"]),
        ("Explain the difference between a stack and a queue, with a short example in Python.",
         ["LIFO", "FIFO"]),
        ("Write a SQL query to find the second highest salary from an Employee table.",
         ["SELECT", "salary"]),
    ],
    "clinical_researcher": [
        ("What are the primary serum tumor markers used in the diagnosis of hepatocellular carcinoma?",
         ["AFP|alpha-fetoprotein", "AFP-L3|DCP|PIVKA"]),
        ("What is the mechanism of action of metformin in type 2 diabetes?",
         ["AMPK|gluconeogenesis|hepatic glucose"]),
        ("Explain the difference between sensitivity and specificity in a diagnostic test.",
         ["true positive", "true negative"]),
        ("What are the first-line treatments for community-acquired pneumonia in an otherwise healthy adult?",
         ["amoxicillin|macrolide|azithromycin|doxycycline"]),
    ],
    "legal_counsel": [
        ("Explain the scope of Article 21 of the Indian Constitution regarding the right to a clean environment.",
         ["Article 21", "environment", "Subhash Kumar|M.C. Mehta|MC Mehta"]),
        ("What is the difference between a bailable and a non-bailable offence under Indian law?",
         ["bail", "CrPC|Code of Criminal Procedure|BNSS"]),
        ("Explain the doctrine of basic structure of the Indian Constitution.",
         ["Kesavananda", "Parliament"]),
        ("What are the essential elements of a valid contract under the Indian Contract Act, 1872?",
         ["offer", "acceptance", "consideration"]),
    ],
}


# ----------------------------------------------------------------------------
# HOOKS - Custom Implementation
# ----------------------------------------------------------------------------
class FastCustomBlock(nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))
        self.enabled = True  

    def forward(self, hidden_states, **kwargs):
        if not getattr(self, "enabled", True):
            return hidden_states  
        orig_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        out = h + self.gate * self.proj(self.norm(h))
        return out.to(orig_dtype)

def format_prompt(persona: str, text: str) -> str:
    if persona == "software_engineer":
        return f"Instruction: {text.strip()}\n\nInput:\n\nResponse:"
    elif persona in ["clinical_researcher", "legal_counsel"]:
        return f"Instruction: {text.strip()}\n\nResponse:"
    else:
        return text

def build_model(precision: str):
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, low_cpu_mem_usage=True).to("cpu")

    aligned_weights = torch.load("version_2_artifacts/aligned_custom_blocks_stage2.pt", map_location="cpu", weights_only=True)
    for idx in [10, 20, 30, 40]:
        block = FastCustomBlock().to("cpu")
        block.load_state_dict(aligned_weights[idx])
        model.backbone.layers[idx] = block

    model = PeftModel.from_pretrained(model, "version_2_artifacts/mamba_fully_adapted_final", adapter_name="base").to("cpu")
    model.load_adapter("./persona_adapter_engineering_v2", adapter_name="software_engineer")
    model.load_adapter("./persona_adapter_clinical", adapter_name="clinical_researcher")
    model.load_adapter("./persona_adapter_legal", adapter_name="legal_counsel")

    if precision == "int8":
        def quantize_base_layers_only(module):
            for name, child in module.named_children():
                if name in ["lora_A", "lora_B"]:
                    continue
                if isinstance(child, nn.Linear):
                    setattr(module, name, torch.quantization.quantize_dynamic(child, {nn.Linear}, dtype=torch.qint8))
                else:
                    quantize_base_layers_only(child)
        quantize_base_layers_only(model)

    return model.eval()

def activate(model, blocks_on: bool, adapter: Optional[str]) -> None:
    for m in model.modules():
        if m.__class__.__name__ == "FastCustomBlock":
            m.enabled = blocks_on
            
    if hasattr(model, "set_adapter"):
        if adapter is None:
            model.base_model.disable_adapter_layers()
        else:
            model.base_model.enable_adapter_layers()
            model.set_adapter(adapter)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj) -> None:
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    tmp.replace(p)


def kw_hit(text: str, keywords) -> Optional[float]:
    if not keywords:
        return None
    t = text.lower()
    hits = sum(any(alt.strip().lower() in t for alt in k.split("|")) for k in keywords)
    return hits / len(keywords)


def distinct_n(ids, n=2) -> Optional[float]:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def has_loop(ids, n=4, reps=3) -> bool:
    c = Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
    return any(v >= reps for v in c.values())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=2, pct=False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def make_configs(skip_backbone: bool):
    cfgs = []
    if not skip_backbone:
        cfgs.append(("backbone", False, None))
    cfgs.append(("hybrid", True, None))
    cfgs += [(p, True, p) for p in PERSONAS]
    return cfgs


def prompts_for(cfg_name: str, limit: Optional[int]):
    """(persona, index, text, keywords) tuples that a config should run."""
    personas = ["general"] + PERSONAS if cfg_name in ("backbone", "hybrid") else [cfg_name]
    out = []
    for p in personas:
        items = PROMPTS[p][:limit] if limit else PROMPTS[p]
        for i, (text, kws) in enumerate(items):
            out.append((p, i, text, kws))
    return out


# ----------------------------------------------------------------------------
# core measurements
# ----------------------------------------------------------------------------
@torch.inference_mode()
def generate(model, tok, text, a):
    ids = tok(text, return_tensors="pt")["input_ids"]
    kw = dict(max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    if a.repetition_penalty != 1.0:
        kw["repetition_penalty"] = a.repetition_penalty
    if a.no_repeat_ngram > 0:
        kw["no_repeat_ngram_size"] = a.no_repeat_ngram
    t0 = time.perf_counter()
    out = model.generate(input_ids=ids, **kw)
    dt = time.perf_counter() - t0
    return ids[0], out[0, ids.shape[1]:], dt


@torch.inference_mode()
def tf_agreement(model, prompt_ids, new_ids) -> Optional[float]:
    """Fraction of positions where the model's argmax equals the given continuation (teacher forcing)."""
    if new_ids.numel() == 0:
        return None
    full = torch.cat([prompt_ids, new_ids]).unsqueeze(0)
    logits = model(input_ids=full).logits[0].float()
    p = prompt_ids.numel()
    pred = logits[p - 1:-1].argmax(-1)
    return (pred == new_ids).float().mean().item()


@torch.inference_mode()
def perplexity(model, tok, text, window, max_tokens):
    ids = tok(text, return_tensors="pt")["input_ids"][0][:max_tokens]
    nll, cnt = 0.0, 0
    for s in range(0, ids.numel() - 1, window):
        chunk = ids[s:s + window + 1]
        if chunk.numel() < 2:
            break
        logits = model(input_ids=chunk[:-1].unsqueeze(0)).logits[0].float()
        nll += torch.nn.functional.cross_entropy(logits, chunk[1:], reduction="sum").item()
        cnt += chunk.numel() - 1
    return math.exp(nll / cnt), cnt


@torch.inference_mode()
def sanity_check(model, tok, configs):
    """Warn if toggling blocks/adapters has no effect (a wiring bug would invalidate every comparison)."""
    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]
    cache = {}
    for name, blocks_on, adapter in configs:
        activate(model, blocks_on, adapter)
        cache[name] = model(input_ids=ids).logits.float()

    def diff(x, y):
        return (cache[x] - cache[y]).abs().max().item()

    if "backbone" in cache and diff("backbone", "hybrid") < 1e-6:
        print("  [warn] backbone == hybrid logits: block toggle is NOT wired (or blocks are still zero-gated).")
    for p in PERSONAS:
        if diff("hybrid", p) < 1e-6:
            print(f"  [warn] hybrid == {p} logits: adapter toggle is NOT wired (or adapter is untrained).")


# ----------------------------------------------------------------------------
# run mode
# ----------------------------------------------------------------------------
def run(a):
    from transformers import AutoTokenizer

    OUT_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    path = OUT_DIR / f"{a.precision}.json"
    settings = dict(max_new_tokens=a.max_new_tokens, repetition_penalty=a.repetition_penalty,
                    no_repeat_ngram=a.no_repeat_ngram, ppl_window=a.ppl_window,
                    ppl_max_tokens=a.ppl_max_tokens)
    res = None
    if path.exists() and not a.fresh:
        res = load_json(path)
        if res["settings"] != settings:
            sys.exit(f"{path} was made with different settings {res['settings']}. Use --fresh to overwrite.")
        print(f"Resuming from {path} ({len(res['generations'])} generations already done)")
    if res is None:
        res = {"settings": settings, "generations": {}, "ppl": {},
               "meta": {"precision": a.precision, "date": datetime.now().isoformat(timespec="seconds"),
                        "torch": torch.__version__, "threads": a.threads,
                        "cpu": platform.processor() or platform.machine()}}

    ref = None
    ref_path = OUT_DIR / "fp32.json"
    if a.precision != "fp32" and ref_path.exists():
        ref = load_json(ref_path)
        print("Found fp32.json -> will compute FP32-vs-this-precision agreement.")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"Building {a.precision} model ...")
    model = build_model(a.precision)
    model.eval()
    configs = make_configs(a.skip_backbone)

    print("Sanity-checking block/adapter toggles ...")
    sanity_check(model, tok, configs)

    print("Warm-up ...")
    activate(model, True, None)
    generate(model, tok, "Hello", argparse.Namespace(max_new_tokens=8, repetition_penalty=1.0, no_repeat_ngram=0))

    # ---- generations ----
    for name, blocks_on, adapter in configs:
        activate(model, blocks_on, adapter)
        for persona, idx, text, kws in prompts_for(name, a.limit):
            pid = f"{persona}-{idx}"
            key = f"{name}|{pid}"
            if key in res["generations"]:
                continue
            formatted = format_prompt(persona if persona != "general" else name, text)
            prompt_ids, new_ids, dt = generate(model, tok, formatted, a)
            out_text = tok.decode(new_ids, skip_special_tokens=True)
            ids_list = new_ids.tolist()
            rec = {
                "config": name, "prompt_id": pid, "persona": persona, "prompt": text,
                "output": out_text, "new_ids": ids_list, "n_new": len(ids_list),
                "seconds": dt, "tok_per_s": (len(ids_list) / dt) if dt > 0 else None,
                "kw_hit": kw_hit(out_text, kws), "distinct2": distinct_n(ids_list, 2),
                "loop": has_loop(ids_list), "tf_agree_self": tf_agreement(model, prompt_ids, new_ids),
            }
            if ref is not None and key in ref["generations"]:
                r_ids = ref["generations"][key]["new_ids"]
                rec["tf_agree_vs_fp32"] = tf_agreement(model, prompt_ids, torch.tensor(r_ids, dtype=torch.long))
                m = min(len(r_ids), len(ids_list))
                div = next((i for i in range(m) if r_ids[i] != ids_list[i]), m)
                rec["exact_match_fp32"] = (r_ids == ids_list)
                rec["first_divergence_frac"] = (div / m) if m else None
            res["generations"][key] = rec
            save_json(path, res)
            print(f"  {key:45s} {rec['n_new']:4d} tok  {rec['tok_per_s'] or 0:5.2f} tok/s", flush=True)

    # ---- perplexity ----
    if PPL_DIR.exists():
        for f in sorted(PPL_DIR.glob("*.txt")):
            tname = f.stem
            if tname == "general":
                cfgs = [c for c in configs if c[0] in ("backbone", "hybrid")]
            elif tname in PERSONAS:
                cfgs = [c for c in configs if c[0] in ("hybrid", tname)]
            else:
                print(f"  [skip] {f.name}: name must be general or one of {PERSONAS}")
                continue
            text = f.read_text(encoding="utf-8")
            for name, blocks_on, adapter in cfgs:
                key = f"{name}|{tname}"
                if key in res["ppl"]:
                    continue
                activate(model, blocks_on, adapter)
                ppl, n = perplexity(model, tok, text, a.ppl_window, a.ppl_max_tokens)
                res["ppl"][key] = {"config": name, "text": tname, "ppl": ppl, "n_tokens": n}
                save_json(path, res)
                print(f"  PPL {key:40s} {ppl:8.2f}  ({n} tokens)", flush=True)
    else:
        print(f"(no {PPL_DIR}/ folder -> skipping perplexity; see the docstring)")

    del model
    gc.collect()
    print(f"\nDone. Saved {path}")


# ----------------------------------------------------------------------------
# report mode
# ----------------------------------------------------------------------------
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def quote(text: str) -> str:
    text = text.strip() or "(empty)"
    return "\n".join("> " + ln for ln in text.splitlines())


def report(a):
    data = {p: load_json(OUT_DIR / f"{p}.json") for p in ("fp32", "int8") if (OUT_DIR / f"{p}.json").exists()}
    if not data:
        sys.exit("No results found. Run `run --precision fp32` / `int8` first.")
    cfg_order = ["backbone", "hybrid"] + PERSONAS
    L = ["# Evaluation report", "",
         "All generations use greedy decoding with identical settings. "
         "Keyword hit-rate is a rough proxy; see grading.csv for manual correctness.", ""]
    for prec, d in data.items():
        L.append(f"- `{prec}`: {d['meta']['date']}, torch {d['meta']['torch']}, {d['meta']['threads']} threads, "
                 f"settings {d['settings']}")
    L.append("")

    # 1. performance + quality proxies
    rows = []
    for prec, d in data.items():
        for cfg in cfg_order:
            g = [r for r in d["generations"].values() if r["config"] == cfg]
            if not g:
                continue
            rows.append([prec, cfg, str(len(g)), fmt(mean([r["tok_per_s"] for r in g])),
                         fmt(mean([r["n_new"] for r in g]), 0), fmt(mean([r["kw_hit"] for r in g]), pct=True),
                         fmt(mean([r["distinct2"] for r in g])),
                         fmt(mean([1.0 if r["loop"] else 0.0 for r in g]), pct=True)])
    L += ["## 1. Speed and output-quality proxies", "",
          md_table(["precision", "config", "n", "tok/s", "avg new tok", "keyword hit", "distinct-2", "loop rate"], rows), ""]

    # 2. speed-up
    if "fp32" in data and "int8" in data:
        rows = []
        for cfg in cfg_order:
            f = mean([r["tok_per_s"] for r in data["fp32"]["generations"].values() if r["config"] == cfg])
            i = mean([r["tok_per_s"] for r in data["int8"]["generations"].values() if r["config"] == cfg])
            if f and i:
                rows.append([cfg, fmt(f), fmt(i), fmt(i / f) + "x"])
        L += ["## 2. Int8 speed-up (tokens/sec, same decoding settings)", "",
              md_table(["config", "fp32 tok/s", "int8 tok/s", "ratio"], rows), ""]

        # 3. quantization fidelity
        rows = []
        for cfg in cfg_order:
            g = [r for r in data["int8"]["generations"].values() if r["config"] == cfg and "tf_agree_vs_fp32" in r]
            if not g:
                continue
            ceiling = mean([r["tf_agree_self"] for r in g])
            rows.append([cfg, str(len(g)), fmt(mean([r["tf_agree_vs_fp32"] for r in g]), pct=True),
                         fmt(ceiling, pct=True), fmt(mean([1.0 if r["exact_match_fp32"] else 0.0 for r in g]), pct=True),
                         fmt(mean([r["first_divergence_frac"] for r in g]), pct=True)])
        L += ["## 3. Int8 vs FP32 fidelity", "",
              "Teacher-forced agreement = how often int8's top-1 token equals the FP32 greedy token on the FP32 text. "
              "'Self ceiling' is int8 agreeing with its own greedy output (below 100% only if repetition penalty / n-gram "
              "guards are on).", "",
              md_table(["config", "n", "int8 vs fp32 agreement", "self ceiling", "exact match", "avg first divergence"], rows), ""]

        # 4. perplexity
        rows = []
        for key, r in data["fp32"]["ppl"].items():
            r2 = data["int8"]["ppl"].get(key)
            if r2:
                rows.append([r["config"], r["text"], fmt(r["ppl"]), fmt(r2["ppl"]),
                             f"{(r2['ppl'] / r['ppl'] - 1) * 100:+.1f}%"])
        if rows:
            L += ["## 4. Perplexity (lower is better)", "",
                  md_table(["config", "text", "fp32", "int8", "change"], rows), ""]
    else:
        for prec, d in data.items():
            rows = [[r["config"], r["text"], fmt(r["ppl"]), str(r["n_tokens"])] for r in d["ppl"].values()]
            if rows:
                L += [f"## Perplexity ({prec})", "", md_table(["config", "text", "ppl", "tokens"], rows), ""]

    # 5. side by side
    L += ["## 5. Side-by-side outputs", ""]
    for persona in ["general"] + PERSONAS:
        for i, (text, _) in enumerate(PROMPTS[persona]):
            pid = f"{persona}-{i}"
            L += [f"### [{persona}] {text}", ""]
            for cfg in cfg_order:
                for prec, d in data.items():
                    r = d["generations"].get(f"{cfg}|{pid}")
                    if r:
                        L += [f"**{cfg} / {prec}** ({r['n_new']} tokens, {fmt(r['tok_per_s'])} tok/s)", "",
                              quote(r["output"]), ""]
    (OUT_DIR / "comparison.md").write_text("\n".join(L), encoding="utf-8")

    # manual-grading CSV
    with open(OUT_DIR / "grading.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "config", "precision", "prompt", "output", "correct_0_or_1", "notes"])
        for prec, d in data.items():
            for r in d["generations"].values():
                w.writerow([r["prompt_id"], r["config"], prec, r["prompt"], r["output"], "", ""])
    print(f"Wrote {OUT_DIR / 'comparison.md'} and {OUT_DIR / 'grading.csv'}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate + measure for one precision")
    r.add_argument("--precision", choices=["fp32", "int8"], required=True)
    r.add_argument("--max-new-tokens", type=int, default=120)
    r.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="set to what your demo uses; keep identical for fp32 and int8")
    r.add_argument("--no-repeat-ngram", type=int, default=0, help="same rule as above")
    r.add_argument("--threads", type=int, default=torch.get_num_threads())
    r.add_argument("--limit", type=int, default=None, help="only the first N prompts per persona (smoke test)")
    r.add_argument("--skip-backbone", action="store_true", help="skip the no-blocks baseline config")
    r.add_argument("--ppl-window", type=int, default=256)
    r.add_argument("--ppl-max-tokens", type=int, default=2048)
    r.add_argument("--fresh", action="store_true", help="ignore/overwrite existing results for this precision")
    r.set_defaults(fn=run)
    p = sub.add_parser("report", help="build comparison.md + grading.csv from saved results")
    p.set_defaults(fn=report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()