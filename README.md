<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,50:003320,100:000000&height=200&section=header&text=LlamaTron%20RS1%20MedThinker&fontSize=42&fontColor=00ff96&fontAlignY=38&desc=Clinical%20Reasoning%20%C2%B7%20Chain-of-Thought%20%C2%B7%20Fine-Tuned%20on%20810K%20Samples&descAlignY=58&descSize=16&descColor=ffffff"/>

<br/>

[![Model on HuggingFace](https://img.shields.io/badge/HuggingFace-Model-00ff96?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Rumiii/LlamaTron-RS1-MedThinker)
[![Base Model](https://img.shields.io/badge/Base-Llama%203.2%201B%20Instruct-00cfff?style=for-the-badge&logo=meta&logoColor=white)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-810K%20Samples-ffffff?style=for-the-badge&logo=databricks&logoColor=black)](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-Trinity-Mini)
[![License](https://img.shields.io/badge/License-Apache%202.0-ffcc00?style=for-the-badge)](LICENSE)

<br/>

---

### A 1B parameter medical reasoning model fine-tuned on 810K chain-of-thought samples,
### built to think through clinical cases step by step like a junior doctor.

---

</div>

<br/>

## Overview

LlamaTron RS1 MedThinker is a domain-specific fine-tune of Meta's Llama 3.2 1B Instruct model, trained exclusively on structured medical reasoning data. Unlike standard medical QA models that return direct answers, MedThinker was trained on chain-of-thought reasoning traces — teaching it not just *what* to answer, but *how to reason* through a clinical case from differential diagnosis to treatment escalation.

The model is designed to mirror the reasoning pattern of a junior physician: it identifies the most likely diagnosis, explains the clinical logic, outlines immediate treatment steps, and flags when to escalate care.

<br/>

## Demo

<div align="center">

**Interface**

<img width="1444" alt="LlamaTron RS1 MedThinker Interface" src="https://github.com/user-attachments/assets/fcfccd70-3113-4eb1-ac90-f3bc40f53cb0"/>

<br/><br/>

**Model Output — Step-by-Step Clinical Reasoning**

<img width="1440" alt="LlamaTron RS1 MedThinker Output" src="https://github.com/user-attachments/assets/9949a73f-c953-41bf-a7d9-525c31b81336"/>

</div>

<br/>

## Model Output Format

Every response follows a strict clinical structure, enforced through few-shot prompting at inference time:

```
DIAGNOSIS       — Primary diagnosis and key differentials
REASONING       — Clinical logic and pathophysiology
IMMEDIATE TREATMENT — Prioritised, dose-specific treatment steps  
RED FLAGS       — Escalation criteria and emergency triggers
```

**Example Input:**
```
3yo boy, barking cough, stridor, worse at night. Diagnosis and treatment?
```

**Example Output:**
```
DIAGNOSIS: Viral croup (laryngotracheobronchitis).
Differentials: epiglottitis, foreign body aspiration, bacterial tracheitis.

REASONING: The classic triad of barking cough, inspiratory stridor, and nocturnal
worsening in a toddler is pathognomonic for croup. Caused by parainfluenza virus
producing subglottic edema. Nighttime worsening occurs due to supine positioning
increasing mucosal congestion.

IMMEDIATE TREATMENT:
- Dexamethasone 0.6mg/kg PO/IM single dose
- Nebulised epinephrine 5ml of 1:1000 if stridor present at rest
- Keep child calm — agitation worsens stridor
- Oxygen via mask if SpO2 < 92%

RED FLAGS: Call emergency services if stridor at rest is not improving,
child is drooling, cyanosis appears, or child becomes exhausted.
```

<br/>

## Training Details

| Parameter | Value |
|---|---|
| Base Model | meta-llama/Llama-3.2-1B-Instruct |
| Fine-Tune Method | LoRA SFT via Unsloth |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Target Modules | q, k, v, o, gate, up, down projections |
| Sequence Length | 2048 |
| Batch Size | 16 per device |
| Gradient Accumulation | 2 steps (effective batch 32) |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Precision | BF16 |
| Hardware | NVIDIA RTX A6000 (48GB VRAM) |
| Framework | Unsloth + TRL |

<br/>

## Dataset

Trained on [OpenMed/Medical-Reasoning-SFT-Trinity-Mini](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-Trinity-Mini), generated using `arcee-ai/Trinity-Mini`.

| Metric | Value |
|---|---|
| Total Samples | 810,374 |
| Total Tokens | 1.52 Billion |
| Reasoning Tokens | 977 Million |
| Content Tokens | 542 Million |
| Language | English |

The dataset contains chain-of-thought reasoning traces alongside every answer. This means the model was trained on both the *answer* and the *thinking process* that produced it — resulting in structured, explainable outputs rather than flat responses.

Dataset credit: Maziyar P.

<br/>

## Quickstart

### Installation

```bash
pip install torch transformers accelerate
```

### Inference

```python
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

MODEL_PATH = "Rumiii/LlamaTron-RS1-MedThinker"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

FEW_SHOT = """CASE: 2yo girl, high fever, tugging right ear, irritable, not sleeping.

DIAGNOSIS: Acute otitis media (AOM). Differentials: otitis externa, teething.

REASONING: Unilateral ear tugging with fever and irritability in a toddler is the
classic AOM presentation. Peak incidence at 6mo-2yr due to horizontal Eustachian
tube anatomy impairing drainage.

IMMEDIATE TREATMENT:
- Amoxicillin 90mg/kg/day divided BID x 10 days
- Ibuprofen/paracetamol for pain and fever
- Re-evaluate in 48-72h if no improvement

RED FLAGS: Refer immediately if mastoid swelling, facial palsy, or no improvement
after 72h of antibiotics."""

def ask(question: str) -> str:
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"You are LlamaTron RS1 MedThinker, a clinical medical assistant. "
        f"Always use the structured format shown.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Answer this case using structured format:\n\n{FEW_SHOT}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{FEW_SHOT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\nCASE: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.35,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=False)
    for stop in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]:
        if stop in raw:
            raw = raw[:raw.index(stop)]
    return raw.strip()

print(ask("25yo male, fever 39.8C, neck stiffness, photophobia, petechial rash. Diagnosis and immediate action?"))
```

<br/>

## Disclaimer

This model is intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. All outputs should be reviewed by a qualified medical professional before any clinical application.

<br/>

## Citation

```bibtex
@model{llamatron_rs1_medthinker,
  title        = {LlamaTron RS1 MedThinker},
  author       = {Rumiii},
  year         = {2026},
  base_model   = {meta-llama/Llama-3.2-1B-Instruct},
  dataset      = {OpenMed/Medical-Reasoning-SFT-Trinity-Mini},
  url          = {https://huggingface.co/Rumiii/LlamaTron-RS1-MedThinker}
}
```

<br/>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,50:003320,100:000000&height=100&section=footer"/>

</div>
