from datasets import load_dataset
import json
import os

# Load dataset (already downloaded, won't re-download)
print("Loading dataset...")
ds = load_dataset("OpenMed/Medical-Reasoning-SFT-Trinity-Mini")
print(f"Total samples: {len(ds['train'])}")

# Output path
output_path = "/home/Dataset/medical_reasoning_sft.jsonl"
os.makedirs("/home/Dataset", exist_ok=True)

# Transform and save
print("Transforming and saving...")
count = 0
skipped = 0

with open(output_path, "w", encoding="utf-8") as f:
    for sample in ds["train"]:
        messages = sample["messages"]
        
        transformed = []
        valid = True
        
        for msg in messages:
            role = msg["role"]
            
            if role == "user":
                transformed.append({
                    "role": "user",
                    "content": msg["content"]
                })
            
            elif role == "assistant":
                # Use reasoning_content if available, fallback to content
                answer = msg["reasoning_content"] if msg["reasoning_content"] else msg["content"]
                
                if not answer:
                    valid = False
                    break
                
                transformed.append({
                    "role": "assistant",
                    "content": answer
                })
        
        if valid and len(transformed) >= 2:
            f.write(json.dumps({"messages": transformed}, ensure_ascii=False) + "\n")
            count += 1
        else:
            skipped += 1

print(f"Saved : {count} samples")
print(f"Skipped: {skipped} samples (missing content)")

# Sanity check - preview first saved sample
print("\n--- First sample preview ---")
with open(output_path, "r") as f:
    first = json.loads(f.readline())
    print(f"User   : {first['messages'][0]['content'][:100]}...")
    print(f"Answer : {first['messages'][1]['content'][:200]}...")
