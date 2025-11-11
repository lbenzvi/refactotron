#!/usr/bin/env python3
"""
Quick script to generate properly formatted training data
"""
import json
import random
import re
import ast

def simple_degrade(code):
    """Apply simple degradations to code"""
    degraded = code

    # Remove docstrings
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.body and isinstance(node.body[0], ast.Expr):
                    if isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                        node.body.pop(0)
        degraded = ast.unparse(tree)
    except:
        pass

    # Remove type hints
    degraded = re.sub(r':\s*[A-Za-z_][\w\[\],\s]*(?=\)|\s*=|\s*,)', '', degraded)
    degraded = re.sub(r'->\s*[A-Za-z_][\w\[\],\s]*:', ':', degraded)

    # Bad variable names
    degraded = re.sub(r'\bdata\b', 'x', degraded)
    degraded = re.sub(r'\bresult\b', 'y', degraded)
    degraded = re.sub(r'\bvalue\b', 'z', degraded)

    return degraded

# Load clean functions
print("Loading clean functions...")
with open('./data/clean_functions_optimized.jsonl', 'r') as f:
    functions = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(functions)} functions")

# Create degraded pairs
print("Creating degraded pairs...")
pairs = []
for func_data in functions:
    clean_code = func_data['code']
    degraded_code = simple_degrade(clean_code)

    # Format for training
    pair = {
        'input': f"### Refactor the following Python code to improve quality:\n\n{degraded_code}\n\n### Refactored code:",
        'output': clean_code
    }
    pairs.append(pair)

print(f"Created {len(pairs)} pairs")

# Shuffle and split
random.seed(42)
random.shuffle(pairs)

n_train = int(len(pairs) * 0.8)
n_val = int(len(pairs) * 0.1)

splits = {
    'train': pairs[:n_train],
    'validation': pairs[n_train:n_train + n_val],
    'test': pairs[n_train + n_val:]
}

# Write with proper newlines
for name, data in splits.items():
    filepath = f'./data/{name}.jsonl'
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    # Verify
    with open(filepath, 'r') as f:
        count = sum(1 for _ in f)

    print(f"✅ {name}.jsonl: {count} samples")

print("\n🎉 Done! Files are ready for training.")
