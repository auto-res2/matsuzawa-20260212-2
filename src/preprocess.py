"""
Dataset preprocessing utilities for GSM8K and other math word problem datasets.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset


def load_gsm8k(
    split: str = "test",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset.
    
    Args:
        split: Dataset split (train/test)
        cache_dir: Directory to cache downloaded dataset
        max_samples: Maximum number of samples to load (None = all)
        shuffle_seed: Random seed for shuffling (None = no shuffle)
    
    Returns:
        List of dictionaries with keys:
            - question: The math word problem
            - answer: The ground truth answer (numeric)
            - solution: The step-by-step solution (if available)
    """
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
    
    # Load GSM8K from HuggingFace datasets
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    
    # Shuffle if requested
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    
    # Limit samples if requested
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Convert to list of dicts
    samples = []
    for item in dataset:
        # Extract numeric answer from the answer field
        # GSM8K answer format: "Step-by-step solution\n#### 42"
        answer_text = item["answer"]
        numeric_answer = extract_numeric_answer(answer_text)
        
        samples.append({
            "question": item["question"],
            "answer": numeric_answer,
            "solution": answer_text,  # Full solution text
        })
    
    return samples


def extract_numeric_answer(answer_text: str) -> float:
    """
    Extract numeric answer from GSM8K answer format.
    
    GSM8K answers are formatted as:
    "Step 1: ... Step 2: ... #### 42"
    
    This extracts the number after ####.
    """
    # Look for #### followed by a number
    match = re.search(r"####\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", answer_text)
    if match:
        return float(match.group(1))
    
    # Fallback: try to find any number at the end
    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", answer_text)
    if match:
        return float(match.group(1))
    
    raise ValueError(f"Could not extract numeric answer from: {answer_text}")


def format_problem_prompt(question: str, method_type: str = "standard") -> str:
    """
    Format a math word problem into a prompt for the LLM.
    
    Args:
        question: The math word problem text
        method_type: Type of prompting method (standard, pc_l2m, dv_l2m)
    
    Returns:
        Formatted prompt string
    """
    if method_type == "pc_l2m":
        # Proof-Carrying L2M prompt
        prompt = f"""You are solving a math word problem. Follow these steps:

1. Break the problem into 3-6 smaller subproblems.
2. Solve each subproblem step by step.
3. For each calculation, provide a CALC block with executable arithmetic:
   - Use only: +, -, *, /, //, %, **, ( )
   - Assign variables as needed (e.g., x = 3*7)
   - End with: answer = <final_expression>

Format your response as:
DECOMPOSITION:
1. [First subproblem]
2. [Second subproblem]
...

SOLUTION:
[Natural language explanation of your reasoning]

CALC:
x = [arithmetic expression]
y = [arithmetic expression]
...
answer = [final expression]

FINAL: [numeric answer]

Problem: {question}"""
    
    elif method_type == "dv_l2m":
        # Decomposition Verifier L2M prompt (standard L2M)
        prompt = f"""You are solving a math word problem. Follow these steps:

1. Break the problem into 3-6 smaller subproblems.
2. Solve each subproblem step by step.
3. Show your reasoning clearly.

Format your response as:
DECOMPOSITION:
1. [First subproblem]
2. [Second subproblem]
...

SOLUTION:
[Step-by-step solution with reasoning]

FINAL: [numeric answer]

Problem: {question}"""
    
    else:
        # Standard CoT prompt
        prompt = f"""Solve this math word problem step by step.

Problem: {question}

Solution:"""
    
    return prompt


def extract_answer_from_response(response: str) -> Optional[float]:
    """
    Extract numeric answer from LLM response.
    
    Looks for:
    1. FINAL: <number>
    2. #### <number>
    3. Last number in the response
    """
    # Try FINAL: format
    match = re.search(r"FINAL:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Try #### format
    match = re.search(r"####\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", response)
    if match:
        return float(match.group(1))
    
    # Try to find last number
    matches = re.findall(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", response)
    if matches:
        return float(matches[-1])
    
    return None


def extract_calc_block(response: str) -> Optional[str]:
    """
    Extract CALC block from PC-L2M response.
    
    Returns the CALC section as a string, or None if not found.
    """
    match = re.search(r"CALC:\s*\n((?:.*\n)*?)(?:FINAL:|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
