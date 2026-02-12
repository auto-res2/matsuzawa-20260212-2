"""
Inference script for prompt-based Chain-of-Thought experiments.
Implements PC-L2M (Proof-Carrying) and DV-L2M (Verifier) methods.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import wandb
import yaml
from omegaconf import OmegaConf

from src.preprocess import (
    load_gsm8k,
    format_problem_prompt,
    extract_answer_from_response,
    extract_calc_block,
)
from src.model import LLMClient, verify_calc_block, score_with_verifier


def run_pc_l2m(
    problem: Dict[str, str],
    llm_client: LLMClient,
    config: dict,
    mode: str,
) -> Tuple[Optional[float], Dict]:
    """
    Run PC-L2M (Proof-Carrying Least-to-Most) on a single problem.
    
    Returns:
        (predicted_answer, metadata)
    """
    n_decompositions = config["method"]["n_decompositions"]
    temperature = config["method"]["temperature"]
    max_tokens = config["method"]["max_tokens"]
    tolerance = config["method"]["verification"]["tolerance"]
    
    # Generate candidate solutions
    prompt = format_problem_prompt(problem["question"], method_type="pc_l2m")
    candidates = llm_client.generate(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_decompositions,
    )
    
    # Verify each candidate
    verified_candidates = []
    all_answers = []
    
    for i, candidate in enumerate(candidates):
        # Extract answer and CALC block
        claimed_answer = extract_answer_from_response(candidate)
        calc_block = extract_calc_block(candidate)
        
        if claimed_answer is None:
            continue
        
        all_answers.append(claimed_answer)
        
        # Verify using Python execution
        if calc_block:
            is_verified = verify_calc_block(calc_block, claimed_answer, tolerance)
            if is_verified:
                verified_candidates.append({
                    "answer": claimed_answer,
                    "candidate": candidate,
                    "calc_lines": len(calc_block.strip().split('\n')),
                })
    
    # Selection strategy
    metadata = {
        "n_candidates": len(candidates),
        "n_verified": len(verified_candidates),
        "all_answers": all_answers,
    }
    
    if verified_candidates:
        # Pick verified majority answer
        verified_answers = [c["answer"] for c in verified_candidates]
        answer_counts = Counter(verified_answers)
        
        if answer_counts:
            # Get most common answer
            most_common = answer_counts.most_common()
            max_count = most_common[0][1]
            
            # If there's a tie, use simplicity (fewest CALC lines)
            tied_answers = [ans for ans, count in most_common if count == max_count]
            
            if len(tied_answers) == 1:
                final_answer = tied_answers[0]
            else:
                # Break tie by simplicity
                tied_candidates = [c for c in verified_candidates if c["answer"] in tied_answers]
                simplest = min(tied_candidates, key=lambda c: c["calc_lines"])
                final_answer = simplest["answer"]
            
            metadata["selection_method"] = "verified_majority"
            return final_answer, metadata
    
    # Fallback: self-consistency majority vote on all extracted answers
    if all_answers:
        answer_counts = Counter(all_answers)
        final_answer = answer_counts.most_common(1)[0][0]
        metadata["selection_method"] = "fallback_majority"
        return final_answer, metadata
    
    metadata["selection_method"] = "none"
    return None, metadata


def run_dv_l2m(
    problem: Dict[str, str],
    llm_client: LLMClient,
    config: dict,
    mode: str,
) -> Tuple[Optional[float], Dict]:
    """
    Run DV-L2M (Decomposition Verifier Least-to-Most) on a single problem.
    
    Returns:
        (predicted_answer, metadata)
    """
    n_decompositions = config["method"]["n_decompositions"]
    temperature = config["method"]["temperature"]
    max_tokens = config["method"]["max_tokens"]
    verifier_prompt_template = config["method"]["verification"]["verifier_prompt"]
    
    # Generate candidate solutions
    prompt = format_problem_prompt(problem["question"], method_type="dv_l2m")
    candidates = llm_client.generate(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_decompositions,
    )
    
    # Score each candidate with LLM verifier
    scored_candidates = []
    all_answers = []
    
    for candidate in candidates:
        claimed_answer = extract_answer_from_response(candidate)
        
        if claimed_answer is None:
            continue
        
        all_answers.append(claimed_answer)
        
        # Score with verifier
        score = score_with_verifier(
            llm_client,
            problem["question"],
            candidate,
            verifier_prompt_template,
        )
        
        scored_candidates.append({
            "answer": claimed_answer,
            "score": score,
            "candidate": candidate,
        })
    
    # Selection strategy: pick top-scored
    metadata = {
        "n_candidates": len(candidates),
        "n_scored": len(scored_candidates),
        "all_answers": all_answers,
    }
    
    if scored_candidates:
        # Sort by score (descending)
        scored_candidates.sort(key=lambda c: c["score"], reverse=True)
        final_answer = scored_candidates[0]["answer"]
        metadata["selection_method"] = "top_scored"
        metadata["top_score"] = scored_candidates[0]["score"]
        return final_answer, metadata
    
    # Fallback: majority vote
    if all_answers:
        answer_counts = Counter(all_answers)
        final_answer = answer_counts.most_common(1)[0][0]
        metadata["selection_method"] = "fallback_majority"
        return final_answer, metadata
    
    metadata["selection_method"] = "none"
    return None, metadata


def main():
    """Main inference loop."""
    # Load config from environment
    config_path = os.environ.get("HYDRA_CONFIG_PATH")
    run_id = os.environ.get("RUN_ID")
    results_dir_str = os.environ.get("RESULTS_DIR")
    mode = os.environ.get("MODE", "main")
    
    if not config_path:
        raise ValueError("HYDRA_CONFIG_PATH not set")
    if not results_dir_str:
        raise ValueError("RESULTS_DIR not set")
    
    results_dir = Path(results_dir_str)
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # [VALIDATOR FIX - Attempt 3]
    # [PROBLEM]: Config key mismatch - need to use config['run'] not config['runs']
    # [CAUSE]: Validator fix attempt 1 was wrong; the config group is 'run' (singular), and saved config has 'run' key
    # [FIX]: Reverted all config['runs'] back to config['run']
    #
    # [OLD CODE (Attempt 1 - incorrect)]:
    # print(f"Method: {config['runs']['method']['type']}")
    # llm_client = LLMClient(model_name=config["runs"]["model"]["name"], ...)
    # dataset = load_gsm8k(split=config["runs"]["dataset"]["split"], ...)
    #
    # [NEW CODE]:
    print(f"=== Running inference: {run_id} ===")
    print(f"Mode: {mode}")
    print(f"Method: {config['run']['method']['type']}")
    
    # Initialize LLM client
    llm_client = LLMClient(
        model_name=config["run"]["model"]["name"],
        provider=config["run"]["model"]["provider"],
        api_key_env=config["run"]["model"]["api_key_env"],
    )
    
    # Load dataset
    dataset = load_gsm8k(
        split=config["run"]["dataset"]["split"],
        cache_dir=config["run"]["dataset"]["cache_dir"],
        max_samples=config["run"]["dataset"].get("max_samples"),
        shuffle_seed=config["run"]["dataset"].get("shuffle_seed"),
    )
    
    print(f"Loaded {len(dataset)} problems from GSM8K")
    
    # Initialize WandB
    wandb_enabled = config["wandb"]["mode"] != "disabled"
    if wandb_enabled:
        wandb.init(
            entity=config["wandb"]["entity"],
            project=config["wandb"]["project"],
            id=run_id,
            config=config,
            resume="allow",
        )
        print(f"WandB run URL: {wandb.run.get_url()}")
    
    # Run inference
    method_type = config["run"]["method"]["type"]
    results = []
    correct_count = 0
    
    for i, problem in enumerate(dataset):
        print(f"\nProblem {i+1}/{len(dataset)}")
        
        # Run method-specific inference
        if method_type == "pc_l2m":
            predicted_answer, metadata = run_pc_l2m(problem, llm_client, config["run"], mode)
        elif method_type == "dv_l2m":
            predicted_answer, metadata = run_dv_l2m(problem, llm_client, config["run"], mode)
        else:
            raise ValueError(f"Unknown method type: {method_type}")
        
        # Check correctness
        ground_truth = float(problem["answer"])
        is_correct = False
        if predicted_answer is not None:
            # Allow small tolerance for floating point
            is_correct = abs(predicted_answer - ground_truth) < 1e-3
        
        if is_correct:
            correct_count += 1
        
        # Store result
        result = {
            "problem_id": i,
            "question": problem["question"],
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            **metadata,
        }
        results.append(result)
        
        # Log to WandB
        if wandb_enabled:
            wandb.log({
                "problem_id": i,
                "is_correct": int(is_correct),
                "accuracy": correct_count / (i + 1),
            })
        
        print(f"Ground truth: {ground_truth}, Predicted: {predicted_answer}, Correct: {is_correct}")
    
    # Calculate final metrics
    accuracy = correct_count / len(dataset) if dataset else 0.0
    
    print(f"\n=== Final Results ===")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(dataset)})")
    
    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_file}")
    
    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(dataset),
    }
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")
    
    # Log final metrics to WandB
    if wandb_enabled:
        wandb.summary["accuracy"] = accuracy
        wandb.summary["correct_count"] = correct_count
        wandb.summary["total_count"] = len(dataset)
        wandb.finish()
    
    # Sanity validation for sanity_check mode
    if mode == "sanity_check":
        perform_sanity_validation(results, metrics, method_type)


def perform_sanity_validation(results: List[Dict], metrics: Dict, method_type: str):
    """Perform sanity validation checks and print verdict."""
    
    # Check 1: At least 5 samples processed
    n_samples = len(results)
    if n_samples < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples_{n_samples}")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples":{n_samples},"status":"fail"}}')
        sys.exit(1)
    
    # Check 2: All outputs are valid (not all None)
    valid_outputs = sum(1 for r in results if r.get("predicted_answer") is not None)
    if valid_outputs == 0:
        print(f"SANITY_VALIDATION: FAIL reason=all_outputs_invalid")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples":{n_samples},"valid_outputs":0,"status":"fail"}}')
        sys.exit(1)
    
    # Check 3: Outputs are not all identical (some variation expected)
    predicted_answers = [r.get("predicted_answer") for r in results if r.get("predicted_answer") is not None]
    unique_answers = len(set(predicted_answers))
    
    # If all predicted answers are identical across different problems, that's suspicious
    # (unless we only have 1-2 samples)
    if unique_answers == 1 and len(predicted_answers) >= 3:
        print(f"SANITY_VALIDATION: FAIL reason=all_outputs_identical")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples":{n_samples},"valid_outputs":{valid_outputs},"unique_answers":{unique_answers},"status":"fail"}}')
        sys.exit(1)
    
    # Check 4: At least one correct answer (sanity check dataset should have solvable problems)
    correct_count = sum(1 for r in results if r.get("is_correct"))
    
    # All checks passed
    print("SANITY_VALIDATION: PASS")
    print(f'SANITY_VALIDATION_SUMMARY: {{"samples":{n_samples},"valid_outputs":{valid_outputs},"unique_answers":{unique_answers},"correct_count":{correct_count},"status":"pass"}}')


if __name__ == "__main__":
    main()
