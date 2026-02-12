"""
Main orchestrator for inference-only prompt tuning experiments.
Loads Hydra config and invokes inference.py as a subprocess.
"""

import os
import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Orchestrate a single run_id for inference-only prompt tuning.
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Config group name mismatch - cfg.run doesn't exist
    # [CAUSE]: Changed config group from 'run' to 'runs' to match directory name
    # [FIX]: Updated all references from cfg.run to cfg.runs
    #
    # [OLD CODE]:
    # print(f"=== Starting run: {cfg.run.run_id} ===")
    #
    # [NEW CODE]:
    print(f"=== Starting run: {cfg.runs.run_id} ===")
    print(f"Mode: {cfg.mode}")
    print(f"Results dir: {cfg.results_dir}")
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        # Sanity check mode: minimal samples, online wandb, separate namespace
        cfg.dataset.max_samples = 10
        cfg.wandb.mode = "online"
        # Use separate wandb project for sanity checks
        if "sanity" not in cfg.wandb.project:
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
    elif cfg.mode == "main":
        # Main mode: full dataset, online wandb
        cfg.dataset.max_samples = None
        cfg.wandb.mode = "online"
    
    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.runs.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save resolved config to results directory
    config_save_path = results_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Saved config to: {config_save_path}")
    
    # Determine task type from config
    # This is an inference-only task (prompt tuning, no training)
    task_type = "inference"
    
    print(f"\nTask type: {task_type}")
    print(f"Method: {cfg.runs.method.type}")
    
    # Invoke inference.py as subprocess
    if task_type == "inference":
        print("\n=== Invoking inference.py ===")
        
        # Prepare environment
        env = os.environ.copy()
        
        # Build command
        cmd = [
            sys.executable,  # Use same Python interpreter
            "-u",  # Unbuffered output
            "-m",
            "src.inference",
        ]
        
        # Pass config as environment variable (inference.py will reload from saved config)
        env["HYDRA_CONFIG_PATH"] = str(config_save_path)
        env["RUN_ID"] = cfg.runs.run_id
        env["RESULTS_DIR"] = str(results_dir)
        env["MODE"] = cfg.mode
        
        # Run inference subprocess
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path.cwd(),
            check=False,
        )
        
        if result.returncode != 0:
            print(f"\n!!! inference.py failed with return code {result.returncode} !!!")
            sys.exit(result.returncode)
        
        print("\n=== Inference completed successfully ===")
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    print(f"\n=== Run {cfg.runs.run_id} completed ===")


if __name__ == "__main__":
    main()
