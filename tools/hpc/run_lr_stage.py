"""Compatibility entry point; new experiments use run_retrain_experiment.py."""
from pathlib import Path
from run_retrain_experiment import main

if __name__ == '__main__':
    main(default_recipe=Path('EdgeFlowNAS/configs/experiments/ft3d_lr.json'))