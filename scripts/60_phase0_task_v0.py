#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_trajectory_v0.task import run

p = argparse.ArgumentParser()
p.add_argument("--cache", default="cache/bnci2014_001_baseline_trajectory_identifiability_v0/baseline_trajectory_covariances.npz")
p.add_argument("--output", default="outputs/bnci2014_001_baseline_trajectory_identifiability_v0")
p.add_argument("--resume", action="store_true")
a = p.parse_args()
run(a.cache, a.output, a.resume)
