#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_trajectory_v0.decision import decide, report
p = argparse.ArgumentParser(); p.add_argument("--output", default="outputs/bnci2014_001_baseline_trajectory_identifiability_v0")
a = p.parse_args(); g = decide(a.output); report(a.output, "NOT_RUN" if not g["conditional_gru"]["run"] else "PENDING")
