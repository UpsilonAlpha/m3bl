#!/usr/bin/env python3
"""
Unified pipeline for protein/ligand preprocessing, MD, and QM/MM.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict

# Import local workflows
import preprocess
import md
import qmmm
import postprocess

METALS = {
    "ZN", #"MG","MN","FE","CO","NI","CU","CD","CA","NA","K","CS","RB","SR","BA"
}

def build_parser():
    parser = argparse.ArgumentParser(prog="m3bl")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # =========================================================
    # PREPROCESS
    # =========================================================
    p1 = subparsers.add_parser("preprocess", help="Prepare system")

    p1.add_argument("input", help="Input PDB file")
    p1.add_argument("--ligand", help="Ligand file (optional)")
    p1.add_argument("--model", type=int, default=1, help="Autodock model (optional)")
    p1.add_argument("--charge", type=int, default=0, help="Ligand charge (optional)")
    p1.add_argument("--cutoff", type=float, default=2.6, help="Cutoff for coordinating atoms in angstroms")
    p1.add_argument("--metals", nargs="+", default=["ZN"])
    p1.add_argument("--skip-solvate", action="store_true")
    p1.add_argument("--implicit", action="store_true")
    

    # =========================================================
    # MD
    # =========================================================
    p2 = subparsers.add_parser("md", help="Run molecular dynamics")

    p2.add_argument("input", help="Input preprocess_results file")
    p2.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p2.add_argument("--temp", type=float, default=300)
    p2.add_argument("--npt-timestep", type=float, default=0.001)
    p2.add_argument("--nvt-timestep", type=float, default=0.004)
    p2.add_argument("--nvt-time", type=float, default=500)
    p2.add_argument("--skip-npt", action="store_true")
    p2.add_argument("--skip-nvt", action="store_true")
    p2.add_argument("--implicit", action="store_true")
    p2.add_argument("--gentle", action="store_true")



    # =========================================================
    # QMMM
    # =========================================================
    p3 = subparsers.add_parser("qmmm", help="Run QM/MM")

    p3.add_argument("input", help="Input JSON file")
    p3.add_argument("--charge", type=int, required=True)
    p3.add_argument("--mult", type=int, required=True)
    p3.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p3.add_argument("--actradius", type=int, default=0)
    p3.add_argument("--interface", default="xtb")
    p3.add_argument("--temp", type=float, default=300)
    p3.add_argument("--timestep", type=float, default=0.001)
    p3.add_argument("--time", type=float, default=10)
    p3.add_argument("--skip-equilibrate", action="store_true")
    p3.add_argument("--skip-optimize", action="store_true")





    # =========================================================
    # NEB
    # =========================================================
    p4 = subparsers.add_parser("neb", help="Nudged elastic band simulation")

    p4.add_argument("input", help="Input PDB file")
    p4.add_argument("--trajectory", help="DCD file")
    p4.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))


    # =========================================================
    # POSTPROCESS
    # =========================================================
    p5 = subparsers.add_parser("postprocess", help="Analyse trajectory")

    p5.add_argument("input", help="Input PDB file")
    p5.add_argument("--trajectory", help="DCD file")
    p5.add_argument("--cutoff", type=float, default=2.6)
    p5.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p5.add_argument("--metals", nargs="+", default=["ZN"])
    p5.add_argument("--spring_constant", type=int, default=5)
    p5.add_argument("--iterations", type=int, default=200)
    p5.add_argument("--occupancy", type=float, default=0.1)
    p5.add_argument("--redraw", action="store_true")
    p5.add_argument("--qmmm", action="store_true")







    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # =========================================================
    # DISPATCH
    # =========================================================
    if args.command == "preprocess":
        result = preprocess.run(args)

    elif args.command == "md":
        result = md.run(args)

    elif args.command == "qmmm":
        result = qmmm.run(args)
    
    elif args.command == "postprocess":
        result = postprocess.run(args)

    else:
        parser.error("Unknown command")

    # =========================================================
    # OUTPUT
    # =========================================================
    if isinstance(result, dict):
        print("\n[RESULT]")
        for k, v in result.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()