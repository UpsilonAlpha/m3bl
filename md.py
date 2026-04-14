#!/usr/bin/env python3
"""
Molecular dynamics workflow (argparse + checkpointing).

Features:
- Optional ligand workflow
- Automatic checkpointing (skip completed steps)
- Flexible CLI
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict

from ash import (
    OpenMMTheory,
    Fragment,
    OpenMM_Opt,
    OpenMM_box_equilibration,
    OpenMM_MD,
    MDtraj_imagetraj,
)

from preprocess import (
    save_run,
    merge_ligand,
    find_coordinators,
)


# ==============================
# Defaults
# ==============================

DEFAULT_FORCEFIELDS = [
    "amber14/protein.ff14SB.xml",
    "amber14/tip3p.xml",
    "openff_LIG.xml",
]

# ==============================
# Unified run() function
# ==============================

def run(args) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    with open(workdir / "preprocess" / "preprocess.json") as f:
        data = json.load(f)
    
    constraints = data["results"]["constraints"]

    mm = OpenMMTheory(
        xmlfiles=DEFAULT_FORCEFIELDS,
        pdbfile=args.input,
        periodic=True,
        numcores=cores,
        autoconstraints='HBonds',
        constraints=constraints,
        rigidwater=True,
    )
    result["mm"] = mm

    print("[STEP] Minimization...")
    OpenMM_Opt(fragment=fragment, theory=mm, maxiter=500, tolerance=1)
        
    if not args.skip_npt:
        print("[STEP] NPT Equilibration...")
        OpenMM_box_equilibration(
            fragment=fragment,
            theory=mm,
            datafilename="nptsim.csv",
            numsteps_per_NPT=10000,
            volume_threshold=1.0,
            density_threshold=0.001,
            temperature=args.temp,
            timestep=args.npt_timestep,
            traj_frequency=100,
            trajfilename='relaxbox_NPT',
            trajectory_file_option='DCD',
            coupling_frequency=1,
        )
    
    if not args.skip_nvt:
        print("[STEP] Production MD...")
        OpenMM_MD(
            fragment=fragment,
            theory=mm,
            timestep=args.nvt_timestep,
            simulation_time=args.nvt_time,
            traj_frequency=args.nvt_time,
            temperature=args.temp,
            integrator='LangevinMiddleIntegrator',
            coupling_frequency=1,
            trajfilename='NVTtrajectory',
            trajectory_file_option='DCD',
            datafilename="nvtsim.csv",
        )
    
        print("[STEP] Reimaging trajectory...")
        MDtraj_imagetraj(
            "NVTtrajectory.dcd",
            "NVTtrajectory_lastframe.pdb",
            format='DCD',
        )

    result["trajfile"] = "NVTtrajectory.dcd"
    result["lastframe"] = "NVTtrajectory_lastframe.pdb"

    print("[DONE] Workflow complete")
    save_run(result, args, "md_results.json")
    return result


# ==============================
# CLI
# ==============================

def add_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MD workflow with checkpointing")

    parser.add_argument("input", help="Input protein PDB")

    parser.add_argument("--temp", type=float, default=300)
    parser.add_argument("--npt-timestep", type=float, default=0.001)
    parser.add_argument("--nvt-timestep", type=float, default=0.004)
    parser.add_argument("--nvt-time", type=float, default=500)

    # checkpoint / control
    parser.add_argument("--skip-npt", action="store_true")
    parser.add_argument("--skip-nvt", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()