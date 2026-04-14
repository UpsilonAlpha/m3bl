#!/usr/bin/env python3
"""
QM/MM workflow (xTB) with automatic QM region selection.

Pipeline:
1. Generate onechain PDB from trajectory
2. Select QM residues (metal coordination)
3. Build QM/MM system
4. Run short QM/MM MD
"""

from __future__ import annotations

import argparse
import os
from typing import List, Set, Dict, Any

from ash import (
    Fragment,
    OpenMMTheory,
    QMMMTheory,
    xTBTheory,
    OpenMM_MD,
)

from preprocess import onechain

# ==============================
# Constants
# ==============================

BACKBONE_ATOMS = {
    "N", "H", "H1", "H2", "H3",
    "CA", "HA", "HA2", "HA3",
    "C", "O", "OXT",
}

AMINO_ACIDS = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
    "THR","TRP","TYR","VAL",
}

DEFAULT_FORCEFIELDS = [
    "amber14/protein.ff14SB.xml",
    "amber14/tip3p.xml",
    "openff_LIG.xml",
]

# ==============================
# QM Region Builder
# ==============================

def build_qm_region(mm: OpenMMTheory, target_residues: List[int]):
    """Construct QM atom list and boundary exclusions."""

    qm_atoms: Set[int] = set()
    boundary_excluded: Set[int] = set()

    for res in mm.topology.residues():
        if int(res.id) in target_residues:
            print(f"[QM] Including residue {res.name} {res.id}")

            for atom in res.atoms():
                if atom.name not in BACKBONE_ATOMS:
                    qm_atoms.add(atom.index)

                if res.name not in AMINO_ACIDS:
                    boundary_excluded.add(atom.index)

    print(f"[INFO] QM atoms: {len(qm_atoms)}")
    return sorted(qm_atoms), sorted(boundary_excluded)


# ==============================
# Pipeline
# ==============================

def run(args) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    #print("[STEP] Generating onechain structure...")
    #target_residues = onechain(args.input)

    with open(workdir / "preprocess" / "preprocess.json") as f:
        data = json.load(f)

    qm_residues = data["results"].get("qm_residues")
    result["qm_residues"] = qm_residues

    print("[STEP] Building MM system...")
    mm = OpenMMTheory(
        xmlfiles=DEFAULT_FORCEFIELDS,
        pdbfile=args.input,
        autoconstraints=None,
        periodic=True,
        rigidwater=True,
    )
    result["mm"] = mm

    fragment = Fragment(pdbfile=args.input)
    result["fragment"] = fragment

    print("[STEP] Building QM region...")
    qm_atoms, boundary_excluded = build_qm_region(mm, target_residues)
    result["qm_atoms"] = qm_atoms
    result["boundary_excluded"] = boundary_excluded

    print("[STEP] Initializing QM/MM...")
    xtb = xTBTheory(xtbmethod=args.method, numcores=cores)
    qmmm = QMMMTheory(
        qm_theory=xtb,
        mm_theory=mm,
        fragment=fragment,
        qmatoms=qm_atoms,
        excludeboundaryatomlist=boundary_excluded,
        embedding="electrostatic",
        printlevel=1,
    )
    result["qmmm"] = qmmm

    print("[STEP] Running QM/MM MD...")
    OpenMM_MD(
        fragment=fragment,
        theory=qmmm,
        timestep=args.timestep,
        simulation_time=args.time,
        traj_frequency=50,
        temperature=args.temp,
        integrator='LangevinMiddleIntegrator',
        trajfilename="QM_MM",
        coupling_frequency=1,
        charge=args.charge,
        mult=args.mult,
    )

    result["trajfile"] = "QM_MM.dcd"
    
    save_run(result, args, "qmmm_results.json")
    print("[DONE] QM/MM workflow complete")
    return result


# ==============================
# CLI
# ==============================

def add_arguments():
    parser = argparse.ArgumentParser(description="QM/MM xTB workflow")

    parser.add_argument("input", help="Input PDB")

    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--mult", type=int, default=2)

    parser.add_argument("--method", default="GFN1")

    parser.add_argument("--temp", type=float, default=300)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--time", type=float, default=10)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = run(args)

    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()