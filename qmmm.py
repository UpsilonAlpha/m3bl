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
from typing import List, Set, Dict, Any, Tuple
from openmm.app import PDBFile

from preprocess import (
    save_result,
    load_section,
)

from ash import *


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
    #"openff_LIG.xml",
]

# ==============================
# QM Region Builder
# ==============================



def build_qm_region(
    pdb_file: str,
    qm_residue_dict: Dict[str, Dict[int, str]],
    backbone_atoms=BACKBONE_ATOMS,
    amino_acids=AMINO_ACIDS,
):
    pdb = PDBFile(pdb_file)
    topology = pdb.topology

    qm_residue_dict = {
        chain: {str(k): v for k, v in residues.items()}
        for chain, residues in qm_residue_dict.items()
    }

    qm_atoms = set()
    boundary_excluded = set()

    print("\n===== QM Region Construction =====")

    for res in topology.residues():
        chain = res.chain.id
        resid = res.id.strip()   # ← FIXED

        if chain in qm_residue_dict and resid in qm_residue_dict[chain]:
            print(f"[QM] Including residue {res.name} {chain}:{resid}")

            for atom in res.atoms():
                if atom.name not in backbone_atoms:
                    qm_atoms.add(atom.index)

                if res.name not in amino_acids:
                    qm_atoms.add(atom.index)
                    boundary_excluded.add(atom.index)

    print(f"\n[INFO] QM atoms: {len(qm_atoms)}")
    print(f"[INFO] Boundary excluded atoms: {len(boundary_excluded)}")

    return sorted(qm_atoms), sorted(boundary_excluded)

# ==============================
# Pipeline
# ==============================

def run(args) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    preprocess = load_section(args.input, "preprocess")
    md = load_section(args.input, "md")


    pdbfile = md["results"]["lastframe"]
    qm_residues = preprocess["results"].get("qm_residues")
    print(qm_residues)

    print("[STEP] Building MM system...")
    mm = OpenMMTheory(
        xmlfiles=DEFAULT_FORCEFIELDS,
        pdbfile=pdbfile,
        autoconstraints=None,
        periodic=True,
        rigidwater=False,
    )

    fragment = Fragment(pdbfile=pdbfile)

    print("[STEP] Building QM region...")
    qm_atoms, boundary_excluded = build_qm_region(pdbfile, qm_residues)
    result["qm_atoms"] = qm_atoms
    result["boundary_excluded"] = boundary_excluded

    if args.interface == "xtb":
        print("[STEP] Initializing QM/MM...")
        xtb = xTBTheory(xtbmethod="GFN1", numcores=args.cores)
        qmmm = QMMMTheory(
            qm_theory=xtb,
            mm_theory=mm,
            fragment=fragment,
            qmatoms=qm_atoms,
            excludeboundaryatomlist=boundary_excluded,
            embedding="electrostatic",
            printlevel=1,
        )

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

        result["trajfile"] = Path("QM_MM.dcd").resolve().as_posix()
        result["lastframe"] = Path("QM_MM_lastframe.pdb").resolve().as_posix()
        save_results(result, args, filename="../results.json", section="qmmm")

    if args.interface == "pyscf":
        pyscf = PySCFTheory(scf_type="RKS", functional="wb97x-v", basis="def2-svp", solvation="ddCOSMO", solvation_eps=78)

        qmmm = QMMMTheory(
            qm_theory=pyscf,
            mm_theory=mm,
            fragment=fragment,
            qmatoms=qm_atoms,
            excludeboundaryatomlist=boundary_excluded,
            embedding="electrostatic",
            printlevel=1,
        )

        if args.actradius != 0:
            actregiondefine(mmtheory=mm, fragment=fragment, radius=args.actradius, originatom=boundary_excluded[0])
            actatoms = read_intlist_from_file("active_atoms")
        else:
            actatoms=qm_atoms


        waterconlist = getwaterconstraintslist(openmmtheoryobject=mm, atomlist=actatoms, watermodel='tip3p')
        waterconstraints = {'bond': waterconlist}


        Optimizer(fragment=fragment, theory=qmmm, ActiveRegion=True, actatoms=actatoms, maxiter=200,
            constraints=waterconstraints, charge=args.charge, mult=args.mult)

    return result
