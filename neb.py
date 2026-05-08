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


def run(args) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    qmmm = load_section(args.input, "qmmm")
    preprocess = load_section(args.input, "preprocess")

    pdbfile = qmmm["results"]["lastframe"]
    qm_residues = preprocess["results"].get("qm_residues")

    numimages=8

    print(qm_residues)

    print("[STEP] Building MM system...")
    mm = OpenMMTheory(
        xmlfiles=DEFAULT_FORCEFIELDS,
        pdbfile=pdbfile,
        autoconstraints=None,
        periodic=True,
        rigidwater=False,
    )

    print("[STEP] Building QM region...")
    qm_atoms, boundary_excluded = build_qm_region(pdbfile, qm_residues)
    result["qm_atoms"] = qm_atoms
    result["boundary_excluded"] = boundary_excluded

    #Theory to use for NEB
    xtb = xTBTheory(xtbmethod="GFN2", numcores=args.cores, runmode='library')

    ################################################
    # Defining reactant and product ASH fragments
    #################################################
    react=Fragment(pdbfile=pdbfile, charge=0, mult=1)
    prod=Fragment(pdbfile=args.product, charge=0, mult=1)

    qmmm = QMMMTheory(
        qm_theory=xtb,
        mm_theory=mm,
        fragment=react,
        qmatoms=qm_atoms,
        excludeboundaryatomlist=boundary_excluded,
        embedding="electrostatic",
        printlevel=1,
    )

    #Run NEB to find saddlepoint. Returns an ASH Results object
    NEB_result = NEB(reactant=react, product=prod, theory=qmmm, images=numimages, runmode='parallel', numcores=args.cores)
    print(NEB_result)

    #Optional NumFreq job on saddlepoint to confirm that a saddlepoint was found.
    NumFreq(theory=xtbcalc, fragment=NEB_result.saddlepoint_fragment)
