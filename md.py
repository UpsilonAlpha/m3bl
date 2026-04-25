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
import json
from pathlib import Path
from typing import Any, Dict

from openmm import XmlSerializer
from openmm.app import PDBFile, Topology, ForceField

from ash import (
    OpenMMTheory,
    Fragment,
    OpenMM_Opt,
    OpenMM_box_equilibration,
    OpenMM_MD,
    MDtraj_imagetraj
)

from preprocess import (
    save_result,
    load_section,
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

    preprocess = load_section(args.input, "preprocess")
    
    constraints = preprocess["results"]["constraints"]
    pdb_file = preprocess["results"]["final_pdb"]

    if args.implicit:

        pdb = PDBFile(pdb_file)
        topology = pdb.topology
        forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml", "implicit/gbn2.xml", "openff_LIG.xml")

        system = forcefield.createSystem(topology, soluteDielectric=1.0, solventDielectric=78.5)

        # Assuming 'system' is your OpenMM System object
        with open('system.xml', 'w') as output:
            output.write(XmlSerializer.serialize(system))

        mm = OpenMMTheory(
            xmlsystemfile='system.xml',
            pdbfile=pdb_file,
            periodic=False,
            numcores=args.cores,
            autoconstraints='HBonds',
            constraints=constraints,
            nonbondedMethod_noPBC='CutoffNonPeriodic',
            nonbonded_cutoff_noPBC=20,
        )

    else:
        mm = OpenMMTheory(
            xmlfiles=DEFAULT_FORCEFIELDS,
            pdbfile=pdb_file,
            periodic=True,
            numcores=args.cores,
            autoconstraints='HBonds',
            constraints=constraints,
            rigidwater=True,
        )

    fragment = Fragment(pdbfile=pdb_file)

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

    result["trajfile"] = Path("NVTtrajectory.dcd").resolve()
    result["lastframe"] = Path("NVTtrajectory_lastframe.pdb").resolve()

    print("[DONE] Workflow complete")
    save_results(result, args,  filename="../results.json", section="md")
    return result
