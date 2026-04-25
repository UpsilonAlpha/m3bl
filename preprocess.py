#!/usr/bin/env python3
"""
Protein/Ligand preprocessing utilities (pipeline-ready).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, Any
from pathlib import Path
from datetime import datetime


import numpy as np
from rdkit import Chem
from openmm.app import PDBFile, Topology

from ash import (
    Fragment,
    OpenMM_Modeller,
    small_molecule_parameterizer,
    merge_pdb_files,
)

# ==============================
# Constants
# ==============================

#RESIDUE_VARIANTS = {"A": {80: "HIE"}}
RESIDUE_VARIANTS = {}


METALS = {
    "ZN",# "MG","MN","FE","CO","NI","CU","CD","CA","NA","K","CS","RB","SR","BA"
}

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

DONOR_ATOMS = {
    "N","O","S",
    "ND1","NE2","OE1","OE2","OD1","OD2",
    "OG","OG1","SG","SD",
}



# ==============================
# Utility
# ==============================


def save_result(
    result: Dict[str, Any],
    args,
    filename: str = "results.json",
    section: str = None,
) -> str:
    """
    Clean, merge, and save results into a shared JSON file.

    - Removes non-serializable objects
    - Stores inputs + metadata
    - Merges into existing JSON (no overwrite)
    - Supports sectioned pipeline output (preprocess/md/qmmm)

    Parameters
    ----------
    result : dict
        Output from run()
    args : argparse.Namespace
        CLI arguments
    filename : str
        JSON file path
    section : str
        Section name (e.g. "preprocess", "md", "qmmm")

    Returns
    -------
    str
        Path to JSON file
    """

    path = Path(filename)

    # ==============================
    # 1. Clean result (JSON-safe)
    # ==============================
    clean_result = {}

    for k, v in result.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean_result[k] = v
        elif isinstance(v, (list, dict)):
            clean_result[k] = v
        else:
            # Skip non-serializable objects (Fragment, OpenMMTheory, etc.)
            continue

    # ==============================
    # 2. Serialize args
    # ==============================
    args_dict = vars(args).copy()

    if "metals" in args_dict and args_dict["metals"] is not None:
        args_dict["metals"] = [m.upper() for m in args_dict["metals"]]

    # ==============================
    # 3. Build payload
    # ==============================
    payload = {
        "inputs": args_dict,
        "results": clean_result,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": getattr(args, "mode", None),
            "command": getattr(args, "command", None),
        },
    }

    # ==============================
    # 4. Load existing JSON
    # ==============================
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # ==============================
    # 5. Merge into section
    # ==============================
    if section:
        data.setdefault(section, {})
        data[section].update(payload)
    else:
        data.update(payload)

    # ==============================
    # 6. Atomic write
    # ==============================
    tmp_path = path.with_suffix(".tmp")

    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)

    tmp_path.replace(path)

    print(f"[INFO] Results updated → {path}")
    return str(path)


def load_section(filename: str, section: str) -> Dict[str, Any]:
    with open(filename, "r") as f:
        data = json.load(f)
    return data.get(section, {})

# ==============================
# Ligand Preparation
# ==============================

def merge_ligand(protein_pdb: str, ligand_file: str, model: int = 1, charge: int = 0) -> Dict[str, Any]:
    if ligand_file.endswith(".pdbqt"):

        smiles, idx, coords = None, [], []
        model_index = 0

        with open(ligand_file) as f:
            for line in f:
                if model_index != model:
                    if line.startswith("MODEL"):
                        model_index += 1
                    continue

                if line.startswith("ENDMDL"):
                    break

                if line.startswith("REMARK SMILES ") and "IDX" not in line:
                    smiles = line.split("SMILES ")[1].strip()
                elif line.startswith("REMARK SMILES IDX"):
                    idx += list(map(int, line.split()[3:]))
                elif line.startswith(("ATOM", "HETATM")):
                    coords.append(tuple(map(float, [line[30:38], line[38:46], line[46:54]])))

        if smiles is None:
            raise ValueError("No SMILES found in ligand file")

        mol = Chem.MolFromSmiles(smiles)
        mapping = {idx[i]-1: idx[i+1]-1 for i in range(0, len(idx), 2)}

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, coords[mapping[i]])

        mol.AddConformer(conf)
        mol = Chem.AddHs(mol, addCoords=True)

        Chem.MolToMolFile(mol, "ligand.mol")

        small_molecule_parameterizer(
            forcefield_option="OpenFF",
            molfile="ligand.mol",
            charge=charge,
        )

        merge_pdb_files(protein_pdb, "LIG.pdb")
        return "merged.pdb"

    elif ligand_file.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(ligand_file)
        mol = Chem.AddHs(mol, addCoords=True)
        Chem.MolToMolFile(mol, "ligand.mol")

        
        small_molecule_parameterizer( forcefield_option="OpenFF", molfile="ligand.mol", charge=charge)

        return protein_pdb

    else:
        print("Not a valid file format for ligand: Choose either .pdbqt or .mol")
    
# ==============================
# Coordination
# ==============================

def find_coordinators(
    pdb_file: str,
    cutoff: float = 2.6,
    metals=METALS,
    exclude_residues=None,
):
    pdb = PDBFile(pdb_file)
    topology = pdb.topology
    atoms = list(topology.atoms())

    coords = np.array([[p.x, p.y, p.z] for p in pdb.positions]) * 10.0  # nm → Å

    elems = [a.element.symbol if a.element else "" for a in atoms]
    names = [a.name for a in atoms]

    # Chain-aware residue info
    resids = [int(a.residue.id) for a in atoms]
    resnames = [a.residue.name for a in atoms]
    chains = [a.residue.chain.id for a in atoms]

    metal_idx = [i for i, e in enumerate(elems) if e.upper() in metals]

    qm_atoms = set()
    boundary_excluded = set()
    qm_residue_dict = {}  # OpenMM-style: {"A": {80: "HIS"}}
    bondconstraints = []

    print("\n===== Metal Coordination Analysis =====")
    print(f"Cutoff: {cutoff:.2f} Å")

    for m in metal_idx:
        print(f"\nMetal {elems[m]} at {chains[m]}:{resids[m]} (atom index {m})")
        qm_residue_dict.setdefault(chains[m], {})[resids[m]] = resnames[m]

        residue_min_dist = {}

        for i in range(len(atoms)):
            if i == m:
                continue

            # donor atom filter
            if all(d not in names[i] for d in DONOR_ATOMS):
                continue

            dist = np.linalg.norm(coords[i] - coords[m])
            if dist > cutoff:
                continue

            key = (chains[i], resids[i], resnames[i])

            if key not in residue_min_dist or dist < residue_min_dist[key][0]:
                residue_min_dist[key] = (dist, i, names[i])

        sorted_res = sorted(residue_min_dist.items(), key=lambda x: x[1][0])

        print("Coordinating residues:")
        for (chain, resid, resn), (dist, idx, aname) in sorted_res:
            
            bondconstraints.append([m, idx])

            # Skip excluded residues
            if exclude_residues and chain in exclude_residues:
                if resid in exclude_residues[chain]:
                    continue

            print(
                f"{resn:>3} {chain}:{resid:<4} ({aname:<4}) | "
                f"{dist:5.2f} Å | atom index {idx}"
            )

            # Build QM atoms (all non-backbone atoms of residue)
            for res in topology.residues():
                if (res.chain.id == chain and int(res.id) == resid):
                    qm_residue_dict.setdefault(chain, {})[resid] = resn

    return qm_residue_dict, bondconstraints





def run(args) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    metals = {m.upper() for m in args.metals}

    if args.ligand:
        print("[STEP] Preparing ligand...")
        pdb_file = merge_ligand(args.input, args.ligand, args.model, args.charge)
        extraxmlfile = "openff_LIG.xml"
        result["ligand_xml"] = Path(extraxmlfile).resolve()
    else:
        pdb_file = args.input

    result["merged_pdb"] = Path(pdb_file).resolve()


    if not args.skip_solvate:
        print("[STEP] Solvating system...")
        if args.implicit:
            modeller, fragment = OpenMM_Modeller(
                pdbfile=pdb_file,
                implicit=True,
                forcefield="Amber14",
                implicit_solvent_xmlfile="implicit/gbn2.xml",
                residue_variants=RESIDUE_VARIANTS,
                use_higher_occupancy=True,
                extraxmlfile=extraxmlfile,
            )
        else:
            modeller, fragment = OpenMM_Modeller(
                pdbfile=pdb_file,
                forcefield="Amber14",
                watermodel="tip3p",
                pH=8.0,
                solvent_padding=10.0,
                ionicstrength=0.1,
                pos_iontype='Na+',
                neg_iontype='Cl-',
                residue_variants=RESIDUE_VARIANTS,
                use_higher_occupancy=True,
                extraxmlfile=extraxmlfile,
            )
        
        pdb_file = Path("finalsystem.pdb").resolve()
            
    else:
        pdb_file = args.input

    result["final_pdb"] = Path(pdb_file).resolve()



    print("[STEP] Detecting coordination constraints...")
    qm_residues, constraints = find_coordinators(pdb_file, args.cutoff, metals=metals)
    result["qm_residues"] = qm_residues
    result["constraints"] = constraints

    
    save_result(result, args, filename="../results.json", section="preprocess")
    

    return result
