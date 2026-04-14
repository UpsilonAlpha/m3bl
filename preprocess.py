#!/usr/bin/env python3
"""
Protein/Ligand preprocessing utilities (pipeline-ready).
"""

from __future__ import annotations

import argparse
import json
import os
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
    "ZN", "MG",#"MN","FE","CO","NI","CU","CD","CA","NA","K","CS","RB","SR","BA"
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
def save_run(result: Dict[str, Any], args, filename: str = "results.json") -> str:
    clean_result = {}

    for k, v in result.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean_result[k] = v
        elif isinstance(v, (list, dict)):
            clean_result[k] = v
        else:
            # Skip non-serializable objects (e.g. Fragment)
            continue

    args_dict = vars(args).copy()

    # Normalize metals (ensure uppercase)
    if "metals" in args_dict:
        args_dict["metals"] = [m.upper() for m in args_dict["metals"]]

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "mode": getattr(args, "mode", "unknown"),
    }

    output = {
        "inputs": args_dict,
        "results": clean_result,
        "metadata": metadata,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[INFO] Results written to {filename}")
    return str(filename)


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


# ==============================
# One-chain conversion
# ==============================

def dep_onechain(input_pdb: str, output_pdb: str = "onechain.pdb") -> str:
    with open(input_pdb) as f:
        lines = f.readlines()

    cryst1 = [l for l in lines if l.startswith("CRYST1")]
    conect = [l for l in lines if l.startswith("CONECT")]
    serials = [int(l[6:11]) for l in lines if l.startswith(("ATOM", "HETATM"))]

    pdb = PDBFile(input_pdb)
    positions = pdb.positions

    new_top = Topology()
    chainA = new_top.addChain("A")

    atom_map = {}
    res_counter = 1

    for chain in pdb.topology.chains():
        for res in sorted(chain.residues(), key=lambda r: r.index):
            new_res = new_top.addResidue(res.name, chainA, id=str(res_counter))
            res_counter += 1

            for atom in res.atoms():
                atom_map[atom] = new_top.addAtom(atom.name, atom.element, new_res)

    for bond in pdb.topology.bonds():
        new_top.addBond(atom_map[bond[0]], atom_map[bond[1]])

    protein_res = [r for r in chainA.residues() if r.name in AMINO_ACIDS]
    last_protein = protein_res[-1]

    het_counter = int(last_protein.id) + 1
    for r in chainA.residues():
        if r.name not in AMINO_ACIDS:
            r.id = str(het_counter)
            het_counter += 1

    serial_idx = 0
    last_serial = None

    with open(output_pdb, "w") as f:
        f.writelines(cryst1)

        for res in chainA.residues():
            for atom in res.atoms():
                pos = positions[atom.index]
                serial = serials[serial_idx]
                last_serial = serial
                serial_idx += 1

                record = "HETATM" if res.name not in AMINO_ACIDS else "ATOM  "

                f.write(
                    f"{record}{serial:>5} {atom.name:<4} {res.name:>3} A{res.id:>4}   "
                    f"{pos.x*10:8.3f}{pos.y*10:8.3f}{pos.z*10:8.3f}\n"
                )

            if res == last_protein:
                f.write(f"TER   {last_serial+1:>5}      {res.name:<3} A{res.id:>4}\n")
                last_serial += 1

        f.write(f"TER   {last_serial+1:>5}      {res.name:<3} A{res.id:>4}\n")
        f.writelines(conect)
        f.write("END\n")

    return find_coordinators(output_pdb, metals=METALS)

def onechain(input_pdb: str, output_pdb: str = "onechain.pdb"):
    from openmm.app import PDBFile, Topology

    # -----------------------------
    # Read original file (metadata)
    # -----------------------------
    with open(input_pdb) as f:
        lines = f.readlines()

    cryst1_lines = [l for l in lines if l.startswith("CRYST1")]
    conect_lines = [l for l in lines if l.startswith("CONECT")]

    # -----------------------------
    # Load with OpenMM
    # -----------------------------
    pdb = PDBFile(input_pdb)
    positions = pdb.positions

    new_top = Topology()
    chainA = new_top.addChain("A")

    atom_map = {}
    res_counter = 1

    # -----------------------------
    # Rebuild topology (single chain)
    # -----------------------------
    for chain in pdb.topology.chains():
        for res in sorted(chain.residues(), key=lambda r: r.index):
            new_res = new_top.addResidue(res.name, chainA, id=str(res_counter))
            res_counter += 1

            for atom in res.atoms():
                new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                atom_map[atom] = new_atom

    # Copy bonds (not written, but keeps consistency if needed later)
    for bond in pdb.topology.bonds():
        new_top.addBond(atom_map[bond[0]], atom_map[bond[1]])

    # -----------------------------
    # Identify protein residues
    # -----------------------------
    protein_residues = [r for r in chainA.residues() if r.name in AMINO_ACIDS]
    last_protein = protein_residues[-1]

    # -----------------------------
    # Renumber HETATM residues
    # -----------------------------
    het_counter = int(last_protein.id) + 1
    for r in chainA.residues():
        if r.name not in AMINO_ACIDS:
            r.id = str(het_counter)
            het_counter += 1

    # -----------------------------
    # Write PDB
    # -----------------------------
    serial = 1
    last_res = None

    with open(output_pdb, "w") as f:
        # Preserve CRYST1
        for line in cryst1_lines:
            f.write(line)

        for res in chainA.residues():
            last_res = res

            for atom in res.atoms():
                pos = positions[atom.index]

                record = "HETATM" if res.name not in AMINO_ACIDS else "ATOM  "

                f.write(
                    f"{record}{serial:>5} {atom.name:<4} {res.name:>3} A{res.id:>4}   "
                    f"{pos.x*10:8.3f}{pos.y*10:8.3f}{pos.z*10:8.3f}\n"
                )
                serial += 1

            # TER after protein block
            if res == last_protein:
                f.write(
                    f"TER   {serial:>5}      {res.name:<3} A{res.id:>4}\n"
                )
                serial += 1

        # FINAL TER (important for downstream tools)
        if last_res is not None:
            f.write(
                f"TER   {serial:>5}      {last_res.name:<3} A{last_res.id:>4}\n"
            )
            serial += 1

        # Preserve CONECT exactly
        for line in conect_lines:
            f.write(line)

        f.write("END\n")

    print(f"[INFO] Wrote {output_pdb}")

    # -----------------------------
    # Return coordinators
    # -----------------------------
    return find_coordinators(output_pdb, metals=METALS)

    
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

        residue_min_dist = {}

        for i in range(len(atoms)):
            if i == m:
                continue

            # skip other metals
            if elems[i].upper() in metals:
                boundary_excluded.add(i)
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
                if (
                    res.chain.id == chain
                    and int(res.id) == resid
                ):

                    qm_residue_dict.setdefault(chain, {})[resid] = resn

                    for a in res.atoms():
                        if a.name not in BACKBONE_ATOMS:
                            qm_atoms.add(a.index)

                        # non-protein → exclude from boundary creation
                        if res.name not in AMINO_ACIDS:
                            boundary_excluded.add(a.index)

            # Also constrain metal–ligand bond
            boundary_excluded.add(m)

    # Convert to sorted lists
    qm_atoms = sorted(qm_atoms)
    boundary_excluded = sorted(boundary_excluded)

    return qm_residue_dict, bondconstraints, qm_atoms, boundary_excluded

# ==============================
# PIPELINE ENTRY
# ==============================

def run(args) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    metals = {m.upper() for m in args.metals}

    if args.ligand:
        print("[STEP] Preparing ligand...")
        pdb_file = merge_ligand(args.input, args.ligand, args.model, args.charge)
        extraxmlfile = "openff_LIG.xml"
    else:
        pdb_file = args.input
        extraxmlfile = None
    result["merged_pdb"] = pdb_file


    if not args.skip_solvate:
        print("[STEP] Solvating system...")
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
        pdb_file = "finalsystem.pdb"
    else:
        pdb_file = args.input
        fragment = Fragment(pdbfile=args.input)

    result["solvated_pdb"] = pdb_file
    result["fragment"] = fragment


    print("[STEP] Detecting coordination constraints...")
    qm_residues, constraints, qm_atoms, boundary_excluded_atoms = find_coordinators(pdb_file)
    result["qm_residues"] = qm_residues
    result["constraints"] = constraints
    result["qm_atoms"] = qm_atoms
    result["boundary_excluded_atoms"] = boundary_excluded_atoms

    save_run(result, args, "preprocess_results")

    return result
