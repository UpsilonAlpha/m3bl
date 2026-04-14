# **Project Documentation: Molecular Simulation Workflows**

This project provides a set of Python workflows for **protein-ligand preparation, solvation, bond constraint detection, and molecular dynamics simulations**, including **QM/MM workflows using xTB**. All scripts are designed with **argparse CLI** interfaces and **modular functions** for interoperability.

---

## **1. `preprocess.py` — Preprocessing Utilities**

### **Purpose**

* Prepare proteins and ligands for simulation.
* Solvate systems, generate one-chain PDBs, and detect metal coordination.
* Output: `merged.pdb` (ligand + protein), `finalsystem.pdb` (solvated system), or `onechain.pdb`.

### **Main Functions**

| Function                                                                        | Description                                                 | Inputs                                                                   | Outputs                                 |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------- |
| `prepare_ligand(protein_pdb, ligand_pdbqt, model, charge)`                      | Merge ligand into protein, parameterize ligand using OpenFF | `protein_pdb` (str), `ligand_pdbqt` (str), `model` (int), `charge` (int) | `merged.pdb` (str)                      |
| `solvate_protein(pdb_file)`                                                     | Solvate protein only                                        | `pdb_file` (str)                                                         | `finalsystem.pdb` (str), `Fragment`     |
| `solvate_system(pdb_file)`                                                      | Solvate protein-ligand system                               | `pdb_file` (str)                                                         | `finalsystem.pdb` (str), `Fragment`     |
| `onechain(input_pdb, output_pdb="onechain.pdb")`                                | Collapse all chains into a single chain                     | `input_pdb` (str)                                                        | `onechain.pdb` (str)                    |
| `find_coordinators(pdb_file, cutoff=2.6, return_residues=False, metals=METALS)` | Detect metal-coordinating residues or bond constraints      | `pdb_file` (str), `cutoff` (float), `metals` (set)                       | List of bond constraints or residue IDs |

### **CLI Modes**

```bash
# Mode 1: Solvate protein and find bond constraints
python preprocess.py protein input.pdb --cutoff 2.6 --metals ZN FE

# Mode 2: Prepare ligand, solvate system, find bond constraints
python preprocess.py ligand protein.pdb ligand.pdbqt 1 0 --cutoff 2.6

# Mode 3: Convert to one chain and find coordinating residues
python preprocess.py onechain input.pdb --cutoff 2.6 --metals ZN
```

---

## **2. `md_workflow.py` — Molecular Dynamics Workflow**

### **Purpose**

* Run MD simulations on a preprocessed system.
* Supports **energy minimization**, **NPT equilibration**, **production MD**, and **trajectory reimaging**.
* Can operate **without ligand preparation** if preprocessing is already done.
* Supports **checkpointing**: skips steps if outputs exist.

### **Workflow Steps**

1. Recalculate bond constraints (`find_coordinators`)
2. Build OpenMM system (`OpenMMTheory`) with constraints
3. Energy minimization (`OpenMM_Opt`)
4. NPT equilibration (`OpenMM_box_equilibration`)
5. Production MD (`OpenMM_MD`)
6. Reimage trajectory (`MDtraj_imagetraj`)

### **CLI Arguments**

| Argument        | Description                       |
| --------------- | --------------------------------- |
| `input`         | Input protein PDB (preprocessed)  |
| `--ligand`      | Optional ligand PDBQT             |
| `--model`       | Ligand model index                |
| `--charge`      | Ligand charge                     |
| `--cores`       | Number of CPU cores               |
| `--cutoff`      | Coordination cutoff (Å)           |
| `--temp`        | Simulation temperature (K)        |
| `--timestep`    | Simulation timestep (ps)          |
| `--time`        | Production simulation length (ps) |
| `--forcefields` | Force field XML files             |
| `--force`       | Recalculate all steps             |
| `--skip-min`    | Skip energy minimization          |
| `--skip-eq`     | Skip NPT equilibration            |
| `--skip-md`     | Skip production MD                |
| `--skip-image`  | Skip trajectory reimaging         |

### **Usage Examples**

```bash
# Full MD workflow with ligand
python md_workflow.py protein.pdb --ligand ligand.pdbqt --model 1 --charge 0 --cores 8

# Recalculate bond constraints and run minimization only
python md_workflow.py finalsystem.pdb --cores 8 --cutoff 2.6 --skip-eq --skip-md --skip-image

# Production MD only (skip minimization and equilibration)
python md_workflow.py finalsystem.pdb --skip-min --skip-eq
```

---

## **3. `qmmm_workflow.py` — QM/MM xTB Workflow**

### **Purpose**

* Perform short QM/MM MD simulations using **xTB QM region**.
* Automatically selects **QM atoms** based on metal coordination.
* Uses **OpenMM for MM** and `Fragment` objects.

### **Workflow Steps**

1. Generate onechain PDB (`onechain`)
2. Build MM system (`OpenMMTheory`)
3. Build QM region (`build_qm_region`)
4. Initialize QM/MM theory (`QMMMTheory`)
5. Run short QM/MM MD (`OpenMM_MD`)

### **CLI Arguments**

| Argument        | Description                            |
| --------------- | -------------------------------------- |
| `input`         | Input trajectory frame PDB             |
| `--pdb`         | QM/MM PDB file (default: onechain.pdb) |
| `--charge`      | System charge                          |
| `--mult`        | Spin multiplicity                      |
| `--cores`       | Number of CPU cores for xTB            |
| `--method`      | xTB method (GFN1, GFN2, etc.)          |
| `--temp`        | Simulation temperature (K)             |
| `--timestep`    | MD timestep (ps)                       |
| `--time`        | MD simulation length (ps)              |
| `--forcefields` | Force field XML files                  |

### **Usage Example**

```bash
# Run QM/MM MD on a trajectory frame
python qmmm_workflow.py frame.pdb --pdb onechain.pdb --charge 2 --mult 2 --cores 8 --time 10

# Change QM method to GFN2
python qmmm_workflow.py frame.pdb --method GFN2
```

---

## **Interoperability Notes**

* **Preprocessing outputs** (`merged.pdb`, `finalsystem.pdb`, `onechain.pdb`) are inputs for MD and QM/MM workflows.
* **Bond constraints** can be recalculated independently using `find_coordinators`.
* `Fragment` objects are always created from **existing solvated PDBs**.
* All scripts follow **modular `run(args)` design** for reuse in pipelines.

---

