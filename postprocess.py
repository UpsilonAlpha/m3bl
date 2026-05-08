import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis import align
from MDAnalysis.coordinates.DCD import DCDWriter


import numpy as np
import networkx as nx
from collections import defaultdict
import polars as pl
import matplotlib.pyplot as plt
import os
from pathlib import Path
from ash import *
import mdtraj

from preprocess import (
    save_result,
    load_section,
)

# =========================
# Helper functions
# =========================

def residue_id(atom):
    """Return a unique residue identifier tuple"""
    r = atom.residue
    return (r.segid, r.resid, r.resname)

def centroid(atoms):
    """Compute centroid of a group of atoms"""
    return atoms.positions.mean(axis=0)

def ring_normal(atoms):
    """Compute approximate normal vector of aromatic ring"""
    coords = atoms.positions
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    n = np.cross(v1, v2)
    return n / np.linalg.norm(n)


def resolve_overlaps(pos, min_dist=0.08, max_iter=100):
    nodes = list(pos.keys())
    for _ in range(max_iter):
        moved = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                d = pos[n1] - pos[n2]
                dist = np.linalg.norm(d)

                if dist < min_dist:
                    direction = d / (dist + 1e-9)
                    shift = (min_dist - dist) * 0.5 * direction
                    pos[n1] += shift
                    pos[n2] -= shift
                    moved = True
        if not moved:
            break
    return pos
# =========================
# Main pipeline
# =========================

def run(args):

    result: Dict[str, Any] = {}

    if args.qmmm:
        qmmm = load_section(args.input, "qmmm")
        md = load_section(args.input, "md")

        pdbfile = md["results"]["lastframe"]
        trajectory = qmmm["results"]["trajfile"]
        optimization = qmmm["results"]["xyztraj"]

        u = mda.Universe(pdbfile, optimization)

        # 2. Setup the DCD writer
        output_dcd = 'output.dcd'
        with DCDWriter(output_dcd, u.atoms.n_atoms) as W:
            # 3. Iterate through frames and write to DCD
            for frame in u.trajectory:
                W.write(u)

    else:
        md = load_section(args.input, "md")
        pdbfile = md["results"]["lastframe"]
        trajectory = md["results"]["trajfile"]
        csv = md["results"]["csv"]

        df = pl.read_csv(csv)

        steps = df['#"Step"']
        time = df['Time (ps)']
        temperature = df['Temperature (K)']

        #Creating data list
        data_list=[temperature]
        #Density and Volume only present for NPT
        try:
            density = df['Density (g/mL)']
            data_list.append(density)
            volume = df['Box Volume (nm^3)']
            data_list.append(volume)
        except:
            pass

        #Looping over data_list and plot
        for pd_col in data_list:
            np_array = pd_col.to_numpy()
            label=pd_col.name
            label_no_unit = label.split('(')[0].replace(' ','')
            print(label_no_unit)
            eplot = ASH_plot(label, num_subplots=1, x_axislabel="Steps", y_axislabel=label)
            eplot.addseries(0, x_list=steps.to_numpy(), y_list=np_array, label=label, color='blue', line=True, scatter=True)
            eplot.savefig(label_no_unit)
    


    #Loading using mdtraj
    system = mdtraj.load(pdbfile)
    traj = mdtraj.load(trajectory, top=system)

    print(f"This trajectory contains {traj.n_frames} frames")

    #Calculating full RMSD (flawed) w.r.t. first frame
    rmsd_all= mdtraj.rmsd(traj, traj[0], 0)

    #Sub-system selection: Defining heavy atoms

    #Selection: All non-H atoms (also flawed because of solvent)
    heavy_atoms = [atom.index for atom in traj.topology.atoms if atom.element.symbol != 'H']

    #Selection: All non-H atoms in protein
    heavy_protein_atoms = traj.topology.select("protein and (element !=  H)")

    #RMSD for heavy protein atoms  w.r.t. first frame
    rmsd_heavy_protein = mdtraj.rmsd(traj, traj[0], 0, atom_indices=heavy_protein_atoms)

    #Plotting using ASH-plot (matplotlib)
    x_label="Frames in trajectory"
    y_label="RMSD (nm)"
    filelabel="RMSD"
    eplot = ASH_plot(filelabel, num_subplots=1, x_axislabel=x_label, y_axislabel=y_label)
    eplot.addseries(0, x_list=traj.time, y_list=rmsd_heavy_protein, label=y_label, color='blue', line=True, scatter=True)
    eplot.savefig(filelabel)



    if args.redraw == False:
        # Load system
        u = mda.Universe(pdbfile, trajectory)

        protein = u.select_atoms("protein")

        # Define atom groups
        acidic = u.select_atoms("resname ASP GLU and name OD1 OD2 OE1 OE2")
        basic  = u.select_atoms("resname LYS ARG and name NZ NH1 NH2")

        aromatic = u.select_atoms("resname PHE TYR TRP HIS")
        hydrophobic = u.select_atoms("resname ALA VAL LEU ILE MET PHE TRP PRO")

        metals = u.select_atoms(f"name {' '.join(args.metals)}")
        donors = u.select_atoms("name N* O* S*")

        # Precompute aromatic atoms per residue
        rings = []
        for res in aromatic.residues:
            ring_atoms = res.atoms.select_atoms("name CG CD1 CD2 CE1 CE2 CZ NE1")
            if len(ring_atoms) >= 5:
                rings.append(ring_atoms)

        # Storage for interaction counts
        edge_counts = defaultdict(lambda: defaultdict(int))
        n_frames = len(u.trajectory)

        # =========================
        # Hydrogen bond analysis
        # =========================

        print("Calculating hydrogen bonds")

        hbond_analysis = HydrogenBondAnalysis(
            u,
            donors_sel="(protein or resname SUB) and (name N* or name O* or name S*) ",
            hydrogens_sel="(protein or resname SUB) and name H*",
            acceptors_sel="(protein or resname SUB) and (name O* or name N* or name S*)",
            d_a_cutoff=3.2,
            d_h_a_angle_cutoff=150,
        )
        hbond_analysis.run()

        # Convert results to Polars DataFrame
        df = pl.DataFrame(
            hbond_analysis.results.hbonds,
            schema=["frame", "donor", "hydrogen", "acceptor", "distance", "angle"]
        ).with_columns([
            pl.col("frame").cast(pl.Int64),
            pl.col("donor").cast(pl.Int64),
            pl.col("acceptor").cast(pl.Int64),
        ])

        # Count H-bond occurrences per frame
        for frame, group in df.group_by("frame", maintain_order=True):

            seen = set()

            donors = group["donor"].to_numpy()
            acceptors = group["acceptor"].to_numpy()

            for d, a in zip(donors, acceptors):
                res1 = residue_id(u.atoms[d])
                res2 = residue_id(u.atoms[a])

                # prevent self-interactions
                if res1 != res2:
                    edge = tuple(sorted([res1, res2]))
                    seen.add(edge)

            for edge in seen:
                edge_counts[edge]["hbond"] += 1            


        # =========================
        # Frame-wise interactions
        # =========================

        print("Calculating other interactions")
        frame_edges = []
        for ts in u.trajectory:

            seen_frame = {
                "salt": set(),
                "pi_pi": set(),
                "cation_pi": set(),
                "coordination": set(),
                "hydrophobic": set()
            }

            coord_residues = set()

            # ---- Coordination bonds ----
            donor_atoms = u.select_atoms("name N* O* S*")
            for m in metals:
                distances = np.linalg.norm(donor_atoms.positions - m.position, axis=1)
                nearby = donor_atoms[distances < args.cutoff]
                for atom in nearby:
                    res = residue_id(atom)
                    coord_residues.add(res)
                    metal_res = ("METAL", m.index, m.name)
                    edge = tuple(sorted([metal_res, res]))
                    seen_frame["coordination"].add(edge)


            # ---- Salt bridges ----
            for a in acidic:
                for b in basic:
                    if residue_id(a) != residue_id(b):
                        if np.linalg.norm(a.position - b.position) < 4.0:
                            edge = tuple(sorted([residue_id(a), residue_id(b)]))
                            seen_frame["salt"].add(edge)
            
            # ---- Pi-Pi stacking ----
            for i, ring1 in enumerate(rings):
                c1 = centroid(ring1)
                n1 = ring_normal(ring1)

                for ring2 in rings[i+1:]:
                    c2 = centroid(ring2)
                    n2 = ring_normal(ring2)

                    dist = np.linalg.norm(c1 - c2)
                    angle = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1, 1)))

                    if dist < 5.0 and (angle < 30 or 60 < angle < 120):
                        edge = tuple(sorted([residue_id(ring1[0]), residue_id(ring2[0])]))
                        seen_frame["pi_pi"].add(edge)

            # ---- Cation–Pi ----
            for ring in rings:
                c = centroid(ring)
                for b in basic:
                    if residue_id(ring[0]) != residue_id(b):
                        if np.linalg.norm(c - b.position) < 6.0:
                            edge = tuple(sorted([residue_id(ring[0]), residue_id(b)]))
                            seen_frame["cation_pi"].add(edge)
            '''
            # ---- Hydrophobic contacts ----
            for i, a in enumerate(hydrophobic):
                for b in hydrophobic[i+1:]:
                    if residue_id(a) != residue_id(b):
                        if np.linalg.norm(a.position - b.position) < 4.0:
                            edge = tuple(sorted([residue_id(a), residue_id(b)]))
                            seen_frame["hydrophobic"].add(edge)
            '''
            # Accumulate counts
            for itype, edges in seen_frame.items():
                for edge in edges:
                    edge_counts[edge][itype] += 1

        # =========================
        # Build network
        # =========================

        print("Building interaction network")

        # build graph and dataframe
        G = nx.Graph()
        rows = []
        for edge, types in edge_counts.items():
            res1, res2 = edge
            for itype, count in types.items():
                occ = count / n_frames
                if occ > args.occupancy:
                    G.add_edge(
                        res1,
                        res2,
                        interaction=itype,
                        occupancy=occ,
                        color={
                            "hbond": "lightblue",
                            "salt": "blue",
                            "pi_pi": "purple",
                            "cation_pi": "red",
                            "coordination": "green",
                            "hydrophobic": "orange"
                        }[itype]
                    )

                    rows.append({
                        "res1_segid": res1[0],
                        "res1_resid": res1[1],
                        "res1_resname": res1[2],

                        "res2_segid": res2[0],
                        "res2_resid": res2[1],
                        "res2_resname": res2[2],

                        "interaction_type": itype,
                        "occupancy": occ,
                        "pair": f"{res1[2]}-{res2[2]}",
                    })

        # =========================
        # Save network plot and csv
        # =========================
        df_out = pl.DataFrame(rows)
        df_out = df_out.sort(["interaction_type", "occupancy"],descending=[False, True])
        df_out.write_csv("interaction_network.csv")

    else:
        print("Redrawing")
        G = nx.Graph()

        df = pl.read_csv("interaction_network.csv")

        for row in df.iter_rows(named=True):
            if row["occupancy"] > args.occupancy:

                res1 = (row["res1_segid"], row["res1_resid"], row["res1_resname"])
                res2 = (row["res2_segid"], row["res2_resid"], row["res2_resname"])


                G.add_edge(
                    res1,
                    res2,
                    interaction=row["interaction_type"],
                    occupancy=row["occupancy"],
                    color={
                        "hbond": "lightblue",
                        "salt": "blue",
                        "pi_pi": "purple",
                        "cation_pi": "red",
                        "coordination": "green",
                        "hydrophobic": "orange"
                    }[row["interaction_type"]]
                )


    plt.figure(figsize=(10, 10))

    labels = {
        node: f"{node[2]}{node[1]}"  # resname + resid
        for node in G.nodes
    }

    node_type = dict()
    for n in G.nodes:
        resname = n[2]

        if resname in ["ASP", "GLU"]:
            node_type[n] = "#b3c7ff"

        elif resname in ["LYS", "ARG", "HIS"]:
            node_type[n] = "#ffb3b3"

        elif resname in ["PHE", "TYR", "TRP"]:
            node_type[n] = "#ffd699"

        elif resname in ["HOH", "WAT"]:
            node_type[n] = "#b3e6ff"

        elif resname in ["ZN", "MG", "FE", "CU"]:  # adjust for your system
            node_type[n] = "#b3ffb3"

        elif resname in ["LIG", "SUB"]:  # your ligand naming
            node_type[n] = "#d9b3ff"

        else:
            node_type[n] =  "#dddddd"

    
    node_colors = [node_type[n] for n in G.nodes]

    node_sizes = [
        100 + 100 * sum(d["occupancy"] for _,_,d in G.edges(n, data=True))
        for n in G.nodes
    ]
    pos = nx.spring_layout(G, seed=42, k=args.spring_constant/ np.sqrt(len(G.nodes)), iterations=args.iterations)
    pos = resolve_overlaps(pos)

    colors = [d["color"] for _, _, d in G.edges(data=True)]
    edge_widths = [2 * d["occupancy"] for _,_,d in G.edges(data=True)]
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="white", edge_color=colors, labels=labels, font_weight="bold", font_size=3, width=edge_widths)
    plt.savefig("interaction_network.png", dpi=500)
    plt.close()

    # =========================
    # RMSF calculation
    # =========================

    print("Calculating RMSF")

    # Align trajectory to first frame using Cα atoms
    align.AlignTraj(
        u,
        u,  # align to itself
        select="protein and name CA",
        in_memory=True
    ).run()

    rmsf = RMSF(protein.select_atoms("name CA")).run()

    # Map RMSF to B-factor
    for res, val in zip(protein.residues, rmsf.results.rmsf):
        for atom in res.atoms:
            atom.tempfactor = val

    # Save structure
    output_pdb = "rmsf_bfactor.pdb"
    protein.write(output_pdb)

    print(f"Saved RMSF structure to {output_pdb}")
    print("Saved interaction network to interaction_network.png")

    result["rmsf_pdb"] = Path("rmsf_bfactor.pdb").resolve().as_posix()
    result["network_graph"] = Path("interaction_network.png").resolve().as_posix()
    result["network_csv"] = Path("interaction_network.csv").resolve().as_posix()


    return output_pdb



#ffmpeg -framerate 10 -pattern_type glob -i "network_frames/frame_*.png" -c:v libx264 -pix_fmt yuv420p out.mp4