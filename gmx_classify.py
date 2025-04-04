#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import re
import glob
import shutil
import pickle
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor, wait

##############################################################################
#                               USER CONFIG                                  #
##############################################################################

DEBUG = False

N_CHUNKS = 6001  # Adjust as needed
MDP_FILE = "step5_production.mdp"
TOPOL    = "topol.top"
INDEX    = "index.ndx"
EQUI_PREFIX = "step4.1_equilibration"

OUTPUT_DIR = "analysis_output"

# We store per-residue φ/ψ histograms in 360x360 arrays (1° bins).
PHI_PSI_BINS = 360  # bin size 1°, range -180..179

# Note: Our system now has 17 residues in total:
# MDTraj residue index 0 = ACE cap
# MDTraj residue indices 1 to 15 = original 15 residues (to be labeled as 0–14)
# MDTraj residue index 16 = NME cap
# We will analyze only residues with topology index 1 <= r.index < 16,
# and then subtract 1 from their index when saving output.

##############################################################################
#                           GLOBAL DATA STRUCTURES                           #
##############################################################################

# Dictionary: shifted residue id (0 to 14) -> 2D histogram (360x360)
PHI_PSI_HISTS = {}

GROUP_DIR_PREFIX = "group_"

##############################################################################
#                   GROMACS RUN + COMMAND WRAPPERS                           #
##############################################################################

def run_command(cmd: str):
    """Simple wrapper around subprocess.run to run shell commands."""
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if p.returncode != 0:
        print(f"ERROR:\n{p.stderr}")
        sys.exit(1)

def grompp_and_mdrun(chunk_i, prev_prefix):
    """Set up and run GROMACS for a single chunk."""
    step_prefix = f"step{chunk_i}"
    gro_in = f"{prev_prefix}.gro"
    cpt_in = f"{prev_prefix}.cpt"

    grompp_cmd = f"gmx grompp -f {MDP_FILE} -c {gro_in} -p {TOPOL} -n {INDEX} -o {step_prefix}.tpr"
    if os.path.exists(cpt_in):
        grompp_cmd += f" -t {cpt_in}"
    run_command(grompp_cmd)

    mdrun_cmd = f"gmx mdrun -s {step_prefix}.tpr -deffnm {step_prefix}"
    run_command(mdrun_cmd)

def run_chunk(chunk_i, prev_prefix):
    """
    Simple wrapper to run a single GROMACS chunk,
    preserving the original code structure.
    """
    grompp_and_mdrun(chunk_i, prev_prefix)

##############################################################################
#                         FILE MOVING AND RESTART LOGIC                      #
##############################################################################

def move_group_files(group_number):
    """
    Move all GROMACS output files for the given group_number
    (which covers chunks group_start..group_end) into a subdirectory
    named group_{group_number}.
    """
    group_start = (group_number - 1) * 1000 + 1
    group_end = group_number * 1000
    dest_folder = f"{GROUP_DIR_PREFIX}{group_number}"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for chunk in range(group_start, group_end + 1):
        pattern = f"step{chunk}*"
        for fpath in glob.glob(pattern):
            basename = os.path.basename(fpath)
            # Skip the MDP file or any stray matches
            if basename == "step5_production.mdp":
                continue
            prefix = f"step{chunk}"
            if len(basename) <= len(prefix) or basename[len(prefix)] not in {'.', '_'}:
                continue
            os.rename(fpath, os.path.join(dest_folder, basename))

def get_last_completed_chunk():
    """
    Determine the last completed chunk by searching for step*.gro
    in the base directory and group_* directories.
    """
    chunks = []
    base_files = glob.glob("step*.gro")
    if base_files:
        for f in base_files:
            m = re.search(r"step(\d+)\.gro", f)
            if m:
                chunks.append(int(m.group(1)))
        if chunks:
            return max(chunks), f"step{max(chunks)}"

    group_dirs = glob.glob(f"{GROUP_DIR_PREFIX}*")
    if group_dirs:
        groups = []
        for d in group_dirs:
            m = re.search(rf"{GROUP_DIR_PREFIX}(\d+)", d)
            if m:
                groups.append(int(m.group(1)))
        if groups:
            last_group = max(groups)
            last_chunk = last_group * 1000
            gro_files = glob.glob(os.path.join(f"{GROUP_DIR_PREFIX}{last_group}", "step*.gro"))
            if gro_files:
                nums = []
                for f in gro_files:
                    m = re.search(r"step(\d+)\.gro", os.path.basename(f))
                    if m:
                        nums.append(int(m.group(1)))
                if nums:
                    max_chunk = max(nums)
                    prev_prefix = os.path.join(f"{GROUP_DIR_PREFIX}{last_group}", f"step{max_chunk}")
                    return max_chunk, prev_prefix

    return 0, EQUI_PREFIX

##############################################################################
#                         PHI/PSI HISTOGRAM FUNCTIONS                        #
##############################################################################

def ensure_residue_hist(shifted_res_id):
    """
    Make sure we have a 360×360 histogram array for the given shifted residue ID
    (i.e. 0 to 14) in the global PHI_PSI_HISTS dictionary.
    """
    global PHI_PSI_HISTS
    if shifted_res_id not in PHI_PSI_HISTS:
        PHI_PSI_HISTS[shifted_res_id] = np.zeros((PHI_PSI_BINS, PHI_PSI_BINS), dtype=np.int64)

def update_phi_psi_histograms(traj):
    """
    Compute φ and ψ angles for every frame in 'traj' and update
    global histograms for each residue's angles.
    
    Since our capped system now has ACE (index 0) and NME (index 16) we
    analyze only residues with topology index 1 to 15 (i.e., original residues 15–29).
    For output, we shift the residue number by -1 so that they appear as 0–14.
    """
    global PHI_PSI_HISTS
    phi_i, phi_v = md.compute_phi(traj)
    psi_i, psi_v = md.compute_psi(traj)

    # Loop over φ entries; only process if residue index is between 1 and 15.
    for i in range(len(phi_i)):
        residue = traj.topology.atom(phi_i[i][1]).residue
        if residue.index < 1 or residue.index >= 16:
            continue  # skip ACE and NME (or any extra)
        shifted_res_id = residue.index - 0  # Actually I removed the shift
        ensure_residue_hist(shifted_res_id)

    # Similarly for ψ entries
    for i in range(len(psi_i)):
        residue = traj.topology.atom(psi_i[i][1]).residue
        if residue.index < 1 or residue.index >= 16:
            continue
        shifted_res_id = residue.index - 1
        ensure_residue_hist(shifted_res_id)

    n_frames = traj.n_frames
    for frame_idx in range(n_frames):
        # Build a dictionary for ψ angles in this frame by original residue index
        psi_dict = {}
        for psi_i_idx in range(len(psi_i)):
            residue = traj.topology.atom(psi_i[psi_i_idx][1]).residue
            if residue.index < 1 or residue.index >= 16:
                continue
            shifted_res_id = residue.index - 1
            angle_deg = np.degrees(psi_v[frame_idx, psi_i_idx])
            psi_dict[shifted_res_id] = angle_deg

        for phi_i_idx in range(len(phi_i)):
            residue = traj.topology.atom(phi_i[phi_i_idx][1]).residue
            if residue.index < 1 or residue.index >= 16:
                continue
            shifted_res_id = residue.index - 1
            phi_deg = np.degrees(phi_v[frame_idx, phi_i_idx])
            if shifted_res_id not in psi_dict:
                continue
            psi_deg = psi_dict[shifted_res_id]

            # Convert angles to bins: from [-180,180) to [0,359]
            phi_bin = int(phi_deg + 180) % 360
            psi_bin = int(psi_deg + 180) % 360

            PHI_PSI_HISTS[shifted_res_id][phi_bin, psi_bin] += 1

def save_phi_psi_histograms():
    """
    Save the current histograms (one file per residue) in analysis_output/res_{shifted_res_id}/
    as .npy files.
    """
    global PHI_PSI_HISTS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for shifted_res_id, hist in PHI_PSI_HISTS.items():
        res_dir = os.path.join(OUTPUT_DIR, f"res_{shifted_res_id:02d}")
        os.makedirs(res_dir, exist_ok=True)
        out_npy = os.path.join(res_dir, f"res_{shifted_res_id:02d}_phi_psi_hist.npy")
        np.save(out_npy, hist)

def create_ramachandran_heatmaps(chunk_i):
    """
    For each residue, create a 2D heatmap from the histogram.
    Overwrite each time (keep only the latest).
    """
    global PHI_PSI_HISTS
    for shifted_res_id, hist in PHI_PSI_HISTS.items():
        res_dir = os.path.join(OUTPUT_DIR, f"res_{shifted_res_id:02d}")
        os.makedirs(res_dir, exist_ok=True)

        plt.figure()
        plt.imshow(hist.T, origin='lower', extent=(-180, 180, -180, 180), aspect='auto')
        plt.colorbar(label="Counts")
        plt.xlabel("Phi (deg)")
        plt.ylabel("Psi (deg)")
        # Use the shifted residue id in title (which now runs 0-14)
        plt.title(f"Residue {shifted_res_id} Ramachandran at step {chunk_i}")
        out_png = os.path.join(res_dir, f"res_{shifted_res_id:02d}_rama_plot.png")
        plt.savefig(out_png, dpi=120)
        plt.close()

def snapshot_phi_psi_histograms(chunk_i):
    """
    Copy the current .npy histogram files into a snapshots subdir,
    with chunk_i in the name for backup.
    """
    snap_dir = os.path.join(OUTPUT_DIR, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    global PHI_PSI_HISTS
    for shifted_res_id in PHI_PSI_HISTS:
        src_dir = os.path.join(OUTPUT_DIR, f"res_{shifted_res_id:02d}")
        src_file = os.path.join(src_dir, f"res_{shifted_res_id:02d}_phi_psi_hist.npy")
        if os.path.exists(src_file):
            dst_file = os.path.join(snap_dir, f"hist_res_{shifted_res_id:02d}_chunk_{chunk_i}.npy")
            shutil.copy(src_file, dst_file)

##############################################################################
#                    ASYNCHRONOUS ANALYSIS FUNCTION                          #
##############################################################################

def analyze_chunk_bg(chunk_i):
    """
    Perform the φ/ψ histogram analysis in the background for chunk_i,
    skipping chunks < 1000. Then downsample the trajectory to save space.
    """
    if chunk_i < 1000:
        print(f"Chunk {chunk_i}: Skipping analysis (chunk < 1000).")
        return

    try:
        step_prefix = f"step{chunk_i}"
        xtc_file = f"{step_prefix}.xtc"
        gro_file = f"{step_prefix}.gro"

        if not (os.path.exists(xtc_file) and os.path.exists(gro_file)):
            print(f"WARNING: Missing {xtc_file} or {gro_file}, skipping chunk {chunk_i}.")
            return

        # Load trajectory with MDTraj
        traj = md.load_xtc(xtc_file, top=gro_file)

        # Update the global φ/ψ histograms (only for residues 1-15, shifted to 0-14)
        update_phi_psi_histograms(traj)
        save_phi_psi_histograms()

        # Downsample trajectory to save space (keep 10% frames)
        ds_xtc = f"{step_prefix}_downsampled.xtc"
        downsample_xtc(traj, 0.1, ds_xtc)
        if os.path.exists(xtc_file):
            os.remove(xtc_file)
        print(f"Chunk {chunk_i}: Downsampled to {ds_xtc}, removed original XTC.")

        # Generate Ramachandran heatmaps every 100 chunks starting at 1050
        if chunk_i >= 1050 and (chunk_i - 50) % 100 == 0:
            print(f"Chunk {chunk_i}: Generating Ramachandran heatmaps...")
            create_ramachandran_heatmaps(chunk_i)

        # Snapshot histograms every 100 chunks (>=1000)
        if chunk_i >= 1000 and (chunk_i % 100 == 0):
            print(f"Chunk {chunk_i}: Saving snapshot of histograms...")
            snapshot_phi_psi_histograms(chunk_i)

        print(f"Chunk {chunk_i}: Analysis complete.")
    except Exception as e:
        print(f"ERROR analyzing chunk {chunk_i} in background: {e}")

def downsample_xtc(traj, fraction: float, out_xtc: str):
    """
    Save a fraction of frames (uniformly) from the trajectory.
    """
    n = traj.n_frames
    keep = int(np.ceil(n * fraction))
    idx = np.linspace(0, n - 1, keep, dtype=int)
    idx = np.unique(idx)
    ds = traj.slice(idx, copy=True)
    ds.save_xtc(out_xtc)

##############################################################################
#                                MAIN                                        #
##############################################################################

def load_existing_histograms():
    """
    On startup, load any existing per-residue .npy histogram files from
    analysis_output/res_{shifted_res_id:02d}/ and store them in PHI_PSI_HISTS.
    """
    global PHI_PSI_HISTS
    if not os.path.exists(OUTPUT_DIR):
        return
    subdirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("res_")]
    for sd in subdirs:
        m = re.search(r"res_(\d+)", sd)
        if not m:
            continue
        shifted_res_id = int(m.group(1))
        npy_file = os.path.join(OUTPUT_DIR, sd, f"res_{shifted_res_id:02d}_phi_psi_hist.npy")
        if os.path.exists(npy_file):
            try:
                hist = np.load(npy_file)
                PHI_PSI_HISTS[shifted_res_id] = hist
                print(f"Loaded existing histogram for residue {shifted_res_id}.")
            except Exception as e:
                print(f"Warning: could not load {npy_file}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_existing_histograms()

    last_chunk, _ = get_last_completed_chunk()
    current_group = ((last_chunk - 1) // 1000) + 1 if last_chunk > 0 else 0

    # Move finished groups if necessary
    all_files = glob.glob("step*.*")
    group_numbers = set()
    for f in all_files:
        m = re.search(r"step(\d+)", f)
        if m:
            chunk_num = int(m.group(1))
            group_num = ((chunk_num - 1) // 1000) + 1
            group_numbers.add(group_num)
    for group in sorted(group_numbers):
        if current_group > 0 and group < current_group:
            print(f"Startup: Moving finished group {group}.")
            move_group_files(group)

    start_chunk = last_chunk + 1
    if start_chunk > N_CHUNKS:
        print("All chunks already completed. Exiting.")
        return

    prev_prefix = EQUI_PREFIX if start_chunk == 1 else f"step{last_chunk}"
    print(f"Resuming from chunk {start_chunk} (previous completed chunk: {last_chunk}).")
    print(f"Starting main() with N_CHUNKS={N_CHUNKS}")

    group_futures = {}
    analysis_futures = []

    with ThreadPoolExecutor() as executor:
        for i in range(start_chunk, N_CHUNKS + 1):
            print(f"\n=== Now running chunk {i} ===")
            t0 = time.time()

            print(f"Running grompp_and_mdrun for chunk {i} with prev_prefix={prev_prefix}...")
            run_chunk(i, prev_prefix)
            t1 = time.time()
            sec = t1 - t0
            cpd = 86400.0 / sec
            print(f"Chunk {i} GROMACS step finished in {sec:.2f}s => {cpd:.2f} chunks/day")

            fut = executor.submit(analyze_chunk_bg, i)
            analysis_futures.append(fut)

            if i > 1000 and (i - 1) % 1000 == 0:
                group_number = (i - 1) // 1000
                print(f"Chunk {i}: Group boundary reached. Waiting for analysis of group {group_number} to finish...")
                wait(group_futures.get(group_number, []))
                print(f"Chunk {i}: Moving group {group_number} output files.")
                move_group_files(group_number)

            group_num = i // 1000
            group_futures.setdefault(group_num, []).append(fut)
            prev_prefix = f"step{i}"

        for future in analysis_futures:
            future.result()

    print("All chunks + analyses completed. Exiting now.")

if __name__ == "__main__":
    main()

