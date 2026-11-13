"""Write ND-LAr + TMS LArCV files for GrapPA, with cross-detector particle matching from truth.

For every spill, only tracks are saved (true particle shape == track, see below):
  - each SPINE reco particle in ND-LAr (reco points, cm -> mm) becomes one LArCV fragment. It is
    linked to its true particle with SPINE's reco -> truth match (reco.match_ids[0], the best
    overlap) and kept if that true particle's shape is track. Reco particles with no truth match
    are dropped (and counted).
  - each TMS reco track (Reco_Tree TrackHitPos, mm) becomes one LArCV fragment. It is kept unless
    its true particle was matched (Method 2.5) to a SPINE true particle whose shape is not track;
    TMS-only particles have no SPINE shape, so their reco tracks are kept.
  - every fragment gets a unique fragment_id (particle.id),
  - group_id is one per true particle: reco particles (in either detector) pointing to the same
    true particle share a group, and an ND-LAr true particle shares the group of a Truth_Spill
    particle when Method 2.5 of match_spine_to_truth_spill.py matched the two.

Matching (see TRUTH_SPILL_MATCHING.md):
  Method 2: SPINE true interaction <-> Truth_Spill vertex by vertex x/y/z (within VERTEX_TOL_MM).
  Method 2.5: within each matched vertex, SPINE particle <-> Truth_Spill particle by track ID,
            confirmed by PDG and birth position (within PARTICLE_TOL_MM).
  TMS reco track -> Truth_Spill particle via Truth_Info.RecoTrackPrimaryParticleIndex.

Run where both spine and larcv are importable (e.g. the spine-nb container).
"""
import glob
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import uproot
import yaml
from larcv import larcv

sys.path.insert(0, '/sdf/data/neutrino/software/spine/src/')
from spine.driver import Driver

VERTEX_TOL_MM = 1.0
PARTICLE_TOL_MM = 1.0


class TMSData:
    """Truth_Spill (vertices, particles), Truth_Info (reco-track truth link) and Reco_Tree (reco hits)."""

    def __init__(self, filename):
        f = uproot.open(filename)
        self.spill = f["Truth_Spill"].arrays(
            ["SpillNo", "TrueVtxGlobalID", "TrueVtxX", "TrueVtxY", "TrueVtxZ",
             "VertexGlobalID", "TrackId", "PDG", "BirthPosition"], library="np")
        self.info = f["Truth_Info"].arrays(
            ["SpillNo", "RecoTrackN", "RecoTrackPrimaryParticleIndex"], library="np")
        # Reco_Tree lines up entry by entry with Truth_Info; TrackHitPos[track][hit] = (x, y, z, t) in mm
        self.reco = f["Reco_Tree"].arrays(["nTracks", "nHits", "TrackHitPos"], library="np")
        self.n_entries = len(self.spill["SpillNo"])

    def vertex_xyz(self, entry):
        return np.stack([self.spill["TrueVtxX"][entry], self.spill["TrueVtxY"][entry],
                         self.spill["TrueVtxZ"][entry]], axis=1)

    def reco_tracks(self, entry):
        """All TMS reco tracks in a Truth_Spill entry: list of (info_entry, k, true particle index, hits (N,3) mm)."""
        spill_no = self.spill["SpillNo"][entry]
        out = []
        for i in np.where(self.info["SpillNo"] == spill_no)[0]:
            for k in range(int(self.reco["nTracks"][i])):
                n = int(self.reco["nHits"][i][k])
                hits = np.asarray(self.reco["TrackHitPos"][i][k][:n, :3])
                p_idx = int(self.info["RecoTrackPrimaryParticleIndex"][i][k]) if k < self.info["RecoTrackN"][i] else -1
                out.append((int(i), k, p_idx, hits))
        return out


def match_vertices(spine_data, tms, tol_mm=VERTEX_TOL_MM):
    """Method 2: {SPINE ixn id: (Truth_Spill entry, vertex index)} for interactions within tol_mm of a vertex."""
    tms_xyz = [tms.vertex_xyz(e) for e in range(tms.n_entries)]
    out = {}
    for ixn in spine_data["truth_interactions"]:
        pos = np.asarray(ixn.vertex) * 10.0  # cm -> mm
        best = None
        for entry, xyz in enumerate(tms_xyz):
            if len(xyz) == 0:
                continue
            dist = np.linalg.norm(xyz - pos, axis=1)
            k = int(dist.argmin())
            if dist[k] <= tol_mm and (best is None or dist[k] < best[2]):
                best = (entry, k, float(dist[k]))
        if best is not None:
            out[int(ixn.id)] = best[:2]
    return out


def match_particles(spine_data, tms, vertex_matches, tol_mm=PARTICLE_TOL_MM):
    """Method 2.5: {SPINE particle id: Truth_Spill particle index} for particles with the same track ID,
    PDG and birth position (within tol_mm) at a Method 2 vertex."""
    by_ixn = {}
    for p in spine_data["truth_particles"]:
        by_ixn.setdefault(int(p.interaction_id), {})[int(p.track_id)] = p
    out = {}
    for ixn_id, (entry, k) in vertex_matches.items():
        gid = tms.spill["TrueVtxGlobalID"][entry][k]
        spine_parts = by_ixn.get(ixn_id, {})
        for idx in np.where(tms.spill["VertexGlobalID"][entry] == gid)[0]:
            p = spine_parts.get(int(tms.spill["TrackId"][entry][idx]))
            if p is None or int(p.pdg_code) != int(tms.spill["PDG"][entry][idx]):
                continue
            birth = np.asarray(tms.spill["BirthPosition"][entry][idx][:3], dtype=float)
            if np.linalg.norm(np.asarray(p.position) * 10.0 - birth) <= tol_mm:  # cm -> mm
                out[int(p.id)] = int(idx)
    return out


def fill_fragment(out_larcv, vsa, vox3dmeta, points_mm, fragment_id, group_id, shape, pdg=None, track_id=None):
    """One LArCV particle + voxel set (same pattern as create_LArCV_minerva_2x2.py)."""
    particle = larcv.Particle(shape)
    particle.id(int(fragment_id))
    particle.group_id(int(group_id))
    if pdg is not None:
        particle.pdg_code(int(pdg))
    if track_id is not None:
        particle.track_id(int(track_id))

    # A voxel set corresponds to all the voxels for one cluster
    voxelset = larcv.VoxelSet()
    voxelset.id(int(fragment_id))
    for x, y, z in points_mm:
        voxelid = vox3dmeta.id(float(x), float(y), float(z))
        if voxelid != larcv.kINVALID_VOXELID:  # skip points outside the box
            voxelset.add(larcv.Voxel(voxelid))

    vsa.insert(voxelset)
    entry_clust3d = out_larcv.get_data("cluster3d", "pcluster")
    entry_clust3d.set(vsa, vox3dmeta)
    entry_particles = out_larcv.get_data("particle", "corrected")
    entry_particles.append(particle)


TRACK_SHAPE = 1  # SPINE / LArCV semantic shape label for tracks


def process_one_spill(spine_data, tms, entry, particle_matches, out_larcv, vsa, vox3dmeta,
                      size_threshold=5, verbose=False):
    """Write the ND-LAr reco tracks and TMS reco tracks of one spill; returns a summary dict."""
    # group_id: one integer per true particle. A Truth_Spill particle is keyed ("tms", index);
    # a SPINE true particle uses its matched Truth_Spill key if Method 2.5 matched it, else ("spine", id).
    group_of = {}

    def group_id_for(key):
        if key not in group_of:
            group_of[key] = len(group_of)
        return group_of[key]

    truth_by_id = {int(t.id): t for t in spine_data["truth_particles"]}
    tms_idx_to_truth = {tms_idx: spine_id for spine_id, tms_idx in particle_matches.items()}

    fragment_counter = 0
    n_nd = n_nd_matched = n_nd_no_truth = n_nd_not_track = n_nd_small = 0
    truth_used = Counter()
    for r in spine_data["reco_particles"]:
        t = truth_by_id.get(int(r.match_ids[0])) if len(r.match_ids) else None
        if t is None:  # no true particle to check the shape against: dropped
            n_nd_no_truth += 1
            continue
        if int(t.shape) != TRACK_SHAPE:
            n_nd_not_track += 1
            continue
        if len(r.points) < size_threshold:
            n_nd_small += 1
            continue
        tms_idx = particle_matches.get(int(t.id))
        key = ("tms", tms_idx) if tms_idx is not None else ("spine", int(t.id))
        truth_used[int(t.id)] += 1
        group_id = group_id_for(key)
        fill_fragment(out_larcv, vsa, vox3dmeta, np.asarray(r.points) * 10.0,  # cm -> mm
                      fragment_counter, group_id, larcv.ShapeType_t.kShapeTrack,
                      pdg=t.pdg_code, track_id=t.track_id)
        if verbose:
            print(f"    ND-LAr reco particle {r.id} -> true particle {t.id} -> "
                  f"fragment {fragment_counter} group {group_id}" + (" (matched to TMS)" if tms_idx is not None else ""))
        fragment_counter += 1
        n_nd += 1
        n_nd_matched += tms_idx is not None

    n_tms = n_tms_shared = n_tms_not_track = 0
    nd_groups = set(group_of.values())
    for info_entry, k, p_idx, hits in tms.reco_tracks(entry):
        if len(hits) == 0:
            continue
        spine_id = tms_idx_to_truth.get(p_idx) if p_idx >= 0 else None
        if spine_id is not None and int(truth_by_id[spine_id].shape) != TRACK_SHAPE:
            n_tms_not_track += 1
            continue
        key = ("tms", p_idx) if p_idx >= 0 else ("tms_reco", info_entry, k)
        group_id = group_id_for(key)
        pdg = int(tms.spill["PDG"][entry][p_idx]) if p_idx >= 0 else None
        track_id = int(tms.spill["TrackId"][entry][p_idx]) if p_idx >= 0 else None
        fill_fragment(out_larcv, vsa, vox3dmeta, hits, fragment_counter, group_id,
                      larcv.ShapeType_t.kShapeTrack, pdg=pdg, track_id=track_id)
        shared = group_id in nd_groups
        if verbose:
            print(f"    TMS reco track (slice {info_entry}, #{k}) true particle {p_idx} -> "
                  f"fragment {fragment_counter} group {group_id}" + (" (shared with ND-LAr)" if shared else ""))
        fragment_counter += 1
        n_tms += 1
        n_tms_shared += shared

    return {"nd_fragments": n_nd, "nd_matched": n_nd_matched, "nd_no_truth": n_nd_no_truth,
            "nd_sharing_truth": sum(n for n in truth_used.values() if n > 1),
            "nd_skipped_not_track": n_nd_not_track, "nd_skipped_small": n_nd_small,
            "tms_fragments": n_tms, "tms_shared": n_tms_shared, "tms_skipped_not_track": n_tms_not_track,
            "groups": len(group_of)}


def setup_larcv(output_filename):
    out_larcv = larcv.IOManager(larcv.IOManager.kWRITE)
    out_larcv.set_out_file(output_filename)
    out_larcv.initialize()
    return out_larcv


def make_larcv_box(xmin=-4000.0, ymin=-3705.0, zmin=4000.0,
                   xmax=4000.0, ymax=1000.0, zmax=19000.0, voxel_size=3.72):
    # Bounding box around ND-LAr and TMS together, in mm (TMS coordinate frame; SPINE points
    # are converted cm -> mm). Active volumes (TMS_Constants.h):
    #   ND-LAr: x[-3478.48, 3478.48], y[-2166.71,  829.28], z[ 4179.24,  9135.88]
    #   TMS:    x[-3730.00, 3730.00], y[-3702.23,  997.77], z[11177.50, 18502.50]
    # Same box as Plotting_validation_TMS.ipynb, which converts these voxels back to mm.
    # Voxel size 3.72 mm (the ND-LAr / SPINE voxel pitch, 0.372 cm).
    xnum = int(math.ceil(abs((xmin - xmax) / voxel_size)))
    ynum = int(math.ceil(abs((ymin - ymax) / voxel_size)))
    znum = int(math.ceil(abs((zmin - zmax) / voxel_size)))

    # the "meta" maps individual 3D voxels within a 3D array to physical positions in the detector
    vox3dmeta = larcv.Voxel3DMeta()
    vox3dmeta.set(xmin, ymin, zmin, xmax, ymax, zmax, xnum, ynum, znum)
    # the voxelsetarray contains all the voxels for one event
    vsa = larcv.VoxelSetArray()

    return vsa, vox3dmeta


def pair_spills(driver, tms, verbose=False):
    """SPINE event -> Truth_Spill entry, by majority vote of the Method 2 vertex matches.
    Also returns each event's spine data and vertex matches so they are computed once."""
    pairs = []
    for ev in range(len(driver)):
        spine_data = driver.process(ev)
        vm = match_vertices(spine_data, tms)
        if not vm:
            if verbose:
                print(f"  SPINE event {ev}: no vertex matched to Truth_Spill, skipping")
            continue
        entry, n = Counter(e for e, _ in vm.values()).most_common(1)[0]
        vm = {ixn: ek for ixn, ek in vm.items() if ek[0] == entry}
        if verbose:
            print(f"  SPINE event {ev} <-> Truth_Spill entry {entry} ({n} vertices matched)")
        pairs.append((ev, entry, spine_data, vm))
    return pairs


if __name__ == "__main__":
    work_directory = "/global/homes/j/jessiem/track_matching_reco_GNN/data/"
    spine_dir = "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-mlreco/MiniProdN5p3_NDComplex_FHC.spine.full.sanddrift/MLRECO_SPINE/0000000"
    tms_dir = "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-tms-reco/MiniProdN5p3_NDComplex_FHC.tmsreco.full.sanddrift.v4/TMSRECO/0000000/"
    training_output = work_directory + "larcv/"
    validation_output = work_directory + "larcv/larcv_validation_set/"
    rewrite_output = False   # True: overwrite existing output files
    TRAIN_FRACTION = 0.8     # fraction of the usable file pairs written to training; the rest go to validation
    N_FILES = None           # int: only use the first N_FILES *.TMSRECO.root files (sorted by name); None: all
    VERBOSE = False

    os.makedirs(training_output, exist_ok=True)
    os.makedirs(validation_output, exist_ok=True)

    tms_files = sorted(glob.glob(os.path.join(tms_dir, "*.TMSRECO.root")))
    print(f"Found {len(tms_files)} TMS file(s)" +
          (f"; using the first {min(N_FILES, len(tms_files))}" if N_FILES is not None else ""))
    if N_FILES is not None:
        tms_files = tms_files[:N_FILES]
    removed_per_spill = []  # (file number, spill, TMS reco tracks removed as non-track, ND-LAr reco without truth)

    # Pair each TMS file with its SPINE file first, so the train/validation split is made on the
    # files that will actually be processed
    file_pairs = []  # (tms_file, base_name, filenum, spine_file)
    for tms_file in tms_files:
        base_name = os.path.basename(tms_file)
        match = re.search(r'\.(\d{7})\.', base_name)
        if not match:
            print(f"Cannot find a 7-digit file number in {base_name}, skipping")
            continue
        filenum = match.group(1)
        spine_candidates = glob.glob(os.path.join(spine_dir, f"*spine*.{filenum}.MLRECO_SPINE.hdf5"))
        if len(spine_candidates) != 1:
            print(f"Expected 1 SPINE file for {filenum}, found {len(spine_candidates)}, skipping")
            continue
        file_pairs.append((tms_file, base_name, filenum, spine_candidates[0]))

    # Train/validation split: round((1 - TRAIN_FRACTION) * n) validation files, spread evenly through the
    # (name-sorted) list so both sets cover the whole file range
    n_pairs = len(file_pairs)
    n_val = int(round((1 - TRAIN_FRACTION) * n_pairs))
    val_ids = set(np.round(np.linspace(0, n_pairs - 1, n_val)).astype(int).tolist()) if n_val > 0 else set()
    print(f"{n_pairs} usable file pair(s): {n_pairs - n_val} training, {n_val} validation")

    for f_id, (tms_file, base_name, filenum, spine_file) in enumerate(file_pairs):
        out_dir = validation_output if f_id in val_ids else training_output
        output_file = os.path.join(out_dir, f"out_{base_name}")
        if os.path.isfile(output_file) and not rewrite_output:
            print(output_file, "already exists, NOT re-writing (set rewrite_output = True to overwrite)")
            continue

        print(f"\n{base_name} + {os.path.basename(spine_file)} -> {output_file}")
        cfg = f'''
        base:
          verbosity: warning
        build:
          mode: both
          fragments: false
          particles: true
          interactions: true
          units: cm
        post:
          match:
            particle:
              match_mode: both
        io:
          reader:
            file_keys: {spine_file}
            skip_unknown_attrs: True
            name: hdf5
        '''
        driver = Driver(yaml.safe_load(cfg))
        tms = TMSData(tms_file)
        pairs = pair_spills(driver, tms, verbose=VERBOSE)

        my_larcv = setup_larcv(output_file)
        for spill_id, (ev, entry, spine_data, vertex_matches) in enumerate(pairs):
            particle_matches = match_particles(spine_data, tms, vertex_matches)
            vsa, vox3dmeta = make_larcv_box()
            summary = process_one_spill(spine_data, tms, entry, particle_matches, my_larcv, vsa, vox3dmeta,
                                        verbose=VERBOSE)
            print(f"  spill {spill_id} (SPINE event {ev}, Truth_Spill entry {entry}): "
                  f"{summary['nd_fragments']} ND-LAr reco tracks ({summary['nd_matched']} matched to TMS, "
                  f"{summary['nd_sharing_truth']} sharing a true particle; skipped {summary['nd_no_truth']} without "
                  f"a true particle, {summary['nd_skipped_not_track']} non-track, {summary['nd_skipped_small']} small), "
                  f"{summary['tms_fragments']} TMS reco tracks ({summary['tms_shared']} share a group with ND-LAr; "
                  f"skipped {summary['tms_skipped_not_track']} non-track), {summary['groups']} groups")
            removed_per_spill.append((filenum, spill_id, summary["tms_skipped_not_track"], summary["nd_no_truth"]))
            my_larcv.set_id(0, 0, spill_id)  # run, subrun, entry
            my_larcv.save_entry()
        my_larcv.finalize()
        print(f"  wrote {len(pairs)} spill(s) to {output_file}")

    print("TMS reco tracks (true particle matched to a non-track SPINE particle) and ND-LAr reco particles with no truth match")
    print("\nRemoved per spill:")
    print("  file     spill  TMS    ND-LAr")
    for filenum, spill_id, n_tms, n_nd in removed_per_spill:
        print(f"  {filenum}  {spill_id:5d}  {n_tms:69d}  {n_nd:41d}")
    if removed_per_spill:
        print(f"  {'total':>14}  {sum(x[2] for x in removed_per_spill):69d}  {sum(x[3] for x in removed_per_spill):41d}")
