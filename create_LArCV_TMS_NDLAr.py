import uproot
from larcv import larcv
import math
import numpy as np
import glob
import os
import re
import sys
import yaml
sys.path.insert(0, '/sdf/data/neutrino/software/spine/src/')

from spine.driver import Driver


class TMSData:
    def __init__(self, filename):
        self.file = uproot.open(filename)

        # Truth_Spill tree: one entry per spill; inner arrays indexed per particle/vertex in spill
        self.PDG = self.file["Truth_Spill"]["PDG"].array(library="np")
        self.TrueVtxPDG = self.file["Truth_Spill"]["TrueVtxPDG"].array(library="np")
        self.TrueVtxID = self.file["Truth_Spill"]["TrueVtxID"].array(library="np")
        self.VertexID = self.file["Truth_Spill"]["VertexID"].array(library="np")
        self.TrackID = self.file["Truth_Spill"]["TrackId"].array(library="np")
        self.TrueVtxN = self.file["Truth_Spill"]["TrueVtxN"].array(library="np")
        self.TrueVtxX = self.file["Truth_Spill"]["TrueVtxX"].array(library="np")
        self.TrueVtxY = self.file["Truth_Spill"]["TrueVtxY"].array(library="np")
        self.TrueVtxZ = self.file["Truth_Spill"]["TrueVtxZ"].array(library="np")
        self.TrueVtxT = self.file["Truth_Spill"]["TrueVtxT"].array(library="np")
        self.PositionTMSStart = self.file["Truth_Spill"]["PositionTMSStart"].array(library="np")
        self.PositionTMSEnd = self.file["Truth_Spill"]["PositionTMSEnd"].array(library="np")
        self.SpillNo_TruthSpill = self.file["Truth_Spill"]["SpillNo"].array(library="np")
        self.EventNo_TruthSpill = self.file["Truth_Spill"]["EventNo"].array(library="np")

        # Truth_Info tree: one entry per reconstructed track
        self.SpillNo = self.file["Truth_Info"]["SpillNo"].array(library="np")
        self.EventNo = self.file["Truth_Info"]["EventNo"].array(library="np")
        self.RunNo = self.file["Truth_Info"]["RunNo"].array(library="np")
        self.NeutrinoX4 = self.file["Truth_Info"]["NeutrinoX4"].array(library="np")
        self.RecoTrackN = self.file["Truth_Info"]["RecoTrackN"].array(library="np")
        self.RecoTrackPrimaryParticleIndex = self.file["Truth_Info"]["RecoTrackPrimaryParticleIndex"].array(library="np")
        self.RecoTrackPrimaryParticlePDG = self.file["Truth_Info"]["RecoTrackPrimaryParticlePDG"].array(library="np")
        self.RecoTrackPrimaryParticleVtxID = self.file["Truth_Info"]["RecoTrackPrimaryParticleVtxId"].array(library="np")
        self.TrueHitX = self.file["Truth_Info"]["TrueHitX"].array(library="np")
        self.TrueHitY = self.file["Truth_Info"]["TrueHitY"].array(library="np")
        self.TrueHitZ = self.file["Truth_Info"]["TrueHitZ"].array(library="np")

    def find_TMS_uniqueID(self, spill):
        """
        Build unique group IDs for TMS particles in a spill, keyed by vertex time + PDG.
        Unique ID = f"{round(TrueVtxT[spill][VertexID[i]], 3)}_{PDG[spill][i]}".
        Using vertex time + PDG ensures each particle species from a given interaction
        gets a distinct group, while still linking the same species across TMS and NDLAr.
        Returns: (tms_fullids list of unique id strings, unique_id_dict)
        """
        vtx_ids = self.VertexID[spill]

        unique_id_store = []

        for i in range(len(self.PDG[spill])):
            vtx_time = round(float(self.TrueVtxT[spill][vtx_ids[i]]), 3)
            pdg      = int(self.PDG[spill][i])
            unique_id_store.append(f"{vtx_time}_{pdg}")

        tms_vtx_times  = list(set(unique_id_store))
        unique_id_dict = {val: i for i, val in enumerate(set(unique_id_store))}

        return tms_vtx_times, unique_id_dict

    def process_one_TMS_entry(self, spill, out_larcv, all_uniqueID_dict, vsa_in, vox3dmeta, fragment_counter, verbose=True):
        """
        For each Truth_Info reconstructed track entry in the given spill, store
        TrueHitX/Y/Z (mm) as LArCV voxels.
        Unique ID = f"{round(TrueVtxT[spill][vtx_ids[0]], 3)}_{pdgs[0]}" — vertex time + PDG,
        matching find_TMS_uniqueID and find_NDLAr_uniqueIDs_remaining.
        vtx_ids[0] (RecoTrackPrimaryParticleVtxId) is the local vertex index in TMS.
        Groups with more than one TMS fragment are added to invalid_groups so the
        caller can exclude them from the crossing-particle count.
        """
        tms_indices = np.where(self.SpillNo == spill)[0]
        if verbose:
            print(f"  TMS spill {spill}: {len(tms_indices)} Truth_Info track entries")


        # Pre-count fragments per fullid so multi-fragment groups are known before any data is written
        frag_pre_counts = {}
        for idx in tms_indices:
            _vtx_ids = self.RecoTrackPrimaryParticleVtxID[idx]
            _pdgs    = self.RecoTrackPrimaryParticlePDG[idx]
            if len(_vtx_ids) == 0 or len(_pdgs) == 0:
                continue
            for k in range(len(_vtx_ids)):
                _vtx_local_idx = int(_vtx_ids[k])
                if _vtx_local_idx < 0 or _vtx_local_idx >= len(self.TrueVtxT[spill]):
                    continue
                _vtx_time = round(float(self.TrueVtxT[spill][_vtx_local_idx]), 3)
                _fullid   = f"{_vtx_time}_{int(_pdgs[k])}"
                if _fullid not in all_uniqueID_dict:
                    continue
                frag_pre_counts[_fullid] = frag_pre_counts.get(_fullid, 0) + 1

        invalid_groups = {fid for fid, n in frag_pre_counts.items() if n > 1}
        max_count = max(frag_pre_counts.values(), default=0)
        print(f"  [TMS] {len(tms_indices)} Truth_Info entries -> {len(frag_pre_counts)} unique fullids "
              f"(max fragments per fullid: {max_count}), "
              f"{len(invalid_groups)} group(s) with multiple fragments excluded"
              + (f": {sorted(invalid_groups)}" if invalid_groups else ""))
        """
        # Collect hits for each valid single-fragment group
        fragment_hits = {}  # fullid -> (hits_x, hits_y, hits_z) as np arrays
        for idx in tms_indices:
            _vtx_ids = self.RecoTrackPrimaryParticleVtxID[idx]
            _pdgs    = self.RecoTrackPrimaryParticlePDG[idx]
            if len(_vtx_ids) == 0 or len(_pdgs) == 0:
                continue
            hx = np.array(self.TrueHitX[idx])
            hy = np.array(self.TrueHitY[idx])
            hz = np.array(self.TrueHitZ[idx])
            if len(hx) == 0:
                continue
            for k in range(len(_vtx_ids)):
                _vtx_local_idx = int(_vtx_ids[k])
                if _vtx_local_idx < 0 or _vtx_local_idx >= len(self.TrueVtxT[spill]):
                    continue
                _vtx_time = round(float(self.TrueVtxT[spill][_vtx_local_idx]), 3)
                _fullid   = f"{_vtx_time}_{int(_pdgs[k])}"
                if _fullid not in all_uniqueID_dict or _fullid in invalid_groups:
                    continue
                fragment_hits[_fullid] = (hx, hy, hz)

        # Exclude fragments whose hits share z positions (|dz| < 0.2 mm) with another
        # fragment but are separated by more than 10 mm in x-y (>10 such pairs -> invalid)
        scattered_groups = set()
        fullid_list = list(fragment_hits.keys())
        for i, fid_a in enumerate(fullid_list):
            if fid_a in scattered_groups:
                continue
            xa, ya, za = fragment_hits[fid_a]
            for fid_b in fullid_list[i + 1:]:
                if fid_b in scattered_groups:
                    continue
                xb, yb, zb = fragment_hits[fid_b]
                overlap_count = 0
                for k in range(len(za)):
                    z_match = np.where(np.abs(zb - za[k]) < 0.2)[0]
                    if len(z_match) == 0:
                        continue
                    dxy = np.sqrt((xb[z_match] - xa[k]) ** 2 + (yb[z_match] - ya[k]) ** 2)
                    overlap_count += int(np.sum(dxy > 10.0))
                if overlap_count > 10:
                    print(f"  [TMS] Scattered hit overlap: {fid_a} vs {fid_b}: "
                          f"{overlap_count} pairs with |dz|<0.2mm and dxy>10mm -> excluding both")
                    scattered_groups.add(fid_a)
                    scattered_groups.add(fid_b)

        if scattered_groups:
            print(f"  [TMS] {len(scattered_groups)} group(s) removed due to scattered z-overlap: "
                  f"{sorted(scattered_groups)}")
        invalid_groups |= scattered_groups
        """
        written_tms = {}  # fullid -> list of (fragment_id, group_id) actually written

        for idx in tms_indices:
            vtx_ids = self.RecoTrackPrimaryParticleVtxID[idx]
            pdgs    = self.RecoTrackPrimaryParticlePDG[idx]

            if len(vtx_ids) == 0 or len(pdgs) == 0:
                continue

            hits_x = self.TrueHitX[idx]
            hits_y = self.TrueHitY[idx]
            hits_z = self.TrueHitZ[idx]

            if len(hits_x) == 0:
                continue

            for k in range(len(vtx_ids)):
                vtx_local_idx = int(vtx_ids[k])
                if vtx_local_idx < 0 or vtx_local_idx >= len(self.TrueVtxT[spill]):
                    if verbose:
                        print(f"    TMS track idx={idx} k={k}: vtx_ids[{k}]={vtx_ids[k]} "
                              f"out of range for spill {spill} TrueVtxT (len={len(self.TrueVtxT[spill])}), skipping")
                    continue
                vtx_time = round(float(self.TrueVtxT[spill][vtx_local_idx]), 3)
                fullid   = f"{vtx_time}_{int(pdgs[k])}"

                if fullid not in all_uniqueID_dict:
                    if verbose:
                        print(f"    TMS track idx={idx} k={k}: fullid={fullid} not in combined ID dict, skipping")
                    continue

                if fullid in invalid_groups:
                    continue

                fragment_id = fragment_counter
                group_id = all_uniqueID_dict[fullid]
                fragment_counter += 1
                written_tms.setdefault(fullid, []).append((fragment_id, group_id))

                if verbose:
                    print(f"    TMS track idx={idx} k={k}: pdg={pdgs[k]} vtxLocalIdx={vtx_local_idx} "
                          f"vtxTime={vtx_time} "
                          f"-> fullid={fullid} fragment={fragment_id} group={group_id} nhits={len(hits_x)}")

                particle = larcv.Particle(larcv.ShapeType_t.kShapeTrack)
                particle.id(int(fragment_id))
                particle.group_id(int(group_id))

                track_as_voxelset = larcv.VoxelSet()
                track_as_voxelset.id(int(fragment_id))

                for h in range(len(hits_x)):
                    voxelid = vox3dmeta.id(hits_x[h], hits_y[h], hits_z[h])
                    voxel = larcv.Voxel(voxelid)
                    if voxelid != larcv.kINVALID_VOXELID:
                        track_as_voxelset.add(voxel)

                vsa_in.insert(track_as_voxelset)

                entry_clust3d = out_larcv.get_data("cluster3d", "pcluster")
                entry_clust3d.set(vsa_in, vox3dmeta)

                entry_particles = out_larcv.get_data("particle", "corrected")
                entry_particles.append(particle)

        print(f"  [TMS written] {len(written_tms)} group(s) saved:")
        for fid, entries in sorted(written_tms.items()):
            frags = [f"frag={f} grp={g}" for f, g in entries]
            print(f"    fullid={fid}: {', '.join(frags)}")

        return fragment_counter, vsa_in, vox3dmeta, out_larcv, all_uniqueID_dict, invalid_groups


def process_one_NDLAr_entry(spine_data, entry, out_larcv, all_uniqueID_dict, vsa, vox3dmeta,
                            fragment_counter, nd_spill_idx, size_threshold=5, verbose=True):

    truth_particles = spine_data['truth_particles']

    # Map SPINE internal interaction ID -> adjusted vertex time (ns)
    interaction_dict = {
        ixn.id: round(float(ixn.t) - 1.2e9 * nd_spill_idx, 3)
        for ixn in spine_data['truth_interactions']
    }

    # Pre-count fragments per fullid so multi-fragment groups are known before any data is written
    frag_pre_counts = {}
    for p in truth_particles:
        if p.shape != 1:
            continue
        if len(p.points[:, 0]) < size_threshold:
            continue
        fullid = f"{interaction_dict[p.interaction_id]}_{p.pdg_code}"
        if fullid not in all_uniqueID_dict:
            continue
        frag_pre_counts[fullid] = frag_pre_counts.get(fullid, 0) + 1

    invalid_ndlar_groups = {fid for fid, n in frag_pre_counts.items() if n > 1}
    if invalid_ndlar_groups:
        print(f"  [NDLAr] {len(invalid_ndlar_groups)} group(s) with multiple NDLAr fragments excluded: "
              f"{sorted(invalid_ndlar_groups)}")

    for p in truth_particles:
        if p.shape != 1:  # track semantic label
            continue

        vtx_time = interaction_dict[p.interaction_id]
        fullid   = f"{vtx_time}_{p.pdg_code}"

        if len(p.points[:, 0]) < size_threshold:
            continue

        if fullid in invalid_ndlar_groups:
            continue

        fragment_id = fragment_counter
        group_id = all_uniqueID_dict[fullid]
        fragment_counter += 1

        if verbose:
            print(f"    NDLAr particle pdg={p.pdg_code} vtxTime={vtx_time} trackID={p.track_id} "
                  f"-> fullid={fullid} fragment={fragment_id} group={group_id} npoints={len(p.points[:,0])}")

        particle = larcv.Particle(larcv.ShapeType_t.kShapeTrack)
        particle.id(int(fragment_id))
        particle.group_id(int(group_id))

        track_as_voxelset = larcv.VoxelSet()
        track_as_voxelset.id(int(fragment_id))

        # NDLAr points are in cm; convert to mm to match TMS coordinate frame
        for vox in range(len(p.points[:, 0])):
            voxelid = vox3dmeta.id(p.points[vox, 0] * 10, p.points[vox, 1] * 10, p.points[vox, 2] * 10)
            voxel = larcv.Voxel(voxelid)
            if voxelid != larcv.kINVALID_VOXELID:
                track_as_voxelset.add(voxel)

        vsa.insert(track_as_voxelset)

        entry_clust3d = out_larcv.get_data("cluster3d", "pcluster")
        entry_clust3d.set(vsa, vox3dmeta)

        entry_particles = out_larcv.get_data("particle", "corrected")
        entry_particles.append(particle)

    return out_larcv, invalid_ndlar_groups


def setup_larcv(output_filename):
    out_larcv = larcv.IOManager(larcv.IOManager.kWRITE)
    out_larcv.set_out_file(output_filename)
    out_larcv.initialize()
    return out_larcv


def process_entry(out_larcv, run, subrun, entry):
    out_larcv.set_id(run, subrun, entry)
    out_larcv.save_entry()


def write_and_close(out_larcv):
    out_larcv.finalize()


def find_NDLAr_uniqueIDs_remaining(spine_data, nd_spill_idx, uniqueID_dict=None, size_threshold=5):
    """
    Builds unique ID dict for NDLAr particles keyed by adjusted vertex time
    str(round(ixn.t - 1.2e9 * nd_spill_idx, 3)), appending new IDs to an existing
    TMS dict so crossing particles share the same group integer ID.
    Only considers particles with semantic shape == 1 (track).
    interaction_dict maps SPINE internal ixn.id -> adjusted vertex time (float, ns).
    """
    vtx_times = []
    allids = []
    truth_particles = spine_data['truth_particles']
    truth_interactions = spine_data['truth_interactions']

    interaction_dict = {}
    for ixn in truth_interactions:
        interaction_dict[ixn.id] = round(float(ixn.t) - 1.2e9 * nd_spill_idx, 3)

    for p in truth_particles:
        if p.shape != 1:  # track semantic label
            continue
        if len(p.points[:, 0]) < size_threshold:
            continue
        adj_time = interaction_dict[p.interaction_id]
        fullid   = f"{adj_time}_{p.pdg_code}"
        vtx_times.append(fullid)
        allids.append(fullid)

    all_NDLAr_vtx_times = set(vtx_times)
    unique_ids = set(allids)

    if uniqueID_dict:
        new_ids = [str(val) for val in unique_ids if val not in uniqueID_dict]
        starting_index = len(uniqueID_dict)
    else:
        uniqueID_dict = {}
        new_ids = [str(val) for val in unique_ids]
        starting_index = 0

    needed_ints = starting_index + len(new_ids)
    for i in range(starting_index, needed_ints):
        key = new_ids[i - starting_index]
        uniqueID_dict[key] = i

    return uniqueID_dict, all_NDLAr_vtx_times, interaction_dict


def find_NDLAr_vertices(spine_data):
    truth_interactions = spine_data['truth_interactions']
    vertex_ids = [ixn.interaction_id for ixn in truth_interactions]
    return [str(val) for val in set(vertex_ids)]


def find_vertex_ids_all_events(spine_driver, TMSHits, time_window=1e-3, verbose=True):
    """
    Match NDLAr spills to TMS spills by finding shared neutrino vertex times.
    NDLAr interaction times are adjusted for the spill offset (t - 1.2e9 * spill_idx ns).
    TMS times come from TrueVtxT[spill] in the Truth_Spill tree.
    Returns aligned (nd_indices, tms_indices) lists.
    """
    n_spine = len(spine_driver)
    n_tms_spills = len(TMSHits.TrueVtxT)

    nd_vtx_times_per_spill = []
    for nd_spill in range(n_spine):
        spine_data = spine_driver.process(nd_spill)
        nd_times = [round(float(ixn.t) - 1.2e9 * nd_spill, 3) for ixn in spine_data['truth_interactions']]
        nd_vtx_times_per_spill.append(nd_times)

    tms_vtx_times_per_spill = []
    for tms_spill in range(n_tms_spills):
        tms_times = [round(float(t), 3) for t in TMSHits.TrueVtxT[tms_spill]]
        tms_vtx_times_per_spill.append(tms_times)

    nd_matched = []
    tms_matched = []

    if verbose:
        print(f"\n--- Spill matching: {n_spine} NDLAr spills, {n_tms_spills} TMS spills ---")
    for nd_idx, nd_times in enumerate(nd_vtx_times_per_spill):
        for tms_idx, tms_times in enumerate(tms_vtx_times_per_spill):
            shared_times = sorted(
                nd_t for nd_t in nd_times
                if any(abs(nd_t - tms_t) < time_window for tms_t in tms_times)
            )
            if shared_times:
                nd_matched.append(nd_idx)
                tms_matched.append(tms_idx)
                if verbose:
                    print(f"  NDLAr spill {nd_idx} <-> TMS spill {tms_idx}: "
                          f"{len(shared_times)}/{len(nd_times)} NDLAr vertex times matched "
                          f"(first few: {shared_times[:5]}{'...' if len(shared_times) > 5 else ''})")
                break
        else:
            if verbose:
                print(f"  NDLAr spill {nd_idx}: no matching TMS spill found")

    return nd_matched, tms_matched


def make_larcv_box(
    xmin=-4000.0, ymin=-3700.0, zmin=4000.0,
    xmax=4000.0, ymax=1000.0, zmax=19000.0
):
    # Bounding box covering ND-LAr active volume and TMS, all in mm (TMS coordinate frame).
    # ND-LAr (cm -> mm): x[-3478, 3478], y[-2167, 829], z[4179, 9136]
    # TMS:               x[-3730, 3730], y[-3702,  998], z[11178, 18503]
    # Voxel size 1.5 mm, matching ND-LAr voxel pitch.
    xnum = int(math.ceil(abs((xmin - xmax) / 1.5)))
    ynum = int(math.ceil(abs((ymin - ymax) / 1.5)))
    znum = int(math.ceil(abs((zmin - zmax) / 1.5)))

    vox3dmeta = larcv.Voxel3DMeta()
    vox3dmeta.set(xmin, ymin, zmin, xmax, ymax, zmax, xnum, ynum, znum)
    vsa = larcv.VoxelSetArray()

    return vsa, vox3dmeta


if __name__ == "__main__":
    directory  = "/global/homes/j/jessiem/track_matching_reco_GNN/"
    TMS_DIR    = "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-tms-reco/MiniProdN5p1_NDComplex_FHC.tmsreco.full.sanddrift/TMSRECO/0002000/"
    NDLAr_DIR  = "/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-mlreco/MiniProdN5p1_NDComplex_FHC.spine.full.sanddrift/MLRECO_SPINE/0002000/"
    output_dir = directory + "mcfiles/larcv_ndlar/"
    output_base = "out_MiniProdN5p1_NDComplex_FHC"
    SPILLS_PER_FILE      = 1
    START_FILE_IDX       = 0     # set to skip ahead to a specific file index in the sorted list
    MAX_SPILLS           = 1   # set to an int to stop early, e.g. MAX_SPILLS = 50
    VERBOSE = False   # set to False to suppress per-spill / per-track printouts

    os.makedirs(output_dir, exist_ok=True)

    tms_files = sorted(glob.glob(os.path.join(TMS_DIR, "*.TMSRECO.root")))
    if not tms_files:
        print(f"No TMS files found in {TMS_DIR}")
        raise SystemExit(1)
    print(f"Found {len(tms_files)} TMS file(s); starting at index {START_FILE_IDX} ({len(tms_files) - START_FILE_IDX} to process)")

    # global_spill_id counts only matched spills actually written; it drives
    # both the output-file rotation and the per-file entry index.
    global_spill_id      = 0
    output_file_idx      = 0
    my_larcv             = None
    current_out_tmp_path = None   # temp path while larcv is open for writing
    current_first_file   = None   # first filenum that wrote to the current output file
    current_last_file    = None   # last filenum that wrote to the current output file
    n_clusters           = []
    n_matches            = []
    done                 = False  # set True to break out of both file and spill loops

    for tms_file in tms_files[START_FILE_IDX:]:
        if done:
            break

        base_name = os.path.basename(tms_file)
        m = re.search(r'\.(\d{7})\.', base_name)
        if not m:
            print(f"Cannot extract 7-digit file number from {base_name}, skipping")
            continue
        filenum = m.group(1)

        spine_file = os.path.join(
            NDLAr_DIR,
            f"MiniProdN5p1_NDComplex_FHC.spine.full.sanddrift.{filenum}.MLRECO_SPINE.hdf5"
        )
        if not os.path.isfile(spine_file):
            print(f"No matching SPINE file for {base_name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing file number {filenum}")
        if VERBOSE:
            print(f"  TMS:   {base_name}")
            print(f"  SPINE: {os.path.basename(spine_file)}")

        cfg = f'''
        base:
          verbosity: warning
        build:
          mode: both
          fragments: false
          particles: true
          interactions: true
          units: cm
        io:
          reader:
            file_keys: {spine_file}
            skip_unknown_attrs: True
            name: hdf5
        '''
        driver  = Driver(yaml.safe_load(cfg))
        TMSHits = TMSData(tms_file)

        # find_vertex_ids_all_events returns only matched pairs — unmatched spills
        # are printed as warnings but never appear in the returned lists.
        nd_spill_indices, tms_spill_indices = find_vertex_ids_all_events(
            driver, TMSHits, verbose=VERBOSE
        )
        print(f"  {len(nd_spill_indices)} matched spill pair(s) found in file {filenum}")

        for spill_id in range(1): #len(nd_spill_indices)):
            # Stop early if the caller-specified limit has been reached
            if MAX_SPILLS is not None and global_spill_id >= MAX_SPILLS:
                print(f"\nReached MAX_SPILLS={MAX_SPILLS}, stopping.")
                done = True
                break

            nd_index  = nd_spill_indices[spill_id]
            tms_index = tms_spill_indices[spill_id]

            # Open a new output file at the very start and every SPILLS_PER_FILE matched spills
            if global_spill_id % SPILLS_PER_FILE == 0:
                if my_larcv is not None:
                    my_larcv.finalize()
                    file_range = (f"{current_first_file}-{current_last_file}"
                                  if current_last_file != current_first_file
                                  else current_first_file)
                    final_path = os.path.join(
                        output_dir,
                        f"{output_base}.{file_range}.larcv.root"
                    )
                    os.rename(current_out_tmp_path, final_path)
                    print(f"\nClosed output file: {final_path}")
                current_first_file   = None
                current_last_file    = None
                current_out_tmp_path = os.path.join(
                    output_dir, f"{output_base}.tmp.larcv.root"
                )
                my_larcv = setup_larcv(current_out_tmp_path)
                print(f"\nOpened output file {output_file_idx} (temp: {current_out_tmp_path})")
                output_file_idx += 1

            # Track first and last source file number for the current output file
            if current_first_file is None:
                current_first_file = filenum
            current_last_file = filenum

            # Build unique ID dict (keyed by vertex time + PDG) from TMS Truth_Spill,
            # then extend with NDLAr particles from the matched spill.
            tms_vtx_times, tms_unique_IDs = TMSHits.find_TMS_uniqueID(tms_index)
            spine_data = driver.process(nd_index)
            all_uniqueID_dict, all_NDLAr_vtx_times, interaction_dict = find_NDLAr_uniqueIDs_remaining(
                spine_data, nd_index, tms_unique_IDs
            )

            # crossing_particle_ids: vtxtime+pdg fullids present in both TMS and NDLAr
            tms_time_set   = set(tms_vtx_times)
            ndlar_time_set = all_NDLAr_vtx_times
            crossing_particle_ids   = tms_time_set & ndlar_time_set
            ndlar_only_particle_ids = ndlar_time_set - tms_time_set
            tms_only_particle_ids   = tms_time_set  - ndlar_time_set

            if VERBOSE:
                print(f"\n=== Global spill {global_spill_id} "
                      f"(file {filenum}, NDLAr[{nd_index}] <-> TMS[{tms_index}]) ===")
                print(f"  TMS fullids   ({len(tms_time_set)}): {sorted(tms_time_set)}")
                print(f"  NDLAr fullids ({len(ndlar_time_set)}): {sorted(ndlar_time_set)}")
                print(f"  Shared fullids ({len(crossing_particle_ids)} crossing particles): "
                      f"{sorted(crossing_particle_ids)}")
                print(f"  TMS-only fullids ({len(tms_only_particle_ids)}): "
                      f"{sorted(tms_only_particle_ids)}")
                print(f"  NDLAr-only fullids ({len(ndlar_only_particle_ids)}): "
                      f"{sorted(ndlar_only_particle_ids)}")
                print(f"  Total group IDs in dict: {len(all_uniqueID_dict)}")
                if crossing_particle_ids:
                    print(f"  Crossing fullids: {sorted(crossing_particle_ids)}")

            n_clusters.append(len(all_uniqueID_dict))

            fragment_counter = 0
            vsa, vox3meta = make_larcv_box()

            fragment_counter, vsa, vox3meta, my_larcv, all_ID_dict, invalid_tms_groups = TMSHits.process_one_TMS_entry(
                tms_index, my_larcv, all_uniqueID_dict, vsa, vox3meta, fragment_counter, verbose=VERBOSE
            )
            crossing_particle_ids -= invalid_tms_groups
            my_larcv, invalid_ndlar_groups = process_one_NDLAr_entry(
                spine_data, nd_index, my_larcv, all_uniqueID_dict, vsa, vox3meta,
                fragment_counter, nd_spill_idx=nd_index, verbose=VERBOSE
            )
            crossing_particle_ids -= invalid_ndlar_groups

            n_matches.append(len(crossing_particle_ids))

            invalid_all = invalid_tms_groups | invalid_ndlar_groups
            print(f"  Spill summary (global {global_spill_id}): "
                  f"{len(all_uniqueID_dict)} total unique interactions in spill, "
                  f"{len(crossing_particle_ids)} expected TMS<->NDLAr crossing interactions, "
                  f"{len(invalid_all)} group(s) removed ({len(invalid_tms_groups)} TMS, {len(invalid_ndlar_groups)} NDLAr)")

            # Entry index resets to 0 at the start of each output file
            entry_in_file = global_spill_id % SPILLS_PER_FILE
            my_larcv.set_id(0, 0, entry_in_file)
            my_larcv.save_entry()
            global_spill_id += 1

    if my_larcv is not None:
        my_larcv.finalize()
        file_range = (f"{current_first_file}-{current_last_file}"
                      if current_last_file != current_first_file
                      else current_first_file)
        final_path = os.path.join(
            output_dir,
            f"{output_base}.{file_range}.larcv.root"
        )
        os.rename(current_out_tmp_path, final_path)
        print(f"\nClosed final output file: {final_path}")

    print(f"\nDone. {global_spill_id} matched spill(s) written across {output_file_idx} output file(s)")
    print("N clusters per spill:", n_clusters)
    print("N matched particles per spill:", n_matches)
