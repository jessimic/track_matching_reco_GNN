"""Match SPINE events (NDLAr) to Truth_Spill entries (TMS ROOT file).

Method 1 (IDs): SPINE's interaction orig_id is NOT the Truth_Spill vertex ID.
  It is <run prefix> + a running counter over SPINE's own interactions, so only
  the prefix agrees (e.g. orig_id 50000000 <-> TrueVtxGlobalID 50000006).
  Instead we match each SPINE interaction to a Truth_Spill vertex by the
  (track ID, PDG) signature of its particles, and use the fact that SPINE
  interactions are in the same order as the Truth_Spill vertices (ascending ID)
  to resolve ambiguities (order-preserving alignment).
Method 2 (position): match vertex x, y, z.
Method 3 (particles/tracks): within each position-matched vertex, match
  Truth_Spill particles to SPINE particles by track ID (confirmed by PDG and
  birth position), and link TMS reco tracks to them via Truth_Info.

Run inside the spine-nb docker container (needs spine + uproot).
"""
import yaml
import numpy as np
import uproot
from spine.driver import Driver

data_dir="/data/"
spine_file = data_dir+"MiniProdN5p3_NDComplex_FHC.spine.full.sanddrift.0000005.MLRECO_SPINE.hdf5"
tms_file = data_dir+"MiniProdN5p3_NDComplex_FHC.tmsreco.full.sanddrift.v4.0000005.TMSRECO.root"
verbose_print=False

THROUGHGOING_FLAG = 10**15  # orig_id flag on external (throughgoing) interactions; not in Truth_Spill

tree = uproot.open(tms_file)["Truth_Spill"]
tms = tree.arrays(["SpillNo", "TrueVtxGlobalID", "VertexGlobalID", "TrackId", "PDG"], library="np")

cfg = f'''
base:
  verbosity: warning
build:
  mode: truth
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
driver = Driver(yaml.safe_load(cfg))
spine_events = [driver.process(i) for i in range(len(driver))]


# ---------------------------------------------------------------------------
# Method 1: (track ID, PDG) signature + order-preserving alignment
# ---------------------------------------------------------------------------
def tms_signatures(entry):
    """Per Truth_Spill vertex (sorted by global ID): (global ID, set of (track, pdg))."""
    gids = np.sort(tms["TrueVtxGlobalID"][entry])
    sig = {int(g): set() for g in gids}
    for g, trk, pdg in zip(tms["VertexGlobalID"][entry], tms["TrackId"][entry], tms["PDG"][entry]):
        g = int(g) % THROUGHGOING_FLAG
        if g in sig:
            sig[g].add((int(trk), int(pdg)))
    return [(int(g), sig[int(g)]) for g in gids]


def spine_signatures(data):
    """SPINE interactions (non-throughgoing, in orig_id order): (ixn, set of (track, pdg))."""
    ixns = sorted((x for x in data["truth_interactions"] if int(x.orig_id) < THROUGHGOING_FLAG),
                  key=lambda x: int(x.orig_id))
    sig = {x.id: set() for x in ixns}
    for p in data["truth_particles"]:
        if p.interaction_id in sig:
            sig[p.interaction_id].add((int(p.track_id), int(p.pdg_code)))
    return [(x, sig[x.id]) for x in ixns]


def align(spine_sig, tms_sig):
    """Order-preserving assignment of SPINE interactions to Truth_Spill vertices.

    Score of a pair = fraction of the Truth_Spill vertex's (track, pdg) pairs found
    in the SPINE interaction, minus half the fraction that are missing.
    Returns (total score, {spine position: tms position}).
    """
    n, m = len(spine_sig), len(tms_sig)
    score = np.zeros((n, m))
    for i, (_, s) in enumerate(spine_sig):
        for j, (_, t) in enumerate(tms_sig):
            score[i, j] = (len(t & s) - 0.5 * len(t - s)) / max(len(t), 1)
    best = np.full((n + 1, m + 1), -1e9)
    best[0, :] = 0
    took = np.zeros((n + 1, m + 1), bool)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            take = best[i - 1, j - 1] + score[i - 1, j - 1]
            best[i, j] = max(best[i, j - 1], take)
            took[i, j] = take >= best[i, j - 1]
    i, j, assign = n, m, {}
    while i > 0 and j > 0:
        if took[i, j]:
            assign[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return best[n, m], assign


def match_by_ids():
    """{SPINE event: (Truth_Spill entry, {SPINE ixn id: Truth_Spill global vertex ID})}"""
    tms_sigs = [tms_signatures(e) for e in range(len(tms["SpillNo"]))]
    out = {}
    for ev, data in enumerate(spine_events):
        s_sig = spine_signatures(data)
        results = [align(s_sig, t_sig) for t_sig in tms_sigs]
        entry = int(np.argmax([r[0] for r in results]))
        assign = results[entry][1]
        out[ev] = (entry, {s_sig[i][0].id: tms_sigs[entry][j][0] for i, j in assign.items()})
        print(f"SPINE event {ev} -> Truth_Spill {entry} "
              f"(SpillNo {tms['SpillNo'][entry]}): {len(assign)}/{len(s_sig)} interactions matched")
        for i, j in sorted(assign.items())[:5]:
            x = s_sig[i][0]
            print(f"    ixn {x.id} (orig_id {x.orig_id}) -> vertex global ID {tms_sigs[entry][j][0]}")
    return out


print("=== ID-based matching ===")
id_matches = match_by_ids()


# ---------------------------------------------------------------------------
# Method 2: match on vertex position (Truth_Spill TrueVtxX/Y/Z vs SPINE ixn.vertex)
# ---------------------------------------------------------------------------
VERTEX_TOL_MM = 1.0

tms_vtx = tree.arrays(["TrueVtxX", "TrueVtxY", "TrueVtxZ"], library="np")
tms_vtx_xyz = [
    np.stack([x, y, z], axis=1)  # (n_vertices, 3) in mm
    for x, y, z in zip(tms_vtx["TrueVtxX"], tms_vtx["TrueVtxY"], tms_vtx["TrueVtxZ"])
]

# All SPINE true interactions: (event, interaction id, vertex in mm)
spine_vtx = []
for spine_idx, data in enumerate(spine_events):
    for ixn in data["truth_interactions"]:
        spine_vtx.append((spine_idx, int(ixn.id), np.asarray(ixn.vertex) * 10.0))  # cm -> mm
spine_vtx_xyz = np.array([v for _, _, v in spine_vtx])


def match_by_position(tol_mm=VERTEX_TOL_MM, verbose=False):
    """Nearest SPINE interaction vertex for every Truth_Spill vertex.

    Returns {(Truth_Spill entry, vertex index): (SPINE event, interaction id, distance mm)}
    for pairs within tol_mm. verbose=True also prints the nearest (unmatched)
    candidate for every vertex.
    """
    out = {}
    for tms_idx, xyz in enumerate(tms_vtx_xyz):
        for k, pos in enumerate(xyz):
            dist = np.linalg.norm(spine_vtx_xyz - pos, axis=1)
            j = int(dist.argmin())
            ev, ixn_id, spine_pos = spine_vtx[j]
            ok = dist[j] <= tol_mm
            if ok and verbose:
                print(f"Truth_Spill {tms_idx} vtx {k} {np.round(pos, 2)} mm -> "
                      f"SPINE event {ev} ixn {ixn_id} {np.round(spine_pos, 2)} mm  "
                      f"dist {dist[j]:.2f} mm  {'MATCH' if ok else 'no match'}")
            if ok:
                out[(tms_idx, k)] = (ev, ixn_id, float(dist[j]))
    return out


print("\n=== Position-based matching ===")
position_matches = match_by_position()

print(f"{len(position_matches)} / {sum(len(x) for x in tms_vtx_xyz)} vertices matched "
      f"within {VERTEX_TOL_MM} mm")

# Per Truth_Spill entry: which SPINE event(s) its matched vertices landed in
print("\nSummary per Truth_Spill entry:")
for tms_idx, xyz in enumerate(tms_vtx_xyz):
    events = [position_matches[(tms_idx, k)][0] for k in range(len(xyz)) if (tms_idx, k) in position_matches]
    if events:
        ev, n = np.unique(events, return_counts=True)
        print(f"  Truth_Spill {tms_idx}: {len(events)}/{len(xyz)} vertices matched; "
              f"SPINE events {dict(zip(ev.tolist(), n.tolist()))}")
    else:
        print(f"  Truth_Spill {tms_idx}: 0/{len(xyz)} vertices matched")


# Cross-check: do the ID matches agree with the position matches?
tms_gid_pos = {}  # (entry, global vertex ID) -> vertex index
for entry in range(len(tms["SpillNo"])):
    for k, g in enumerate(tms["TrueVtxGlobalID"][entry]):
        tms_gid_pos[(entry, int(g))] = k
agree = total = 0
for ev, (entry, assign) in id_matches.items():
    for ixn_id, gid in assign.items():
        pm = [(key, v) for key, v in position_matches.items() if v[:2] == (ev, ixn_id)]
        if not pm:
            continue
        total += 1
        agree += any(key == (entry, tms_gid_pos[(entry, gid)]) for key, _ in pm)
print(f"\nID matches that agree with position matches: {agree}/{total}")


# ---------------------------------------------------------------------------
# Method 3: particle and track matching within each position-matched vertex
# ---------------------------------------------------------------------------
# Within a correctly matched vertex, Truth_Spill TrackId == SPINE track_id
# identifies the same Geant4 particle; PDG and birth position (< PARTICLE_TOL_MM)
# are checked as confirmation. Truth_Spill particles with no SPINE counterpart
# are mostly neutrals (n, pi0, gamma, nu) that leave no charge in ND-LAr.
# TMS reco tracks (Truth_Info) point at their true particle through
# RecoTrackPrimaryParticleIndex, an index into that spill's Truth_Spill
# particle arrays, which links them to the SPINE particle.
PARTICLE_TOL_MM = 1.0

tms_part = tree.arrays(["BirthPosition", "DeathPosition", "IsPrimary"], library="np")
tms_reco = uproot.open(tms_file)["Truth_Info"].arrays(
    ["SpillNo", "RecoTrackN", "RecoTrackPrimaryParticleIndex"], library="np")
spill_to_entry = {int(s): e for e, s in enumerate(tms["SpillNo"])}

# (entry, Truth_Spill particle index) -> list of (Truth_Info entry, reco track index)
reco_tracks_by_particle = {}
for i, (spill_no, n_trk) in enumerate(zip(tms_reco["SpillNo"], tms_reco["RecoTrackN"])):
    entry = spill_to_entry[int(spill_no)]
    for k in range(n_trk):
        p_idx = int(tms_reco["RecoTrackPrimaryParticleIndex"][i][k])
        reco_tracks_by_particle.setdefault((entry, p_idx), []).append((i, k))


def match_particles(entry, gid, ev, ixn_id, tol_mm=PARTICLE_TOL_MM):
    """Match Truth_Spill particles of vertex gid to SPINE particles of interaction ixn_id.

    Returns a list of dicts, one per Truth_Spill particle, with status:
      'matched'  - same track ID, PDG and birth position within tol_mm
      'rejected' - same track ID but PDG or birth position disagree
      'tms_only' - no SPINE particle with this track ID (e.g. neutrals)
    """
    spine_parts = {int(p.track_id): p for p in spine_events[ev]["truth_particles"]
                   if p.interaction_id == ixn_id}
    out = []
    for idx in np.where(tms["VertexGlobalID"][entry] == gid)[0]:
        trk, pdg = int(tms["TrackId"][entry][idx]), int(tms["PDG"][entry][idx])
        birth = np.asarray(tms_part["BirthPosition"][entry][idx][:3], dtype=float)
        rec = {"tms_particle_idx": int(idx), "track_id": trk, "tms_pdg": pdg,
               "is_primary": bool(tms_part["IsPrimary"][entry][idx]),
               "reco_tracks": reco_tracks_by_particle.get((entry, int(idx)), []),
               "spine_particle_id": None, "spine_pdg": None, "birth_dist_mm": None}
        p = spine_parts.get(trk)
        if p is None:
            rec["status"] = "tms_only"
        else:
            dist = float(np.linalg.norm(np.asarray(p.position) * 10.0 - birth))  # cm -> mm
            rec.update(spine_particle_id=int(p.id), spine_pdg=int(p.pdg_code), birth_dist_mm=dist)
            rec["status"] = "matched" if (int(p.pdg_code) == pdg and dist <= tol_mm) else "rejected"
        out.append(rec)
    return out


print("\n=== Particle / track matching within matched vertices ===")
particle_matches = {}  # (SPINE event, ixn id) -> list of particle records
for (entry, k), (ev, ixn_id, _) in position_matches.items():
    gid = int(tms["TrueVtxGlobalID"][entry][k])
    particle_matches[(ev, ixn_id)] = match_particles(entry, gid, ev, ixn_id)

records = [r for recs in particle_matches.values() for r in recs]
n_status = {s: sum(r["status"] == s for r in records) for s in ("matched", "rejected", "tms_only")}
print(f"Truth_Spill particles in matched vertices: {len(records)}  {n_status}")
missing_pdgs, counts = np.unique([r["tms_pdg"] for r in records if r["status"] == "tms_only"],
                                 return_counts=True)
print("  tms_only PDGs:", dict(sorted(zip(missing_pdgs.tolist(), counts.tolist()), key=lambda x: -x[1])))

n_reco = sum(len(r["reco_tracks"]) for r in records)
n_reco_matched = sum(len(r["reco_tracks"]) for r in records if r["status"] == "matched")
print(f"TMS reco tracks from matched vertices: {n_reco}, "
      f"linked to a SPINE particle: {n_reco_matched}")

if verbose_print:
    for (ev, ixn_id), recs in sorted(particle_matches.items()):
        linked = [r for r in recs if r["status"] == "matched" and r["reco_tracks"]]
        if not linked:
            continue
        print(f"SPINE event {ev} ixn {ixn_id}:")
        for r in linked:
            print(f"    track {r['track_id']} PDG {r['tms_pdg']} -> SPINE particle {r['spine_particle_id']} "
                  f"(birth dist {r['birth_dist_mm']:.3f} mm), TMS reco tracks {r['reco_tracks']}")
