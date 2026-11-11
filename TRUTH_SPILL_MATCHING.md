# Truth-Spill Event Matching: SPINE (ND-LAr) <-> TMS

Documentation for work done in the "Truth Spill event matching" conversation, covering
[match_spine_to_truth_spill.py](match_spine_to_truth_spill.py),
[SPINE_TMS_vertex_matching_viewer.ipynb](SPINE_TMS_vertex_matching_viewer.ipynb) and
[create_LArCV_NDLAr_TMS_matched.py](create_LArCV_NDLAr_TMS_matched.py).

## Goal

MiniProd files are produced separately for the two detectors:
- **SPINE** (`*.MLRECO_SPINE.hdf5`) holds ND-LAr truth/reco interactions, built from the edep-sim
  output via the `spine` driver.
- **TMS** (`*.TMSRECO.root`, `Truth_Spill` / `Truth_Info` trees) holds the Muon Spectrometer
  (TMS) truth vertices, particles, and reconstructed tracks.

Neither file cross-references the other directly, and the ID fields that look like they should
line up don't: SPINE's interaction `orig_id` is **not** the Truth_Spill vertex `TrueVtxGlobalID`
or `VertexGlobalID` (it's a run prefix + a running counter over SPINE's own interactions). The
scripts in this doc establish, and cross-check, a correspondence between:

1. a SPINE true interaction (ND-LAr) and a Truth_Spill vertex (same physical neutrino
   interaction, seen by both detectors),
2. the individual true particles at that vertex in each detector, and
3. the TMS reconstructed tracks that point back to those particles.

Both files were written for and tested against:
- `MiniProdN5p3_NDComplex_FHC.spine.full.sanddrift.0000005.MLRECO_SPINE.hdf5`
- `MiniProdN5p3_NDComplex_FHC.tmsreco.full.sanddrift.v4.0000005.TMSRECO.root`

Both must be run inside the `spine-nb` docker container (needs `spine` + `uproot`).

## match_spine_to_truth_spill.py

Standalone script, run start to finish, that performs the matching and prints a validation
summary to stdout. Three methods, each building on the last:

### Method 1 — ID-based matching (`match_by_ids`)

Since `orig_id` doesn't map onto Truth_Spill IDs, each SPINE interaction is instead matched by
the **signature** of its particles: the set of `(track_id, pdg)` pairs in the interaction.
External (throughgoing) vertices carry a `THROUGHGOING_FLAG = 10**15` offset in *all* the ID
fields: Truth_Spill `TrueVtxGlobalID`, the particles' `VertexGlobalID` (e.g. `1000000052000000`),
and SPINE `orig_id`. In file `0000005` they are 92–96% of each Truth_Spill entry's vertices
(e.g. 427/455 in entry 0) and 504 of the 700 position-matched SPINE interactions. Normal and
throughgoing interactions form **two separate ID sequences**, so Method 1 handles them as two
independent alignments:
- `tms_signatures(entry, throughgoing)`: per Truth_Spill entry, builds one signature set per
  vertex of the requested kind (keyed by `TrueVtxGlobalID`), from the `VertexGlobalID` /
  `TrackId` / `PDG` branches. Particles are tied to vertices by exact ID; both sides carry the
  flag, so nothing is stripped.
- `spine_signatures(data, throughgoing)`: per SPINE event, builds one signature set per true
  interaction of the requested kind (`orig_id >= THROUGHGOING_FLAG` or not), sorted by `orig_id`.
  SPINE's `orig_id` (and `interaction_id`, which is identical: 700/700) is **not** a Truth_Spill
  vertex ID for either kind (0/196 normal and 0/504 throughgoing equal their position-matched
  `TrueVtxGlobalID`). For normal interactions it is the run prefix plus a running counter over
  SPINE's own kept interactions, e.g. `50000000, 50000001, ...` for vertices
  `50000006, 50000023, ...`. Its values can collide with the *wrong* vertex — e.g. interaction 6
  has `orig_id 50000006`, which is the vertex ID of interaction 0. Throughgoing `orig_id`s are a
  different numbering again (e.g. `…52000013, …52000014, …52000033` for vertices
  `…52000002, …52000003, …52000007`). Only the **ordering** is useful.
- `align(spine_sig, tms_sig)`: within one spill and one kind, SPINE interactions sorted by
  `orig_id` follow the same ascending order as their Truth_Spill vertex IDs (verified against
  the position matches; 0 violations in 497 consecutive throughgoing pairs), so alignment is
  solved as a 1-D order-preserving assignment (a small DP, like a simplified sequence
  alignment/LCS), rather than a general bipartite matching. The pair score is the fraction of the
  Truth_Spill vertex's `(track, pdg)` pairs found in the SPINE interaction, minus half the
  fraction missing.
- `match_by_ids()`: for every SPINE event, aligns both kinds against every Truth_Spill entry and
  keeps the entry with the best combined score, i.e. it also discovers *which* Truth_Spill entry
  (spill) a SPINE event corresponds to, not just the vertex-level alignment within it.

**Throughgoing ID matches are unreliable** (16% correct, see Testing); use Method 2 for those.

### Method 2 — position-based matching (`match_by_position`)

A fully independent, geometry-only match, and the **reference** method (Method 3 is built on
its matches): each Truth_Spill vertex (`TrueVtxX/Y/Z`, mm) is
matched to its nearest SPINE true-interaction vertex (`ixn.vertex`, converted cm → mm), and kept
only if the distance is within `VERTEX_TOL_MM` (1.0 mm). This method makes no use of particle
content or ordering, so agreement with Method 1 confirms the signature/alignment logic is sound.

The script then cross-checks the two methods directly: for every ID-based match, it checks
whether the same (event, interaction) pair also has a position match to the same Truth_Spill
vertex.

### Method 3 — particle/track matching (`match_particles`)

Added after Methods 1 & 2 were validated. Within each *position-matched* vertex (from Method 2),
individual particles are linked across detectors:
- A Truth_Spill particle and a SPINE particle are the same Geant4 particle if they share
  `TrackId`/`track_id`. PDG code and birth position (within `PARTICLE_TOL_MM` = 1.0 mm, cm → mm)
  are checked as confirmation, giving each Truth_Spill particle a status:
  - `matched` — same track ID, PDG, and birth position agree,
  - `rejected` — same track ID but PDG or birth position disagree,
  - `tms_only` — no SPINE particle has that track ID at all (expected for particles that leave no
    charge in ND-LAr: neutrons, pi0, gammas, neutrinos).
- TMS reconstructed tracks (from the `Truth_Info` tree) are linked to their truth particle via
  `RecoTrackPrimaryParticleIndex`, an index into that spill's Truth_Spill particle arrays — this
  is what finally connects a TMS *reco* track to a SPINE truth particle.

Particles are tied to vertices by exact `VertexGlobalID == TrueVtxGlobalID`, with no
`THROUGHGOING_FLAG` stripping: `TrueVtxGlobalID` already contains the flagged (`>= 10**15`)
external vertices, and every particle's `VertexGlobalID` appears in that list as written.

Output is `particle_matches`, a dict keyed by `(SPINE event, ixn id)` holding one record per
Truth_Spill particle at that vertex: `status`, `track_id`, `tms_pdg`, `spine_pdg`,
`spine_particle_id`, `birth_dist_mm`, `is_primary`, and `reco_tracks` (list of
`(Truth_Info entry, reco track index)`). Set `verbose_print = True` to print every
reco-track → SPINE-particle link.

`match_spine_to_truth_spill.py` is otherwise unchanged in logic from commit `1496eec`.

## SPINE_TMS_vertex_matching_viewer.ipynb

Interactive companion notebook: reruns Methods 1, 2 and 3 from the script with the same logic
and the same numbering (Method 2, position, is the reference), then builds a vertex table, a
particle table, and a 3D event display that plots either whole vertices or individual
particle/track matches. Its printed results match the script's on file `0000005`.

- **Load files** — loads `Truth_Spill`; `Truth_Info` for the TMS true hits (`TrueHitX/Y/Z`,
  tagged by `TrueHitVertexId` and `TrueHitPrimaryId`) and the reco-track truth link
  (`RecoTrackPrimaryParticleIndex`); and `Reco_Tree` for the reco-track hit positions
  (`TrackHitPos[track][hit]`, first `nHits[track]` entries). `Reco_Tree` lines up entry by entry
  with `Truth_Info` (same spill, event and track count). Runs the `spine` driver with
  `build.mode: both` (truth *and* reco).
- **TMS hits per particle** — `TrueHitPrimaryId` is a Truth_Spill `TrackId`: for 99.3% of TMS
  true hits (113,506/114,290) it equals the track ID of a Truth_Spill particle at the hit's
  vertex. Selecting hits by vertex ID + track ID therefore gives a particle's TMS hits, including
  hits from its descendants.
- **Method 1 / Method 2 / cross-check** — same logic as the script (both Method 1 streams), with
  the cross-check printed separately for normal and throughgoing interactions.
- **Method 3** — same as the script; results in `particle_matches`.
- **Vertex table** (`match_df`) — one row per position-matched vertex: `throughgoing`,
  `id_method_agrees`, `n_particles_matched`, `n_reco_tracks_linked`, and hit counts in each
  detector (`n_NDLAr_points`, `n_TMS_hits`). `both` keeps the 391 vertices with hits in both
  detectors.
- **Particle table** (`particle_df`) — one row per *matched* particle (1,065): track ID, PDG,
  SPINE particle id, birth distance, `n_NDLAr_points` (the SPINE particle's own points),
  `n_TMS_hits` (TMS true hits with that track ID, so descendants included), and the linked
  `reco_tracks`. `particles_both` keeps the 373 with hits in both detectors; `particles_reco`
  keeps the 228 with a TMS reco track.
- **3D event display** — plotly `Scatter3d` with the TMS and ND-LAr volumes drawn as wireframe
  boxes (geometry from `TMS_Constants.h`). Two entry points, described below:
  `draw_vertices(rows)` and `draw_particles(rows, ...)`.

### Setup

The first cells set `SOFTWARE_DIR` (added to `sys.path` so `spine` can be imported from a local
install) and `data_dir` (where the two input files live). Both need editing for each machine; in
the `spine-nb` container `spine` is already importable and the files are under `/data`.

### `draw_vertices(rows)`

Takes `match_df` rows: all ND-LAr points of the SPINE interaction, all TMS true hits of the
vertex, and both vertex markers.

### `draw_particles(rows, ...)`

Takes `particle_df` rows. For each chosen particle it draws the SPINE particle's ND-LAr points
(strong colour), the TMS true hits with its track ID (light shade of the same colour), and each
linked TMS reco track (hits joined by a line). The notebook's example selects every particle in
`particles_reco` from one spill (spill 6: 28 particles, 24 from throughgoing vertices).

```
draw_particles(rows, show_vertex=True, show_legend=True, show_other_particles=False,
               show_table=True, given_title=None, show_all_activity=False, toggles=True)
```

| Option | Effect |
|---|---|
| `show_vertex` | Vertex markers: SPINE vertex (red open square) and Truth_Spill vertex (black cross), once per vertex. |
| `show_other_particles` | Everything else at the chosen particles' vertices: other *matched* particles in colour, drawn like the chosen ones; *unmatched* activity in gray, one trace per vertex for SPINE particles with no Truth_Spill match (dark gray `#7f7f7f`, ND-LAr) and one for TMS true hits not from a matched particle (light gray `#c7c7c7`, TMS). |
| `show_all_activity` | Everything else in the same spill(s), in faint gray and smaller markers: every other SPINE interaction's ND-LAr points (`#b5b5b5`), every other Truth_Spill vertex's TMS true hits (`#dedede`), and every TMS reco track not already drawn (`#9e9e9e` lines). One trace per kind per spill. Implies `show_other_particles`. |
| `toggles` | `True` (default): all three optional layers above are always built, and an independent **on/off button** for each is placed above the plot (plotly `updatemenus`; the first click flips the layer, the second flips it back). The three `show_*` options then only set what is visible at first. `False`: only the requested layers are built, with no buttons. |
| `show_legend` | `False` removes the plot legend entirely, so the table is the reference. |
| `show_table` | Displays a table under the plot with one row per legend entry: a colour swatch, colour, marker type, item (spill, interaction, track, PDG), what is shown, a count, the **layer** it belongs to, and whether it was **visible at start**. With toggles on it lists every layer, including hidden ones. |
| `given_title` | Plot title (default: the list of `particle_df` rows). |

**Colours.** The first 7 particles use the notebook's 7-colour palette; later ones use plotly's
Dark24 (strong) / Light24 (light) sets, so colours repeat after 31 objects — the table's item
column tells repeats apart. Grays are reserved for unmatched and background activity.

**Things to know:**
- With `toggles=True` the faint activity layer (about 100,000 ND-LAr points per spill) is
  always built, even when hidden, so rendering is slower; use `toggles=False` if that matters.
- The activity layer leaves out what the "other particles" layer draws. Turning "other
  particles" off while "all other activity" is on hides those particles in both.
- ND-LAr charge that isn't assigned to any SPINE true interaction is not drawn by any layer.
- `draw_particles` was tested in the container by building the figures (trace counts, initial
  visibility, and which traces each button controls); the buttons have not been clicked in a
  rendered plot as part of that test.

The container has no `ipywidgets`, so rows are chosen by editing the list in the cell that calls
each plotting function, and layer toggling uses plotly's own buttons rather than widgets.

## create_LArCV_NDLAr_TMS_matched.py

Writes ND-LAr + TMS LArCV files for GrapPA training, using the Method 2 + Method 3 matches as the
cross-detector truth. Modelled on `create_LArCV_minerva_2x2.py` (the per-particle fragment
pattern in its lines 269–300, and its file loop and train/validation split), with the 2x2 +
MINERvA inputs replaced by SPINE + TMS. It is separate from the older
`create_LArCV_TMS_NDLAr.py`, which matches by vertex time and PDG.

**One LArCV entry per spill.** SPINE events are paired with Truth_Spill entries by majority vote
of the Method 2 vertex matches (`pair_spills`), not by assuming equal indices.

**Fragments** (each one LArCV `Particle` + `VoxelSet`, written to `cluster3d_pcluster` and
`particle_corrected` as in the reference):
- **ND-LAr:** one fragment per SPINE true particle, using its true points (cm → mm), with the
  reference's 5-point size threshold. Shape is SPINE's true semantic shape (track, shower,
  delta, Michel, ...); the reference marks everything as a track — to match it, change
  `larcv.ShapeType_t(int(p.shape))` to `kShapeTrack`.
- **TMS:** one fragment per TMS reco track, using its `Reco_Tree` hit positions
  (`TrackHitPos[track][:nHits]`, mm), shape track.
- `fragment_id` (`particle.id`, also the voxel set id) is a counter that restarts at 0 in each
  spill. `pdg_code` and `track_id` are also set on each particle, for debugging (the reference
  doesn't set them).

**`group_id` is one per true particle.**
- A TMS reco track gets the group of its true Truth_Spill particle (via
  `RecoTrackPrimaryParticleIndex`), so two reco tracks from the same particle share a group.
- An ND-LAr particle that Method 3 matched to that same Truth_Spill particle gets the same
  group. These shared groups are the cross-detector matches GrapPA should learn.
- Everything else gets its own group.

**Box (`make_larcv_box`).** x −4000 to 4000, y −3705 to 1000, z 4000 to 19000 mm, 3 mm voxels
(2667 × 1569 × 5000), covering both active volumes (limits in the code comments). This is the
same box `Plotting_validation_TMS.ipynb` uses to convert voxels back to mm. The older
`create_LArCV_TMS_NDLAr.py` uses 1.5 mm voxels and y from −3700, which cuts 2.2 mm off the bottom
of TMS; the voxel size is a single argument (`voxel_size`).

**Running it.** The `__main__` loop globs `directory` (default `./`) for `*.TMSRECO.root`,
pairs each with the SPINE file of the same 7-digit number, puts every 5th file pair in
`larcv/larcv_validation_set/` and the rest in `larcv/`, and skips existing outputs unless
`rewrite_output = True`. Unlike the reference (where they are commented out), it calls
`save_entry()` for every spill and `finalize()` at the end of each file. Needs `spine` and
`larcv` (both available in the `spine-nb` container).

## Testing

All results below are from file index `0000005` only (7 spills, `SpillNo` 0–6), run inside the
`spine-nb` container by directly invoking `match_spine_to_truth_spill.py`.

An earlier attempt on file `0000001` used an
`orig_id − 9,000,000 == VertexGlobalID` offset rule; that rule was a coincidence of small IDs,
not a real mapping, and was abandoned.

### Method 1 — ID-based matching

All 7 SPINE events were assigned to the Truth_Spill entry with the same index (event N → entry N),
and every SPINE interaction was assigned a Truth_Spill vertex: 196 normal and 504 throughgoing.
"Assigned" means the alignment picked a vertex, not that it is *correct*; correctness is measured
by the cross-check against Method 2 further down.

```
SPINE event 0 -> Truth_Spill 0 (SpillNo 0): 28/28 normal and 65/65 throughgoing interactions assigned
SPINE event 1 -> Truth_Spill 1 (SpillNo 1): 32/32 normal and 81/81 throughgoing interactions assigned
SPINE event 2 -> Truth_Spill 2 (SpillNo 2): 19/19 normal and 75/75 throughgoing interactions assigned
SPINE event 3 -> Truth_Spill 3 (SpillNo 3): 27/27 normal and 78/78 throughgoing interactions assigned
SPINE event 4 -> Truth_Spill 4 (SpillNo 4): 29/29 normal and 71/71 throughgoing interactions assigned
SPINE event 5 -> Truth_Spill 5 (SpillNo 5): 27/27 normal and 69/69 throughgoing interactions assigned
SPINE event 6 -> Truth_Spill 6 (SpillNo 6): 34/34 normal and 65/65 throughgoing interactions assigned
```

Set `verbose_print = True` to also print the first few `ixn -> vertex` assignments per event.

### Method 2 — position-based matching

```
=== Position-based matching ===
700 / 3135 vertices matched within 1.0 mm

Summary per Truth_Spill entry:
  Truth_Spill 0: 93/455 vertices matched; SPINE events {0: 93}
  Truth_Spill 1: 113/446 vertices matched; SPINE events {1: 113}
  Truth_Spill 2: 94/429 vertices matched; SPINE events {2: 94}
  Truth_Spill 3: 105/444 vertices matched; SPINE events {3: 105}
  Truth_Spill 4: 100/449 vertices matched; SPINE events {4: 100}
  Truth_Spill 5: 96/461 vertices matched; SPINE events {5: 96}
  Truth_Spill 6: 99/451 vertices matched; SPINE events {6: 99}
```

Only ~22% of Truth_Spill vertices (700/3135) have a SPINE counterpart within 1 mm. This is
**expected, not a matching failure**: Truth_Spill lists every neutrino vertex in the spill,
wherever it happened — most are in the surrounding rock, hall, or TMS (the flagged external
vertices are 92–96% of each entry). SPINE keeps only interactions that leave charge in ND-LAr. That
includes some external vertices whose particles enter ND-LAr — Method 2 matches these too, since
it uses positions rather than IDs — but most vertices in the spill never put anything into
ND-LAr and so have no SPINE interaction to match. In the matched rows inspected, the
vertex distance was below 0.001 mm, i.e. the same vertex up to float precision. Each Truth_Spill entry's matched vertices land in exactly one
SPINE event, the one with the same index (the "SPINE events" dict never splits across events),
confirming the entry-to-event correspondence.

### Cross-check — Method 1 vs Method 2

```
ID matches that agree with position matches: 274/700 (normal 195/196, throughgoing 79/504)
```

Every ID-based match also had a position match to compare against.

| | Earlier Method 1 (normal only) | Current Method 1 |
|---|---|---|
| Normal interactions | 185/196 (94%) | **195/196 (99.5%)** |
| Throughgoing interactions | not attempted | **79/504 (16%)** |
| Total | 185/196 | 274/700 |

**Normal interactions: nearly exact.** In the earlier version, the flagged Truth_Spill vertices
stayed in the normal alignment as empty-signature candidates (their particles' IDs were stripped
of the flag, so they never matched the still-flagged vertex keys). Taking them out of the normal
alignment removed 10 of the 11 wrong assignments. The earlier per-event breakdown (from a debug
run of that version) was 27/28, 32/32, 13/19, 25/27, 29/29, 26/27, 33/34 for events 0–6; the
per-event breakdown of the single remaining error was not recorded.

**Throughgoing interactions: mostly wrong, and not fixable by tuning.** The ordering is not the
problem (0 violations in 497 consecutive pairs). The signatures carry almost no information:
- SPINE's throughgoing interactions are mostly secondary particles created outside ND-LAr: only
  414 of their 2,647 particles are primaries. Truth_Spill lists mostly primaries, so secondaries
  can't overlap: all 403 SPINE particles that do appear in their true vertex's signature are
  primaries.
- 373 of 504 interactions share exactly **one** `(track, pdg)` pair with their true vertex,
  usually just the muon `(0, 13)`; 118 share **none**; only 13 share two or more.
- The true vertex has the highest overlap outright for only 6 of 504 interactions; for 363 it
  ties with other vertices, since many vertices also have a `(0, 13)` muon.
- With ~70 SPINE throughgoing interactions against ~420 throughgoing Truth_Spill vertices per
  spill and nothing to distinguish them, the ordering alone cannot pick the right vertex.
- An alternative score that rewards SPINE particles found in the Truth_Spill vertex (rather than
  the reverse) did worse: 53/504.

This limit comes from what the files contain, not from the score. **Use Method 2 as the
reference** for all interactions; Method 1 is an independent cross-check that is reliable for
normal interactions only. Method 3 builds on Method 2's matches, not Method 1's.

### Vertex time as an alternative to position (tested, not used)

Tested whether the vertex time could replace x/y/z for vertex matching: SPINE `ixn.t` vs
Truth_Spill `TrueVtxT` (both in ns, both absolute, with spill N starting at N × 1.2e9 ns, so no
offset is needed). Each of the 700 position-matched SPINE interactions was matched to the
Truth_Spill vertex nearest in time, within a tolerance, and compared with the position match.
This was a one-off debug check and is not part of `match_spine_to_truth_spill.py`.

| Tolerance | Spill 0 | Spills 1–6 | Total |
|---|---|---|---|
| 0.001 ns | 93/93 | 0/607 | 93/700 |
| 0.01 ns | 93/93 | 0/607 | 93/700 |
| 1 ns | 93/93 | 1/607 | 94/700 |
| 300 ns | 93/93 | 118/607 | 211/700 |

**Time is not reliable in these files: it works for spill 0 only.**

**Why spills 1–6 fail.** `TrueVtxT` is stored as **float32** (`float[]`), and the times are
absolute. At 1.2e9–7.2e9 ns float32 can only represent times in steps of 128 ns (spill 1),
256 ns (spill 2) or 512 ns (spills 4–6). The measured SPINE − Truth_Spill differences are exactly
half those steps: up to ±64 ns (spill 1), ±128 ns (spills 2–3) and ±256 ns (spills 4–6). SPINE
keeps full precision. In spills 1–6 most Truth_Spill vertices round to the same few values (the
median gap between consecutive vertex times is 0 ns), so time cannot distinguish them.

**Why spill 0 works, but with little margin.** Spill 0 times are 0–10,000 ns, where float32 is
precise to ~0.001 ns; SPINE and Truth_Spill agree to ~0.0001 ns (std 0.00014 ns). However, the
beam's bunch structure packs vertices close together in time: for 59 of the 93 matched
vertices another vertex lies within 1 ns, and the closest is 0.005 ns away. Time matching
therefore needs a ~0.001 ns tolerance, whereas position matching separates vertices by
millimetres or more.

**Could it be fixed?** Not from these files; the precision was lost when the time was written.
If the TMS reco stored `TrueVtxT` as a double, or relative to the start of its spill, time
matching would probably work in every spill (not tested). Other TMS time branches (e.g.
`NeutrinoX4` in `Truth_Info`) were not checked for the same problem.

**Conclusion:** keep position (Method 2) as the reference. Time could serve as an extra check in
spill 0 only.

### Method 3 — particle/track matching

Run end-to-end on file `0000005` (all 7 spills) inside the `spine-nb` container:

```
=== Particle / track matching within matched vertices ===
Truth_Spill particles in matched vertices: 5276  {'matched': 1065, 'rejected': 0, 'tms_only': 4211}
  tms_only PDGs: {2112: 1172, 2212: 1138, 111: 518, 211: 466, -211: 296, 22: 288, 13: 108, 14: 83, ...}
TMS reco tracks from matched vertices: 255, linked to a SPINE particle: 228
```

| | Count |
|---|---|
| Truth_Spill particles in matched vertices | 5,276 |
| Matched (track ID + PDG + birth position) | 1,065 |
| Rejected (same track ID, disagreeing PDG or position) | 0 |
| No SPINE particle with that track ID (`tms_only`) | 4,211 |
| TMS reco tracks from matched vertices | 255 |
| Reco tracks linked to a SPINE particle | 228 |

**Track IDs are reliable within a correctly matched vertex.** Before writing Method 3, a check on
the first 3 spills found that every Truth_Spill particle whose track ID exists in the matched
SPINE interaction (265/265) has the same PDG and a birth position within 1 mm. The full run
agrees: 0 rejected out of 1,065 track-ID matches.

**The reco-track link is exact.** All 525 TMS reco tracks in `Truth_Info` were checked:
`RecoTrackPrimaryParticleIndex` always points to a Truth_Spill particle whose `VertexGlobalID`
equals `RecoTrackPrimaryParticleVtxGlobalID` and whose `PDG` equals
`RecoTrackPrimaryParticlePDG` (525/525 for both).

**Why so many Truth_Spill particles are `tms_only`:**
- About half are neutral (neutrons, π⁰, photons, neutrinos), which leave no charge in ND-LAr, so
  SPINE has no particle for them.
- Of the 2,017 charged ones (μ±, p, π±), 1,929 come from vertices **outside** ND-LAr; 1,927 of
  those both start and end outside ND-LAr, so they never enter it and SPINE correctly has no
  record of them.
- About 88 charged particles come from vertices **inside** ND-LAr. This was not investigated
  further; the likely explanation (unverified) is that they deposit too little charge for SPINE
  to keep them.
- Renumbering check: if a Truth_Spill particle existed in SPINE under a *different* track ID, we
  would be undercounting matches. Only 25 still-unmatched SPINE particles share a PDG and birth
  point with a `tms_only` particle (e.g. sibling primaries, which all start at the vertex), and
  **none** has the same initial energy (within 1 MeV). So track-ID matching is not missing
  renumbered particles.

**Unlinked reco tracks.** The 27 reco tracks (of 255) without a SPINE link all belong to
`tms_only` particles: by true PDG, 9 photons, 2 K⁰L, 1 neutron, 1 antineutron, 6 μ⁻, 1 μ⁺, 4 π⁺, 1 π⁻,
2 protons. The neutral ones are expected; where the charged ones come from was not checked.

### LArCV writer — `create_LArCV_NDLAr_TMS_matched.py`

Run on file `0000005` in the `spine-nb` container by calling the script's functions directly
(`pair_spills`, `match_particles`, `make_larcv_box`, `process_one_spill`) and writing to a
temporary file, then reading the file back with a larcv `IOManager`. The `__main__` loop itself
was not run, to avoid sweeping every file in the data directory; it calls the same functions.

| Spill | ND-LAr fragments (matched) | TMS fragments (sharing a group with ND-LAr) | Groups |
|---|---|---|---|
| 0 | 537 (140) | 80 (30) | 585 |
| 1 | 774 (168) | 78 (37) | 812 |
| 2 | 544 (116) | 75 (35) | 582 |
| 3 | 844 (156) | 73 (32) | 883 |
| 4 | 837 (142) | 84 (32) | 888 |
| 5 | 694 (142) | 72 (34) | 732 |
| 6 | 891 (160) | 63 (28) | 926 |

Checks:
- **Spill pairing:** every SPINE event paired with the Truth_Spill entry of the same index
  (event N → entry N, from 93–113 matched vertices each).
- **Read-back:** the file has 7 entries (one per spill), and in every entry:
  - the number of voxel sets equals the number of particles,
  - every `fragment_id` is unique,
  - every voxel set's id equals its particle's `fragment_id`,
  - no voxel set is empty (no point fell outside the box).
- **Shared groups agree with Method 3:** across the 7 spills, 228 TMS reco tracks share a group
  with an ND-LAr fragment — the same 228 reco tracks Method 3 links to a SPINE particle.
- **Matched ND-LAr fragments are fewer than Method 3 particle matches** (e.g. spill 0: 140
  written vs 149 matched) because matched particles with fewer than 5 points are dropped by the
  size threshold.
- **Groups with more than one fragment** (e.g. 32 in spill 0) are slightly more than the shared
  ND-LAr/TMS groups (30), since a Truth_Spill particle can have more than one TMS reco track.
- **Box meta read back as expected:** x −4000 to 4000, y −3705 to 1000, z 4000 to 19000 mm,
  2667 × 1569 × 5000 voxels.
- **Shapes** (spill 0, all 617 fragments): 307 track (227 ND-LAr + the 80 TMS reco tracks),
  214 delta, 88 shower, 8 Michel — ND-LAr particles keep their SPINE shape; TMS reco tracks are
  tracks.
