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
or `VertexGlobalID` (it's a run prefix + a running counter over the interactions edep-sim saved;
edep-sim, not SPINE, does not save empty interactions). The
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
  the interactions edep-sim saved (edep-sim, not SPINE, does not save empty interactions, so the
  counter skips vertices), e.g. `50000000, 50000001, ...` for vertices
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

A fully independent, geometry-only match, and the **reference** method (Method 2.5 is built on
its matches): each Truth_Spill vertex (`TrueVtxX/Y/Z`, mm) is
matched to its nearest SPINE true-interaction vertex (`ixn.vertex`, converted cm → mm), and kept
only if the distance is within `VERTEX_TOL_MM` (1.0 mm). This method makes no use of particle
content or ordering, so agreement with Method 1 confirms the signature/alignment logic is sound.

The script then cross-checks the two methods directly: for every ID-based match, it checks
whether the same (event, interaction) pair also has a position match to the same Truth_Spill
vertex.

### Method 2.5 — particle/track matching (`match_particles`)

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

Interactive companion notebook: reruns Methods 1, 2 and 2.5 from the script with the same logic
and the same numbering (Method 2, position, is the reference), then builds a vertex table, a
particle table, and a 3D event display that plots either whole vertices or individual
particle/track matches. Its printed results match the script's on file `0000005`.

**ND-LAr is drawn with SPINE reco particles, TMS with true hits and reco tracks.** Each SPINE reco
particle is assigned to its best-match true particle, `reco.match_ids[0]` — the same reco → truth
match `create_LArCV_NDLAr_TMS_matched.py` uses — so the colour coding matches the LArCV grouping:
a chosen particle's ND-LAr reco particles, TMS true hits and TMS reco tracks all share its colour.

- **Load files** — loads `Truth_Spill`; `Truth_Info` for the TMS true hits (`TrueHitX/Y/Z`,
  tagged by `TrueHitVertexId` and `TrueHitPrimaryId`) and the reco-track truth link
  (`RecoTrackPrimaryParticleIndex`); and `Reco_Tree` for the reco-track hit positions
  (`TrackHitPos[track][hit]`, first `nHits[track]` entries). `Reco_Tree` lines up entry by entry
  with `Truth_Info` (same spill, event and track count). Runs the `spine` driver with
  `build.mode: both` (truth *and* reco) and `post: match: particle: match_mode: both`, which fills
  each reco particle's `match_ids` (this post-match step makes loading slower).
- **Reco index** — `reco_by_truth[(event, true particle id)]` lists the reco particles whose
  best match is that true particle; `reco_no_truth[event]` lists reco particles with no truth
  match. On `0000005`: 5,131 reco particles, 5,096 with a truth match, 35 without (the same 35
  the LArCV script drops). `ndlar_reco_points(event, true_ids)` returns the reco points (mm).
- **TMS hits per particle** — `TrueHitPrimaryId` is a Truth_Spill `TrackId`: for 99.3% of TMS
  true hits (113,506/114,290) it equals the track ID of a Truth_Spill particle at the hit's
  vertex. Selecting hits by vertex ID + track ID therefore gives a particle's TMS hits, including
  hits from its descendants.
- **Method 1 / Method 2 / cross-check** — same logic as the script (both Method 1 streams), with
  the cross-check printed separately for normal and throughgoing interactions.
- **Method 2.5** — same as the script; results in `particle_matches`.
- **Vertex table** (`match_df`) — one row per position-matched vertex: `throughgoing`,
  `id_method_agrees`, `n_particles_matched`, `n_reco_tracks_linked`, and hit counts:
  `n_NDLAr_points` (true), `n_NDLAr_reco_points` (reco particles matched to the interaction's true
  particles) and `n_TMS_hits`. `both` keeps the 384 vertices with ND-LAr *reco* points and TMS
  hits (391 when it used true points).
- **Particle table** (`particle_df`) — one row per *matched* particle (1,065): track ID, PDG,
  SPINE particle id, birth distance, `n_NDLAr_points` (the true particle's own points),
  `n_reco_particles` and `n_NDLAr_reco_points` (its matched reco particles), `n_TMS_hits` (TMS
  true hits with that track ID, so descendants included), and the linked `reco_tracks`. 959 of
  the 1,065 have at least one reco particle, 188 have more than one; the other 106 show no
  ND-LAr points in the plots. `particles_both` keeps the 363 with ND-LAr reco points and TMS hits
  (373 when it used true points); `particles_reco` keeps the 228 with a TMS reco track.
- **3D event display** — plotly `Scatter3d` with the TMS and ND-LAr volumes drawn as wireframe
  boxes (geometry from `TMS_Constants.h`). Two entry points, described below:
  `draw_vertices(rows)` and `draw_particles(rows, ...)`.

### Setup

The first cells set `SOFTWARE_DIR` (added to `sys.path` so `spine` can be imported from a local
install) and `data_dir` (where the two input files live). Both need editing for each machine; in
the `spine-nb` container `spine` is already importable and the files are under `/data`.

### `draw_vertices(rows)`

Takes `match_df` rows: the ND-LAr reco particles matched to the interaction's true particles, all
TMS true hits of the vertex, and both vertex markers.

### `draw_particles(rows, ...)`

Takes `particle_df` rows. For each chosen particle it draws, in its strong colour, every ND-LAr
SPINE reco particle whose best match is that true particle (one legend/table entry per reco
particle — the plotting equivalent of the LArCV script's "different fragment ID, same group"),
the TMS true hits with its track ID (light shade of the same colour), and each linked TMS reco
track (hits joined by a line). The notebook's example selects every particle in `particles_reco`
from one spill (set by `spill_num`; e.g. spill 6 has 28 such particles, 24 from throughgoing
vertices), and saves the plot to HTML.

```
draw_particles(rows, show_vertex=True, show_legend=True, show_other_particles=False,
               show_table=True, given_title=None, show_all_activity=False, toggles=True,
               save_html=None, return_fig=False, tracks_only=False, throughgoing_only=False,
               vertex_filter="both")
```

**Layers** (each has an on/off button when `toggles=True`):

| Option | Effect |
|---|---|
| `show_vertex` | Vertex markers: SPINE vertex (red open square) and Truth_Spill vertex (black cross), once per vertex. |
| `show_other_particles` | Everything else at the chosen particles' vertices: other *matched* particles in colour, drawn like the chosen ones; *unmatched* activity in gray, per vertex: the reco particles of SPINE true particles with no Truth_Spill match (dark gray `#7f7f7f`, ND-LAr) and TMS true hits not from a matched particle (light gray `#c7c7c7`, TMS), each split into track / non-track traces. |
| `show_all_activity` | Everything else in the same spill(s), in faint gray and smaller markers: the reco particles of every other SPINE interaction (ND-LAr, `#b5b5b5`), every other Truth_Spill vertex's TMS true hits (`#dedede`), and every TMS reco track not already drawn (`#9e9e9e` lines). Split per spill into track / non-track × throughgoing / non-throughgoing traces. Reco particles with **no truth match** have no true vertex, so they appear only here, as two extra traces split by their *reco* shape and left alone by the vertex filter. Implies `show_other_particles`. |

**Filters** (buttons when `toggles=True`); they act on every layer that is on, and **do not turn
any layer on**:

| Option | Effect |
|---|---|
| `tracks_only` | On/off button. Only track-shaped particles, judged by the *true* particle (as in the LArCV script): a reco particle follows its matched true particle's shape; reco particles with no truth match use their reco shape; TMS reco tracks count as tracks (when drawn with a matched particle they follow that particle's shape); TMS true hits of TMS-only particles, which have no SPINE shape, count as track-like if their PDG is μ±, π±, K± or p/p̄ (`TRACK_LIKE_PDGS`, a PDG proxy). E.g. Tracks only + All other activity shows only the tracks in the activity layer. |
| `vertex_filter` | A **three-button group, Throughgoing only \| Vertex inside only \| Both**: one option is selected at a time and stays highlighted (a plotly button can only flip between two states, so one button per option is used instead of a single cycling button). Throughgoing only shows particles, and vertex markers, from throughgoing (1e15-flagged) vertices; Vertex inside only shows those from non-throughgoing vertices; Both (default) shows all. The option sets the starting choice: `"throughgoing"`, `"inside"` or `"both"`. Activity-layer TMS reco tracks are throughgoing if their true particle's vertex is. `throughgoing_only=True` is the older name for `vertex_filter="throughgoing"`. |

**Other options:**

| Option | Effect |
|---|---|
| `toggles` | `True` (default): all three optional layers are always built, with a row of layer on/off buttons and a second row with the Tracks only on/off button and the three-button vertex-filter group (plotly `updatemenus`; an on/off button's first click flips the setting, the second flips it back). A trace shows only if its layer is on and no active filter hides it. The `show_*`, `tracks_only` and `vertex_filter` options then only set the starting state. `False`: only what starts visible is built, with no buttons. |
| `show_legend` | `False` removes the plot legend entirely, so the table is the reference. |
| `show_table` | Displays a table under the plot with one row per legend entry: a colour swatch, colour, marker type, item (spill, interaction, track, PDG), what is shown (e.g. "ND-LAr reco particle 58 (reco shape 1)"), a count, the **layer** it belongs to, whether it counts as a **track** and as **throughgoing** (`–` = not filtered, e.g. detector boxes, or reco particles with no truth match for throughgoing), and whether it was **visible at start** (layers and filters both accounted for). With toggles on it lists every layer, including hidden ones. |
| `given_title` | Plot title (default: the list of `particle_df` rows). |
| `save_html` | Path of an HTML file to also write the plot to, with the legend table below it. Self-contained (plotly.js embedded, ~3.5 MB), so it opens offline in any browser with all buttons working. Hidden layers are saved too, so with toggles on the file is about 8 MB. `draw_vertices(rows, save_html=...)` saves its plot the same way, without a table. |
| `return_fig` | Return the plotly figure for further changes; assign it (`fig = draw_particles(...)`) so Jupyter doesn't display it twice. Off by default. |

**Colours.** The first 7 particles use the notebook's 7-colour palette; later ones use plotly's
Dark24 (strong) / Light24 (light) sets, so colours repeat after 31 objects — the table's item
column tells repeats apart. Grays are reserved for unmatched and background activity.

**Things to know:**
- **Rerun from the loading cell** after updating: the SPINE config gained the post-match step.
- **How the buttons combine.** Layer buttons switch traces' `visible` property; the filters hide
  traces by making them transparent instead (Tracks only: trace opacity; vertex filter: marker
  opacity and line width). Plotly buttons cannot see each other's state, so using different
  properties is what lets them all combine. Side effect: filtered traces stay in the plot legend
  and may still show hover labels.
- **Trace count.** Splitting every layer by track and throughgoing, and giving each reco
  particle its own entry, makes the legend and table long: 244 traces for the spill-0 example
  with all activity on.
- With `toggles=True` the faint activity layer is always built, even when hidden, so rendering
  is slower; use `toggles=False` if that matters.
- The activity layer leaves out what the "other particles" layer draws. Turning "other
  particles" off while "all other activity" is on hides those particles in both.
- A matched true particle with no reco particle (106 of 1,065) shows no ND-LAr points; ND-LAr
  charge not in any SPINE reco particle is not drawn by any layer.
- **Testing.** The notebook was run end to end in the container on `0000005`, building the
  figures and simulating what each trace renders at the start (the buttons have not been clicked
  in a rendered plot). Checks: each chosen particle got one ND-LAr entry per matched reco
  particle, in its colour; the activity layer contains all four track/throughgoing reco buckets
  plus the no-truth-match reco particles; Tracks only controls exactly the traces the table marks
  track "no"; the three vertex-filter options hide exactly the throughgoing-"no" traces, the
  throughgoing-"yes" traces, and nothing; `vertex_filter="inside"` starts with no throughgoing
  trace rendered and the table's "visible at start" agrees; `throughgoing_only=True` starts on
  "Throughgoing only"; and `toggles=False` with `"inside"` keeps only inside traces.

The container has no `ipywidgets`, so rows are chosen by editing the list in the cell that calls
each plotting function, and toggling uses plotly's own buttons rather than widgets.

## create_LArCV_NDLAr_TMS_matched.py

Writes ND-LAr + TMS LArCV files for GrapPA training, using the Method 2 + Method 2.5 matches as the
cross-detector truth. Both detectors contribute **reco** tracks: SPINE reco particles in ND-LAr and
TMS reco tracks in TMS; truth is used only to decide which ones to keep and how to group them.
Modelled on `create_LArCV_minerva_2x2.py` (the per-particle fragment pattern in its lines
269–300, and its file loop and train/validation split), with the 2x2 + MINERvA inputs replaced by
SPINE + TMS. It is separate from the older `create_LArCV_TMS_NDLAr.py`, which matches by vertex
time and PDG.

**One LArCV entry per spill.** SPINE events are paired with Truth_Spill entries by majority vote
of the Method 2 vertex matches (`pair_spills`), not by assuming equal indices.

**Fragments** (each one LArCV `Particle` + `VoxelSet`, written to `cluster3d_pcluster` and
`particle_corrected` as in the reference). **Only tracks are saved**, judged by the *true*
particle's shape:
- **ND-LAr: one fragment per SPINE reco particle**, using its reco points (cm → mm). Each reco
  particle is linked to its true particle with SPINE's reco → truth match, `reco.match_ids[0]`
  (the best-overlap true particle; the driver runs `post: match: particle: match_mode: both` so
  these are filled). A reco particle is
  - **dropped** if it has no truth match (counted and printed per spill),
  - **dropped** if its true particle's shape is not track (`shape != 1`), whatever its reco shape,
  - **dropped** if it has fewer than 5 points (the reference's size threshold),
  - otherwise saved.
- **TMS: one fragment per TMS reco track**, using its `Reco_Tree` hit positions
  (`TrackHitPos[track][:nHits]`, mm). A TMS reco track is dropped only if its true Truth_Spill
  particle was matched (Method 2.5) to a SPINE true particle whose shape is not track (counted and
  printed per spill). TMS-only particles have no SPINE shape, so their reco tracks are kept.
- Every saved fragment is labelled `kShapeTrack`. `fragment_id` (`particle.id`, also the voxel set
  id) is a counter that restarts at 0 in each spill. `pdg_code` and `track_id` of the true
  particle are also set on each particle, for debugging (the reference doesn't set them).

**`group_id` is one per true particle**, keyed on the Truth_Spill particle where there is one:
1. SPINE reco track → SPINE true particle (`reco.match_ids[0]`).
2. SPINE true particle → Truth_Spill particle (Method 2.5: Method 2 vertex match, then track ID,
   confirmed by PDG and birth position within 1 mm).
3. Truth_Spill particle → every TMS reco track whose `RecoTrackPrimaryParticleIndex` points to
   it (the true particle TMS reco attributes the track to).

The ND-LAr reco track reaches the Truth_Spill particle through steps 1–2 and each TMS reco track
through step 3, so they share its group: these shared groups are the cross-detector matches
GrapPA should learn. Consequences:
- Two or more SPINE reco tracks pointing to the same true particle (e.g. a split reco track) get
  different fragment IDs and the same group; so do two TMS reco tracks from the same particle.
- A SPINE true particle not matched to TMS gets its own group (keyed on the SPINE true particle).
- Every TMS reco track is written whether or not an ND-LAr reco track reached the same
  Truth_Spill particle; if the ND-LAr side was dropped (no truth match, not a track, too small),
  the TMS track keeps the group alone.

**Box (`make_larcv_box`).** x −4000 to 4000, y −3705 to 1000, z 4000 to 19000 mm, covering both
active volumes (limits in the code comments); the same box `Plotting_validation_TMS.ipynb` uses.
`voxel_size=3.72` mm, the ND-LAr / SPINE pitch (0.372 cm). Because the box edges are not whole
multiples of 3.72 mm, the voxel count is rounded up (2151 × 1265 × 4033) and the actual voxel
size is slightly smaller, **3.7192 mm** in x. To get exactly 3.72 mm, each box maximum would need
extending to the next multiple of 3.72 mm (under 4 mm per axis; not done). The older
`create_LArCV_TMS_NDLAr.py` uses 1.5 mm voxels and y from −3700, which cuts 2.2 mm off the bottom
of TMS.

**Running it.** The `__main__` loop globs `directory` (default `./`) for `*.TMSRECO.root`
(sorted by name; set `N_FILES` to an integer to use only the first N, `None` for all) and pairs each
with the SPINE file of the same 7-digit number, dropping TMS files without exactly one match.
It then splits the *usable* pairs by `TRAIN_FRACTION` (0.8): `round(0.2 × n)` validation files,
spread evenly through the name-sorted list (`np.linspace`), go to `larcv/larcv_validation_set/`
and the rest to `larcv/` — e.g. 100 pairs → 80/20, 23 → 18/5, 10 → 8/2, 5 → 4/1, and 1 or 2
pairs → all training. It skips existing outputs unless
`rewrite_output = True`. Unlike the reference (where they are commented out), it calls
`save_entry()` for every spill and `finalize()` at the end of each file. It prints a summary line
per spill, and at the end a table of, per file and spill, the TMS reco tracks removed (true
particle matched to a non-track SPINE particle) and the ND-LAr reco particles dropped for having
no truth match. Needs `spine` and `larcv` (both available in the `spine-nb` container).

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
vertices are 92–96% of each entry). On the ND-LAr side, edep-sim — not SPINE — does not save
empty interactions, so only interactions that deposit energy in ND-LAr reach SPINE. That
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
normal interactions only. Method 2.5 builds on Method 2's matches, not Method 1's.

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

### Method 2.5 — particle/track matching

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

**Track IDs are reliable within a correctly matched vertex.** Before writing Method 2.5, a check on
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

Run on file `0000005` in the `spine-nb` container through the real `__main__` loop, with
`directory` pointed at a temporary folder holding links to just the 0000005 SPINE and TMS files
(at the time the split was "every 5th file", so this single file went to the validation set;
with the current split a single file goes to training). The output was then read back with a larcv `IOManager`. This
is the reco-track version; an earlier version that saved SPINE *true* particles (3 mm voxels) is
superseded.

Per-spill summary:

| Spill | ND-LAr reco tracks saved | … sharing a true particle with another reco track | Dropped: no truth match | Dropped: true particle not a track | Dropped: < 5 points | TMS reco tracks saved | … sharing a group with ND-LAr | TMS dropped: not a track | Groups |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 236 | 45 | 5 | 303 | 15 | 80 | 30 | 0 | 260 |
| 1 | 325 | 50 | 3 | 427 | 24 | 77 | 36 | 1 | 337 |
| 2 | 181 | 27 | 2 | 324 | 13 | 75 | 35 | 0 | 205 |
| 3 | 357 | 56 | 6 | 502 | 19 | 73 | 32 | 0 | 365 |
| 4 | 317 | 42 | 6 | 478 | 20 | 84 | 32 | 0 | 347 |
| 5 | 278 | 34 | 5 | 370 | 15 | 72 | 34 | 0 | 297 |
| 6 | 405 | 54 | 8 | 464 | 23 | 63 | 28 | 0 | 410 |
| **Total** | | | **35** | | | | | **1** | |

The run's own per-spill printout also gives "matched to TMS" (e.g. 146 in spill 0): ND-LAr reco
tracks whose true particle Method 2.5 matched to a Truth_Spill particle. That does not mean a TMS
reco track exists for it — that is the "sharing a group with ND-LAr" column.

Checks:
- **Spill pairing:** every SPINE event paired with the Truth_Spill entry of the same index.
- **Read-back:** the file has 7 entries, and in every entry every `fragment_id` is unique, every
  voxel set's id equals its particle's `fragment_id`, no voxel set is empty, and every fragment is
  labelled track (`shape 1`).
- **Shared ND-LAr/TMS groups:** the TMS reco tracks sharing a group with ND-LAr (30, 36, 35, 32,
  32, 34, 28) are within one of the earlier true-hit version (30, 37, 35, 32, 32, 34, 28): spill
  1's single TMS reco track removed as non-track accounts for the difference.
- **Groups with more than one fragment:** 41–51 per spill (47 in spill 0), from split ND-LAr reco
  tracks, ND-LAr/TMS matches, and TMS particles with several reco tracks.
- **Box meta read back:** 2151 × 1265 × 4033 voxels, 3.7192 mm in x (see the box note above).
- `N_FILES` and the `TRAIN_FRACTION` split were added after this run and have not been run on
  files; the split logic was checked on its own for 1–100 files (counts as listed above).
