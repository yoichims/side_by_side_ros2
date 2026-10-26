# Side-by-side motion for personal mobility vehicles

Two personal mobility vehicles that navigate a shared global path while holding
a side-by-side social formation, so their passengers can hold a conversation.

Each vehicle runs its own model-predictive planner at 10 Hz. The joint decision
is a bimatrix game over pairs of candidate trajectories, and nothing crosses
between the vehicles but the odometry each could observe of the other. The
behavioural parameters are fitted from a corpus of 94 runs of human-driven
pairs; the lane geometry is derived offline from the corridor rather than tuned.

## Running the simulation

This repository is self-contained: it carries its own container definition, and
the repository itself is the colcon workspace, mounted at `/home/user/sbs_ws`.

Build the image once (`sbs_devel:humble`):

```bash
make
```

Start the container:

```bash
./run_container.sh
```

Then, inside it:

```bash
colcon build
source install/setup.bash
ros2 launch sbs_bringup joint_planning.launch.py direction:=ccw objects:=default
```

To open another shell in the running container:

```bash
./join_container.sh
```

`run_container.sh` probes the host at runtime and falls back rather than
assuming: NVIDIA passthrough only when the Container Toolkit is actually
installed (the driver alone is not enough, and `--gpus=all` without the
toolkit fails the container outright), `/dev/dri` rendering otherwise, and the
Wayland socket exposed on a Wayland session while `QT_QPA_PLATFORM` stays
pinned to `xcb` because RViz's OGRE render system requires GLX. Pass
`--wayland` to override that for pure-Qt tools, `-d <distro>` for a different
ROS 2 release, or set `SBS_NO_NVIDIA=1` to skip the NVIDIA path.

It also exports `FASTRTPS_DEFAULT_PROFILES_FILE` container-wide, so a second
shell running `ros2 topic echo` or its own RViz gets the large-message
transport without setting it by hand.

### Building

Inside the container:

```bash
source ./compile_and_source.sh
```

Extra arguments pass through to `colcon build`, so
`source ./compile_and_source.sh --packages-select sbs_planner` works.

### NVIDIA rendering

`run_container.sh` uses the NVIDIA GPU only if the **NVIDIA Container Toolkit**
is installed and falls back to `/dev/dri`
rendering on the integrated GPU when it is missing.

The toolkit can be installed with the following method:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`SBS_NO_NVIDIA=1 ./run_container.sh` skips the NVIDIA path even where it works.

## The joint planner

The system that is developed and measured now:

```bash
ros2 launch sbs_bringup joint_planning.launch.py direction:=ccw objects:=default
ros2 launch sbs_bringup joint_planning.launch.py direction:=cw  objects:=default
```

RViz and per-tick metrics are on by default. `objects:=none` runs the same route
with no cones. v1 is the outer lane (cyan fan), v2 the inner (orange).

Behaviour worth knowing before reading a run:

- The pair rides side by side where the corridor allows it and drops into **file** at
  pinches -- three places in ccw, three in cw, about 12 m of an ~88 m lap. That
  is deliberate. It is decided by an offline flag derived from the map *and* by
  the corridor measured live every tick, which is what lets a cone be moved
  without regenerating anything. Both can only ever withdraw the side by side
  option, never grant one. `runtime_room:=false` is the control arm.
- **Who passes an obstacle first is not decided.** The drivers imposed no order
  -- over 148 recorded passes the blocked vehicle led 51% of the time, and 78%
  of passes simply kept the order the pair already had. They did not need a rule
  because they were strung out to 1.62 m by the time they arrived. The
  along-track target is fitted from those passes (1.60 m, from 180 observations;
  the previous 1.45 came from six), and the pair now files far enough that the
  geometry settles the order. `yield_when_blocked:=true` restores the imposed
  rule: it buys 12 points of conversational geometry and costs two thirds of the
  obstacle clearance.
- Top speed is **1.24 m/s**, the vehicles' governor. The drivers' speeds pile
  up at that value because the governor capped them, so it is their ceiling
  too. Station keeping closes a stagger by *slowing the leader*, never by speeding the follower past the
  speed law fitted from the corpus.
- A **velocity limiter** sits between the planner and the vehicle, enforcing the
  same limits the planner samples from. It should never bind; if it logs
  `limiter bound N commands`, that is a bug report about the planner.

Tools, all reading the shipped files so a picture that disagrees with a run is
evidence about the run:

```bash
# offline geometry
python3 src/sbs_planner/tools/derive_lane_offsets.py <centerline> <map.yaml> --objects <objects.yaml> --write
python3 src/sbs_planner/tools/trim_objects.py --write   # fit surveyed cones to a cone's size
python3 src/sbs_planner/tools/make_layouts.py --count 8 --cones 4 --seed 1 --out layouts

# pictures
python3 src/sbs_planner/tools/plot_lanes.py             # lanes, per-obstacle insets
python3 src/sbs_planner/tools/plot_contacts.py '<metrics glob>'    # where bodies touched
python3 src/sbs_planner/tools/plot_pinned_site.py       # free bands vs the target asked for

# campaigns, and the analysis that reads them
bash src/sbs_planner/tools/run_pairs.sh                 # both arms over the cone layouts
OUT=ccw_runs DIRS=ccw REPEATS=3 bash src/sbs_planner/tools/run_repeats.sh
python3 src/sbs_planner/tools/summarise_layouts.py layout_runs
python3 src/sbs_planner/tools/compare_repeats.py ccw_runs
./in_container.sh python3 src/sbs_planner/tools/stall_harness.py   # replay recorded stall poses
```

Sweep output is not in the repository -- a campaign is ~100 MB of metrics CSVs
and node logs, and the runners regenerate it.

### Regenerating the cone layouts

The eight layouts used in the paper are in `layouts/`. They were generated with:

```bash
python3 src/sbs_planner/tools/make_layouts.py --count 8 --cones 4 --seed 1 --out layouts
```

The same seed regenerates them to within a millimetre; use the tracked files
to run against exactly the same cones.
`--cones` counts *obstacles*, and each is emitted as a pair of cones 0.53 m
apart -- the mean of the three pairs in the surveyed environment. A single
cone is not what the drivers met: a pair spans about 0.83 m, and the 0.23 m
between the two takes no vehicle, so the obstacle is wide rather than a gate.

Every obstacle is checked, in **both** directions, to leave a free band the
pair can use:

- at least **0.30 m** wide, the tightest band anywhere in the surveyed
  environment -- narrower than that and the vehicles cannot track through it,
- within **1.00 m** of each lane, the furthest the recorded drivers moved at a
  cone, and the cap on the planner's lateral target.

Both bounds are there because layouts that broke them produced failures that
read as planner defects. A band of 0.05-0.20 m took 14 of 20 stalls in one
campaign, and a 1.05 m band that sat 1.19 m from the outer lane pinned a
vehicle for 138 s of a 168 s run, in all six runs of an A/B, because the
target correctly refused to promise a swerve the vehicle could not track.

**Regenerate after any change to the map, the objects, or the lane profiles**
-- all three feed the checks above.

**After any edit to `map.pgm`, regenerate the lane offsets for both directions.**
They are derived from the map and nothing updates them automatically; a stale
profile once put one vehicle's lane 0.19 m inside newly added geometry.

## What runs

Per vehicle, inside its namespace (`v1`, `v2`):

```
map ---------------------> object_detector --> objects
                                 ^                |
utm_simulator --> scan ----------+                v
                                            joint_planner
odom <-- odo_simulator <-- cmd_vel <-- velocity_limiter <-+
```

Shared: one `nav2_map_server`, the reference publisher, the world-object
publisher, and the metrics node that watches both vehicles.

The planner is the controller -- there is no separate trajectory follower. It
samples candidates, scores them jointly with the partner's, and publishes the
first command of its choice at 10 Hz.

## Data

The recordings of the human-driven pairs used to fit the planner's parameters
are not distributed with this repository. They are available from the authors
on request.

## License

Apache License 2.0, see `LICENSE`.
