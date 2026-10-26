"""Two simulated PMVs planning jointly, each running its own MPC.

    publish/observe partner odom -> joint_planner -> cmd_vel -> odo_simulator

No trajectory_tracker: the planner is the controller. Each cycle it samples its
own rollouts, samples a prediction of the partner's from the partner's observed
state, scores every pair and commands the first segment of its choice.

Nothing but state crosses between the vehicles. Each planner subscribes to the
partner's odom -- what a driver could see -- and never to a plan.

    ros2 launch sbs_bringup joint_planning.launch.py
    ros2 launch sbs_bringup joint_planning.launch.py concept:=independent
    ros2 launch sbs_bringup joint_planning.launch.py objects:=none
    ros2 launch sbs_bringup joint_planning.launch.py direction:=ccw

Obstacles are data, not a second map. `utm_simulator` holds the survey as a
static grid and stamps whatever arrives on `dynamic_obstacles` into a copy of
it before each raycast, so `publish_world_objects.py` publishes the static
objects plus the partner's body and that is the whole obstacle story. One
publisher owns that topic on purpose: it is last-message-wins, so two would
overwrite each other.

    objects:=none      obstacle-free, laser still running
    objects:=default   the environment's objects.yaml (4 objects, 279 cells)

The old arrangement baked the objects into a second occupancy grid served to
the laser alone, which meant the obstacle-free variant was not obstacle-free
and the only way to remove them was to remove the laser. That second grid and
the `map_variant` switch that selected it are gone: there is one map.

Build up the cost one term at a time; each has a check:

    w_rel_range_rate:=0 w_rel_dist_end:=0 w_rel_yaw_rate:=0
        path following only -- isolates the control loop from the social
        model. If the pair tracks its lanes badly with every social term
        silenced, the MPC is at fault and not the cost.
    then restore the formation weights and watch rel_dist against the recorded
    1.41 m (cw) / 1.46 m (ccw).

Every run writes a per-tick CSV of both vehicles' state, their relative state
and their clearances:

    metrics_file:=<dir or .csv>      default is $SBS_WS/runs/, timestamped
    metrics:=false                   turn the logging off

The default lands inside the workspace on purpose: the dev container mounts
only the workspace, so a run logged to the container's /tmp cannot be read
from the host and does not survive the container.

Watching RViz cannot distinguish a pair that is holding formation badly from a
pair that is holding a bad formation well, which is what the CSV is for.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


#: Same rectangle the safety limiter and the partner-visibility publisher
#: use: 0.90 m long, 0.55 m wide, base_link 0.35 m behind the front face.
FOOTPRINT = [0.55, 0.275, 0.55, -0.275, -0.35, -0.275, -0.35, 0.275]

#: One colour per vehicle, matching the footprint colours in the RViz config,
#: so two overlaid rollout fans can be told apart.
FAN_COLOUR = {'v1': [0.14, 0.82, 1.00], 'v2': [1.00, 0.67, 0.00]}


def vehicle(ns, partner, side, start, reference, concept, gamma, weights,
            profile,
            publish_matrix, detect_objects, objects_file, inflate,
            yield_when_blocked, object_horizon, diagnose, sections,
            side_by_side_lookahead, wall_inflate, v_max, w_max,
            lat_acc_max, w_foresight, runtime_room, w_centre,
            pair_steps, escape_tolerance, escape_unstick, sbs_stagger,
            formation_separation, stop_when_blocked,
            brake_latency, backup_retrace, partner_hold, lane_band,
            exact_clearance, object_reach_min, yield_wait, speed_v0,
                objects_starved,
            boost_above_preferred,
            station_gain, speed_split,
            band_target, heading_cap,
            d_max, d_stop, braking_authority):
    x, y, yaw = start
    base = f'{ns}/base_link'
    laser = f'{ns}/laser'
    nodes = [
        PushRosNamespace(ns),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='map_to_odom',
             arguments=['0', '0', '0', '0', '0', '0', 'map', f'{ns}/odom']),
        Node(package='simple_simulator', executable='odo_simulator',
             name='odo_simple_simulator', output='screen',
             parameters=[{'odom_frame_id': f'{ns}/odom',
                          'base_frame_id': base,
                          'initialPoseX': x, 'initialPoseY': y,
                          'initialPoseTheta': yaw}],
             remappings=[('~/base_pose_ground_truth', 'odom'),
                         ('~/cmd_vel', 'cmd_vel')]),
        Node(package='sbs_planner', executable='joint_planner',
             name='joint_planner', output='screen',
             parameters=[{
                 'reference': reference,
                 'side': side,
                 'concept': concept,
                 'gamma': gamma,
                 'publish_matrix': publish_matrix,
                 'profile': profile,
                 'fan_color': FAN_COLOUR.get(ns, [0.14, 0.82, 1.00]),
                 'yield_when_blocked': yield_when_blocked,
                 'side_by_side_lookahead': side_by_side_lookahead,
                 'wall_inflate': wall_inflate,
                 'w_foresight': w_foresight,
                 'runtime_room': runtime_room,
                 'w_centre': w_centre,
                 'pair_steps': pair_steps,
                 'escape_tolerance': escape_tolerance,
                 'stop_when_blocked': stop_when_blocked,
                 'brake_latency': brake_latency,
                 'backup_retrace': backup_retrace,
                 'partner_hold': partner_hold,
                 'lane_band': lane_band,
                 'exact_clearance': exact_clearance,
                 'object_reach_min': object_reach_min,
                 'yield_wait': yield_wait,
                 'speed_v0': speed_v0,
                 'objects_starved': objects_starved,
                 'escape_unstick': escape_unstick,
                 'sbs_stagger': sbs_stagger,
                 'formation_separation': formation_separation,
                 'boost_above_preferred': boost_above_preferred,
                 'station_gain': station_gain,
                 'speed_split': speed_split,
                 'band_target': band_target,
                 'heading_cap': heading_cap,
                 'd_max': d_max,
                 'd_stop': d_stop,
                 'braking_authority': braking_authority,
                 'v_max': v_max,
                 'object_horizon': object_horizon,
                 'diagnose': diagnose,
                 **weights,
             }],
             remappings=[('partner_odom', f'/{partner}/odom'),
                         ('structure', 'world_structure'),
                         ('map', '/map'),
                         # Through the limiter, never straight to the vehicle.
                         ('cmd_vel', 'cmd_vel_raw')]),
        # Enforces the vehicle's own limits on whatever is commanding it. Same
        # numbers the planner samples from, so it is a no-op unless something
        # upstream is wrong -- and it scales v and w together, so a clamp
        # slows the vehicle along its path instead of reshaping the path.
        Node(package='sbs_planner', executable='velocity_limiter',
             name='velocity_limiter', output='screen',
             parameters=[{'v_max': v_max, 'w_max': w_max,
                          'lat_acc_max': lat_acc_max}]),
        # Draws the vehicle body. Without it RViz shows a bare odometry arrow,
        # and 0.20 m from a wall looks the same as 0.20 m of real clearance.
        Node(package='sbs_bringup', executable='publish_footprint.py',
             name='footprint_publisher', output='screen',
             parameters=[{'footprint': FOOTPRINT, 'frame_id': base}]),
    ]

    if detect_objects:
        nodes += [
            Node(package='tf2_ros', executable='static_transform_publisher',
                 name='tf_laser',
                 arguments=['0.03', '0', '0.24', '0', '0', '0', base, laser]),
            # Raycasts the *clean* map and whatever arrives on
            # `dynamic_obstacles`, which the simulator stamps into a copy of
            # the grid before each cast. There is no second occupancy grid any
            # more: objects are data.
            Node(package='simple_simulator', executable='utm_simulator',
                 name='simulator_hokuyo', output='screen',
                 parameters=[{'frame_id': laser, 'map_topic': '/map'}],
                 remappings=[('~/scan', 'scan')]),
            # Subtracts the *clean* map, so what is left is what the map does
            # not explain. The shadow filter and uni_density are skipped here:
            # they exist to clean real LiDAR shadows, which the raycast
            # simulator does not produce.
            Node(package='sbs_planner', executable='object_detector',
                 name='object_detector', output='screen',
                 parameters=[{'frame_id': 'map', 'wall_tolerance': 0.40,
                              'partner_radius': 0.80, 'max_range': 8.0}],
                 remappings=[('map', '/map'),
                             ('structure', 'world_structure'),
                             ('partner_odom', f'/{partner}/odom')]),
            # Everything the survey does not contain, as one cloud: the
            # static objects plus the partner's body. One publisher, because
            # `dynamic_obstacles` is last-message-wins -- two nodes writing to
            # it would each be visible about half the time.
            #
            # Injecting the partner is also what makes the two vehicles
            # visible to each other's LiDAR at all; without it they pass
            # straight through and neither occludes the other. The detector
            # still masks the partner out of `objects` by its observed pose,
            # because the formation terms own the partner: this changes what
            # the laser sees, not what counts as an obstacle.
            Node(package='sbs_bringup',
                 executable='publish_world_objects.py',
                 name='world_objects', output='screen',
                 parameters=[{'frame_id': 'map', 'spacing': 0.05,
                              'footprint': FOOTPRINT, 'rate': 10.0,
                              'objects': objects_file,
                              'sections': sections,
                              'inflate': inflate,
                              'vehicles': [f'/{partner}/odom']}]),
        ]

    return GroupAction(nodes)


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory('sbs_bringup')
    workspace = os.environ.get('SBS_WS', os.path.expanduser('~/sbs_ws'))
    env = os.path.join(workspace, 'environments', 'experiment')

    direction = LaunchConfiguration('direction').perform(context)
    concept = LaunchConfiguration('concept').perform(context)
    gamma = float(LaunchConfiguration('gamma').perform(context))
    publish_matrix = LaunchConfiguration('publish_matrix').perform(
        context).lower() == 'true'
    profile = LaunchConfiguration('profile').perform(context).lower() == 'true'

    reference = os.path.join(env, 'centerlines', f'{direction}_centerline.csv')
    map_file = os.path.join(env, 'map.yaml')
    for path in (reference, map_file):
        if not os.path.exists(path):
            raise RuntimeError(f'missing: {path}')

    weights = {f'w_{name}': float(LaunchConfiguration(f'w_{name}').perform(context))
               for name in ('rel_range_rate', 'rel_dist_end', 'rel_yaw_rate',
                            'motion_accel', 'motion_speed', 'path_dist',
                            'path_progress', 'rel_along_track', 'rel_yaw',
                            'path_cross_track', 'path_lane_yaw',
                            'rel_angle')}

    # Start each vehicle on its own lane, a little way along the reference.
    # Vehicle 1 takes the outer lane: stage 1 orders the recorded pair by path
    # length, so vehicle 1's data -- and therefore its lane -- is the outside
    # one.
    # Heading comes from the lane's own tangent. Taking it from the centreline
    # was wrong: the three curves are resampled independently, so row 5 of a
    # lane and row 5 of the centreline are not the same place, and the vehicle
    # started with a heading that did not match the lane it was on.
    import csv
    import math
    # Start each vehicle *on its target*, which since the offsets became a
    # derived profile is no longer the recorded lane. Taking the start from
    # the lane file left every run beginning 0.03-0.21 m off target -- a mean
    # of 0.12 m against a steady-state cross-track error of 0.08 m, so the
    # run opened with its largest tracking error and had to correct it while
    # accelerating from rest. On ccw that transient lands squarely on the
    # tightest stretch of the route, 5 m from the start, where the best
    # achievable body clearance is 0.111 m.
    starts = {}
    offsets_path = reference.replace('_centerline.csv', '_offsets.csv')
    offsets = None
    if os.path.exists(offsets_path):
        with open(offsets_path, newline='') as handle:
            table = list(csv.DictReader(handle))
        offsets = {'outer': [float(r['outer']) for r in table],
                   'inner': [float(r['inner']) for r in table]}
    with open(reference, newline='') as handle:
        centre = list(csv.DictReader(handle))

    for side in ('outer', 'inner'):
        lane = reference.replace('_centerline.csv', f'_lane_{side}.csv')
        if not os.path.exists(lane):
            raise RuntimeError(f'missing: {lane}')
        with open(lane, newline='') as handle:
            rows = list(csv.DictReader(handle))
        here, ahead = rows[5], rows[10]
        yaw = math.atan2(float(ahead['y']) - float(here['y']),
                         float(ahead['x']) - float(here['x']))
        x, y = float(here['x']), float(here['y'])

        if offsets is not None:
            # The lane row fixes *where along* the route to start; the
            # profile fixes how far across. Placed from the centreline so the
            # offset means the same thing it means to the planner.
            index = min(range(len(centre)),
                        key=lambda k: (float(centre[k]['x']) - x) ** 2
                        + (float(centre[k]['y']) - y) ** 2)
            cx, cy = float(centre[index]['x']), float(centre[index]['y'])
            tangent = float(centre[index]['yaw'])
            offset = offsets[side][index]
            x = cx - offset * math.sin(tangent)
            y = cy + offset * math.cos(tangent)
            yaw = tangent
        starts[side] = (x, y, yaw)

    detect_objects = LaunchConfiguration('detect_objects').perform(
        context).lower() == 'true'

    # Static objects, injected into the laser's world by message. Empty means
    # a genuinely obstacle-free run with the laser still running -- which the
    # old arrangement could not express, because the cones were baked into the
    # grid the laser raycast and the only way to remove them was to remove the
    # laser.
    inflate = float(LaunchConfiguration('inflate').perform(context))
    yield_when_blocked = LaunchConfiguration(
        'yield_when_blocked').perform(context).lower() == 'true'
    side_by_side_lookahead = float(
        LaunchConfiguration('side_by_side_lookahead').perform(context))
    wall_inflate = float(
        LaunchConfiguration('wall_inflate').perform(context))
    w_foresight = float(LaunchConfiguration('w_foresight').perform(context))
    runtime_room = LaunchConfiguration(
        'runtime_room').perform(context).lower() == 'true'
    w_centre = float(LaunchConfiguration('w_centre').perform(context))
    pair_steps = int(LaunchConfiguration('pair_steps').perform(context))
    escape_tolerance = float(
        LaunchConfiguration('escape_tolerance').perform(context))
    stop_when_blocked = LaunchConfiguration(
        'stop_when_blocked').perform(context).lower() == 'true'
    brake_latency = float(
        LaunchConfiguration('brake_latency').perform(context))
    backup_retrace = LaunchConfiguration(
        'backup_retrace').perform(context).lower() == 'true'
    partner_hold = LaunchConfiguration(
        'partner_hold').perform(context).lower() == 'true'
    lane_band = LaunchConfiguration(
        'lane_band').perform(context).lower() == 'true'
    exact_clearance = LaunchConfiguration(
        'exact_clearance').perform(context).lower() == 'true'
    object_reach_min = float(
        LaunchConfiguration('object_reach_min').perform(context))
    yield_wait = LaunchConfiguration(
        'yield_wait').perform(context).lower() == 'true'
    speed_v0 = float(LaunchConfiguration('speed_v0').perform(context))
    objects_starved = LaunchConfiguration(
        'objects_starved').perform(context).lower() == 'true'
    # Along-track offset asked for while side by side. 0.0 reproduces every
    # result measured before 2026-09-24; the drivers held 0.59-0.99 m.
    sbs_stagger = float(
        LaunchConfiguration('sbs_stagger').perform(context))
    escape_unstick = (
        LaunchConfiguration('escape_unstick').perform(context).lower()
        == 'true')
    formation_separation = (
        LaunchConfiguration('formation_separation').perform(context).lower()
        == 'true')
    boost_above_preferred = (
        LaunchConfiguration('boost_above_preferred').perform(context).lower()
        == 'true')
    station_gain = float(LaunchConfiguration('station_gain').perform(context))
    speed_split = float(LaunchConfiguration('speed_split').perform(context))
    band_target = LaunchConfiguration(
        'band_target').perform(context).lower() == 'true'
    heading_cap = LaunchConfiguration(
        'heading_cap').perform(context).lower() == 'true'
    d_max = float(LaunchConfiguration('d_max').perform(context))
    d_stop = float(LaunchConfiguration('d_stop').perform(context))
    braking_authority = LaunchConfiguration(
        'braking_authority').perform(context).lower() == 'true'
    v_max = float(LaunchConfiguration('v_max').perform(context))
    w_max = float(LaunchConfiguration('w_max').perform(context))
    lat_acc_max = float(
        LaunchConfiguration('lat_acc_max').perform(context))
    object_horizon = float(
        LaunchConfiguration('object_horizon').perform(context))
    diagnose = LaunchConfiguration('diagnose').perform(context).lower() == 'true'
    # `objects:=none` drops the cones and KEEPS the structure: the cones are
    # the experiment, the aisle is the building. Loading nothing at all was
    # the first behaviour here and it deleted the aisle along with the
    # obstacles, which would have made every obstacle-free run a different
    # world rather than the same one minus the cones.
    requested = LaunchConfiguration('objects').perform(context)
    default_objects = os.path.join(env, 'objects.yaml')
    if requested in ('', 'none'):
        objects_file, sections = default_objects, ['structure']
    elif requested == 'default':
        objects_file, sections = default_objects, ['objects', 'structure']
    else:
        objects_file, sections = requested, ['objects', 'structure']
    if not os.path.exists(objects_file):
        if requested not in ('', 'none'):
            raise RuntimeError(f'missing: {objects_file}')
        objects_file, sections = '', []
    managed = ['map_server']
    nodes = [
        # The one map: the planner's reference for wall clearance, the
        # background the detector subtracts, and the grid the laser raycasts.
        Node(package='nav2_map_server', executable='map_server',
             name='map_server', output='screen',
             parameters=[{'yaml_filename': map_file, 'frame_id': 'map'}]),
    ]
    nodes.append(
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_map', output='screen',
             parameters=[{'autostart': True, 'node_names': managed}]))

    return nodes + [
        Node(package='sbs_planner', executable='reference_publisher',
             name='reference_publisher', output='screen',
             parameters=[{'reference': reference, 'frame_id': 'map'}]),
        # Watches both vehicles and writes one CSV row per tick. Off the
        # command path entirely -- it subscribes and never publishes -- so it
        # can be left on for every run.
        Node(package='sbs_planner', executable='formation_metrics',
             name='formation_metrics', output='screen',
             condition=IfCondition(LaunchConfiguration('metrics')),
             parameters=[{
                 'reference': reference,
                 'output': LaunchConfiguration('metrics_file'),
                 'v1_side': 'outer', 'v2_side': 'inner'}]),
        vehicle('v1', 'v2', 'outer', starts['outer'], reference, concept,
                gamma, weights, profile, publish_matrix, detect_objects,
                objects_file, inflate, yield_when_blocked,
                object_horizon, diagnose, sections,
                side_by_side_lookahead, wall_inflate, v_max, w_max,
                lat_acc_max, w_foresight, runtime_room,
                w_centre, pair_steps, escape_tolerance, escape_unstick,
                sbs_stagger,
                formation_separation, stop_when_blocked,
                brake_latency, backup_retrace, partner_hold, lane_band,
                exact_clearance, object_reach_min, yield_wait, speed_v0,
                objects_starved,
                boost_above_preferred, station_gain, speed_split, band_target,
                heading_cap, d_max, d_stop, braking_authority),
        vehicle('v2', 'v1', 'inner', starts['inner'], reference, concept,
                gamma, weights, profile, publish_matrix, detect_objects,
                objects_file, inflate, yield_when_blocked,
                object_horizon, diagnose, sections,
                side_by_side_lookahead, wall_inflate, v_max, w_max,
                lat_acc_max, w_foresight, runtime_room,
                w_centre, pair_steps, escape_tolerance, escape_unstick,
                sbs_stagger,
                formation_separation, stop_when_blocked,
                brake_latency, backup_retrace, partner_hold, lane_band,
                exact_clearance, object_reach_min, yield_wait, speed_v0,
                objects_starved,
                boost_above_preferred, station_gain, speed_split, band_target,
                heading_cap, d_max, d_stop, braking_authority),
        Node(package='rviz2', executable='rviz2', name='rviz2',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(share, 'config',
                                           'joint_planning.rviz')]),
    ]


def generate_launch_description():
    share = get_package_share_directory('sbs_bringup')
    arguments = [
        DeclareLaunchArgument('direction', default_value='cw',
                              description="'cw' or 'ccw'"),
        DeclareLaunchArgument('concept', default_value='social',
                              description='independent | anticipated | social '
                                          '| gamma | optimistic'),
        DeclareLaunchArgument('gamma', default_value='0.5'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('publish_matrix', default_value='false',
                              description='record cost_matrix for offline IRL'),
        DeclareLaunchArgument('profile', default_value='false',
                              description='log a per-phase tick breakdown'),
        DeclareLaunchArgument(
            'objects', default_value='default',
            description="static objects injected into the laser's world: "
                        "'default' for the environment's objects.yaml, "
                        "'none' for an obstacle-free run with the laser "
                        'still running, or a path to another objects file'),
        DeclareLaunchArgument(
            'w_max', default_value='0.80',
            description='top yaw rate [rad/s], enforced by the limiter'),
        DeclareLaunchArgument(
            'lat_acc_max', default_value='1.50',
            description='comfort bound on |v*w| [m/s^2], enforced by the '
                        'limiter; the passenger-facing limit'),
        DeclareLaunchArgument(
            'v_max', default_value='1.24',
            description='top speed [m/s]: the WHILL governor, where the '
                        "drivers' speed peaks. 1.05 was set when it removed "
                        'the last wall contact; on the current planner 1.24 '
                        'gives zero contact in 48 runs and moves the speed '
                        'peak from 1.04 to 1.24 (speed-peak-was-the-cap)'),
        DeclareLaunchArgument(
            'w_foresight', default_value='0.5',
            description='weight on preferring plans that stay clear over the '
                        'whole horizon; a cost, not a constraint. Must stay '
                        'below the progress reward, since a stationary plan '
                        'has no penalty at all. 0 disables'),
        DeclareLaunchArgument(
            'wall_inflate', default_value='0.10',
            description='metres the planner thickens the map by before '
                        'reasoning about it; quantised to the grid. Off by '
                        'default: measured at 0.05 it made wall contact '
                        'worse (38 -> 196 mm) because the admissibility '
                        'fallback returns a penetrating candidate when the '
                        'mask empties. Metrics and detector are never '
                        'inflated'),
        DeclareLaunchArgument(
            'inflate', default_value='0.05',
            description='thicken the surveyed object cells by this radius; '
                        'the survey holds only the surface the LiDAR saw, so '
                        'a cone is a thin arc with a hollow interior'),
        DeclareLaunchArgument(
            'diagnose', default_value='false',
            description='per-tick line separating constraint-limited from '
                        'cost-chosen speed'),
        DeclareLaunchArgument(
            'object_horizon', default_value='5.0',
            description='how far ahead an obstacle moves the lateral target, '
                        'in SECONDS -- the manoeuvre needs a fixed time, not '
                        'a fixed distance, since shedding one stagger length '
                        'takes about 4.1 s whatever the speed. A fixed 8 m '
                        'was 8.9 s at cruise and far more at cornering '
                        'speed, and the pair committed to file far too '
                        'early'),
        DeclareLaunchArgument(
            'braking_authority', default_value='true',
            description='let the planner use the braking the corpus says it '
                        'has, in the two places that need it together: the '
                        'speed grid reaches d_stop of deceleration, and '
                        'foresight asks for the candidate stopping distance '
                        'rather than a fixed 0.35 m. From 0.93 m/s the '
                        'symmetric grid could only command 0.48 m/s, which '
                        'is 1.10 m/s2 against the 1.50 the drivers used'),
        DeclareLaunchArgument(
            'd_max', default_value='0.60',
            description='comfortable deceleration, and what the speed caps '
                        'assume. The recorded drivers were symmetric in '
                        'ordinary driving -- median 0.20 and p95 0.84 both '
                        'accelerating and braking -- so this matches a_max'),
        DeclareLaunchArgument(
            'd_stop', default_value='1.50',
            description='deceleration available when the alternative is a '
                        'collision, above the drivers p99 of 1.18 and half '
                        'the 2.93 they were observed to reach. Used to ask '
                        'whether a stop is possible, never to raise a speed '
                        'cap'),
        DeclareLaunchArgument(
            'heading_cap', default_value='false',
            description='cap the speed on the clearance along the path the '
                        'vehicle is already committed to, not only along its '
                        'lane. Two of nine real wall contacts in a 48-run '
                        'campaign were a vehicle off its lane heading at a '
                        'wall, where the lane-based cap reads v_max. Off: '
                        'measured harmful, progress 0.89 -> 0.28 and contact '
                        '0 -> 275 mm, because the projected arc meets a wall '
                        'wherever the corridor curves'),
        DeclareLaunchArgument(
            'band_target', default_value='true',
            description='snap the lateral target to free space measured '
                        'ahead. Without it the target is built from obstacle '
                        'offsets alone and can land between two free bands -- '
                        'measured, it asked for -0.68 m where the only band '
                        'open across the next 0.8 m was [+0.60,+1.15], and '
                        'the vehicle pinned there for 95 s'),
        DeclareLaunchArgument(
            'speed_split', default_value='0.326',
            description='fraction of the geometric speed difference the outer '
                        'vehicle gets through a bend. 1.0 is rigid formation; '
                        '0.326 is what the corpus fits, and measured it leaves '
                        '0.126 m/s systematically pulling the pair apart on '
                        'every corner'),
        DeclareLaunchArgument(
            'station_gain', default_value='0.0',
            description='along-track error [m] at which station keeping '
                        'reaches full authority. 0 keeps the old behaviour, '
                        'which borrowed file_stagger (1.60 m) -- the target '
                        'of a file manoeuvre -- so the correction saturated '
                        'at twice the offset ever observed and ran at a third '
                        'of its authority'),
        DeclareLaunchArgument(
            'boost_above_preferred', default_value='false',
            description='let station keeping ask the vehicle behind for MORE '
                        'than its fitted speed, instead of only slowing the '
                        'one ahead. Off by default: at 1.54x the catching '
                        'vehicle ran 1.26 m/s where no driver exceeded 1.12, '
                        'and could no longer stop -- 1.84 m of stopping '
                        'distance against the 1.41 m it had, two seconds '
                        'before a 78 mm wall contact. Measured, the clamp '
                        'also leaves the boost never applied at all: 0%% of '
                        'ticks, so catch-up rests entirely on the leader'),
        DeclareLaunchArgument(
            'escape_unstick', default_value='false',
            description='let a recovery candidate be chosen on the floor '
                        'rather than on a 1 mm improvement, where the '
                        'standing pose is within one map cell of a surface '
                        'and the comparison is below the resolution'),
        DeclareLaunchArgument(
            'formation_separation', default_value='true',
            description='separation target from the measured per-formation '
                        'constants (1.47 m side by side, 1.75 m filing) '
                        'rather than the pooled object-distance law, which '
                        'is a mixture of the two'),
        DeclareLaunchArgument(
            'sbs_stagger', default_value='0.0',
            description='along-track offset asked for while side by side. '
                        'Zero holds the pair level and is what every '
                        'result before 2026-09-24 was measured under; the '
                        'drivers held 0.59-0.63 m in a clear aisle and '
                        '0.97-0.99 m at an obstacle, against our 0.24-0.43.'),
        DeclareLaunchArgument(
            'speed_v0', default_value='1.24',
            description='straight-line value of the speed law v*(kappa) '
                        '[m/s]; the fit to median speeds gives 1.094, biased '
                        'low because the WHILL governor capped them at 1.24, '
                        'so the default is the governor.'),
        DeclareLaunchArgument(
            'objects_starved', default_value='false',
            description='reproduce a fixed bug for observation: the '
                        'objects subscription shares the timer\'s callback '
                        'group, so the planner gets cones every ~10 s and '
                        'drives into them. '
                        'Never set it for an evaluation.'),
        DeclareLaunchArgument(
            'yield_wait', default_value='false',
            description='filing, the vehicle whose lane is blocked stops short '
                        'of the obstacle until its partner is file_stagger '
                        'ahead, instead of creeping on at the speed floor.'),
        DeclareLaunchArgument(
            'object_reach_min', default_value='1.6',
            description='shortest distance (m) the lateral target looks ahead '
                        'for objects; 1.6 (file_stagger) is the old rule, '
                        'which shrank the warning as the pair slowed and was '
                        'too late to file at the layout_07 ccw cones.'),
        DeclareLaunchArgument(
            'exact_clearance', default_value='true',
            description='body clearance from points around the whole outline; '
                        'false checks only the cells nearest base_link, which '
                        'missed corners (a body 50 mm inside a wall read as '
                        '178 mm clear). Every arm before 2026-10-22 ran false.'),
        DeclareLaunchArgument(
            'lane_band', default_value='false',
            description='keep both vehicles within the band between their '
                        'two lanes: drop plans that go more than 0.70 m '
                        'outside it (the drivers p99 near cones), or further '
                        'out than the vehicle already is. false is arm hold.'),
        DeclareLaunchArgument(
            'partner_hold', default_value='true',
            description='the vehicle ahead along the route stops while its '
                        'partner is more than 3.0 m away, until it is within '
                        '2.2 m (the drivers p99 and p90 among cones), at most '
                        '60 s. false is arm backup.'),
        DeclareLaunchArgument(
            'backup_retrace', default_value='true',
            description='when stuck, reverse 0.8 m along the path this '
                        'vehicle drove, under the same brake test, then '
                        'rejoin below 0.4 m/s for 4 s. false is arm '
                        'stop_latency: zero contact, 5 of 24 runs ended stuck.'),
        DeclareLaunchArgument(
            'brake_latency', default_value='0.2',
            description='seconds between the state a command is computed '
                        'from and the moment it acts; the stop test runs the '
                        'command for one tick plus this before braking. 0.0 '
                        'stopped 0.17 s late and touched a wall by 57 mm.'),
        DeclareLaunchArgument(
            'stop_when_blocked', default_value='true',
            description='publish a stop instead of a chosen candidate that is '
                        'not clear of walls and objects over the executed '
                        'prefix, and wait for a clear plan. false is the '
                        '48-run baseline, which executed the least-bad '
                        'candidate and put the body 42-103 mm into a wall.'),
        DeclareLaunchArgument(
            'escape_tolerance', default_value='0.0',
            description='how far below its standing clearance a recovery may '
                        'dip, and only when that clearance is already inside '
                        'the map resolution. Zero is the control arm: it '
                        'refused every escape candidate 225 times over 32 '
                        'runs, at a median standing clearance of 0.000 m. '
                        'Off: once the layouts are reachable and '
                        'band_target is on, the arms are identical on stalls '
                        'and progress and differ only in contact -- 1 over '
                        '25 mm in 24 runs against 5, worst 26 mm against '
                        '130 mm'),
        DeclareLaunchArgument(
            'pair_steps', default_value='10',
            description='steps of the 30-step horizon over which the two '
                        'bodies are checked against each other. 10 is 1.0 s '
                        'of a 3 s rollout, so the later two thirds are '
                        'unchecked'),
        DeclareLaunchArgument(
            'runtime_room', default_value='true',
            description='measure the corridor width from the map and the '
                        'objects every tick, and let it withdraw the side-by-side '
                        'option and cap the separation target. The offline '
                        'side-by-side flag cannot see a cone that has moved since '
                        'the profile was derived; false is the control arm'),
        DeclareLaunchArgument(
            'w_centre', default_value='0.0',
            description='weight on not crowding one edge of the corridor '
                        'while the other is free. Off: measured over 16 '
                        'layout runs it introduced a stall in each direction '
                        'and cost 0.05 of progress, while runtime_room '
                        'delivered the wall-contact improvement without it'),
        DeclareLaunchArgument(
            'side_by_side_lookahead', default_value='4.1',
            description='seconds of route the side-by-side flag is read ahead; '
                        'the time the file manoeuvre needs, not a distance. '
                        'Negative ignores the flag'),
        DeclareLaunchArgument(
            'yield_when_blocked', default_value='false',
            description='impose a pass order: the vehicle whose lane an '
                        'obstacle blocks takes the position behind. Off by '
                        'default, because the drivers did not do this -- over '
                        '148 recorded passes the blocked vehicle led 51% of '
                        'the time, and 78% of passes simply kept whatever '
                        'order the pair already had. They did not need a rule '
                        'because they were strung out to 1.62 m by the time '
                        'they arrived. True is the alternative mode: it buys '
                        '12 points of conversational geometry at the cost of '
                        'an invented coordination behaviour and a third of '
                        'the obstacle clearance'),
        DeclareLaunchArgument('metrics', default_value='true',
                              description='log per-tick formation metrics'),
        DeclareLaunchArgument(
            'metrics_file', default_value='',
            description='CSV path for the metrics log; empty picks a '
                        'timestamped file under /tmp'),
        DeclareLaunchArgument(
            'detect_objects', default_value='true',
            description='simulate the laser and detect objects from it. The '
                        'laser raycasts the injected objects while the '
                        'detector subtracts the map, so cones are found '
                        'rather than read out of a file.'),
    ]
    # Calibrated defaults. The discriminative terms come from MaxEnt IRL over
    # the recorded choice sets (stage 15); the structural ones are set by what
    # the closed loop needs, because one-step ranking cannot see their job --
    # IRL puts rel_dist_end at 0.13 and speed at 0.43, and at those values the
    # pair drifts 0.3 m too close and runs 0.3 m/s too fast.
    calibrated = {
        # Polar formation, default from 2026-10-01. The pair interaction is
        # expressed by two terms -- how far apart (`rel_dist_end`) and at what
        # bearing (`rel_angle`) -- which together fix the geometry completely:
        # lateral = d sin R_a and along-track = d cos R_a are then implied,
        # not free. The three terms below were specifying a third and fourth
        # constraint on a two-degree-of-freedom geometry.
        #
        # Measured over 72 runs against the five-term set: fidelity to the
        # drivers 25.21 -> 23.49, the largest single gain recorded, with
        # cone contact 18.2 -> 6.6 mm. The cost is formation: `conv` falls
        # 51.7 -> 32.4 and filing rises 20.4 -> 28.6, both far outside noise,
        # and stalls rise 0.04 -> 0.29. More human-like and simpler, but it
        # holds the conversational formation less. The terms are kept in the
        # code at zero rather than deleted, so restoring any of them is a
        # number rather than a patch.
        'rel_range_rate': '0.00',     # IRL
        'rel_yaw_rate': '0.00',   # IRL
        'motion_accel': '1.05',          # IRL
        'motion_speed': '3.00',          # structural: sets the cruise speed
        'path_dist': '1.91',      # IRL, and structural over a lap
        # 2.00 from the weight sweep of 2026-09-30, at target 0.40. Fidelity
        # is flat across 0.00-4.00 -- 22.09 to 23.05, inside the run-to-run
        # spread -- so it was not chosen on that. What replicates is the count
        # of quantities inside the drivers' own spread: 2 of 12 at weights
        # 0.00 and 1.00, and 4 of 12 at 2.00, 3.00 and a 3.00 repeat, falling
        # to 3 of 12 at 4.00. Contact decides between the three that tie:
        # 6 mm here, 17 and 9 mm at 3.00, and 55 mm at 4.00.
        #
        # The target matters more than the weight. At target 0.00 with this
        # term at full strength the pair is the worst arm measured (+4.26
        # fidelity, 28 mm contact): forcing the vehicles exactly level is
        # considerably worse than having no term at all.
        # Zero from 2026-10-01: redundant with `rel_angle`, since
        # along-track = separation * cos(R_a) once the bearing is specified.
        'rel_along_track': '0.00',        # structural: makes the formation exist
        'rel_yaw': '0.00',        # available, not yet calibrated
        # These four are the configuration every good run used, and they were
        # being typed on the command line while the defaults here stayed at
        # the pair-relative baseline. That cost a full set of measurements:
        # runs launched without the overrides had cross_track and lane_yaw at
        # zero -- the two terms that hold the route and forbid spinning -- and
        # the vehicles duly ignored the centreline and spun. Defaults are what
        # gets run, so the defaults are now the calibrated configuration.
        'path_cross_track': '6.00',    # signed lateral position: forbids swapping
        # Heading vs lane tangent. Raised from 1.00 after the stalls were
        # traced to it: cross-track error follows heading error at r = +0.71,
        # and at 1.00 the heading tail reached a p99 of 95 degrees -- the
        # vehicle pointing sideways -- which becomes metre-scale position
        # error and walks the body into a wall. 3.00 reproduces the drivers'
        # own heading distribution (median 4.5 vs their 5.0, p90 12.0 vs
        # 13.2, beyond 20 degrees 2.2% vs 2.7%) and improves cross-track p90,
        # progress and conversational geometry with it. 6.00 was also tried:
        # it holds heading *tighter* than the humans and cross-track p90
        # degrades 0.31 -> 0.46, heading and offset fighting on a curve.
        # Dropped to 0.00 on 2026-09-30 and restored on 2026-10-01. The
        # ablation that justified dropping it ran only on the corridor
        # conditions and measured fidelity, contact, stalls and side-swaps --
        # none of which can see a heading. Measured afterwards on the
        # generated layouts, where the pair has to manoeuvre, the cost of
        # removing it is a vehicle rotating off the lane: p99 heading error
        # 19.6 -> 78.6 deg, a 179.9 deg maximum, and 0.88% of all ticks beyond
        # 90 deg, against 0.00% with the term present.
        #
        # `rel_angle` is not a substitute even in its signed form. It
        # constrains the vehicle against its *partner*, and a vehicle can
        # satisfy it by turning instead of translating -- rotating to put the
        # partner at the right bearing is cheaper over a 3 s horizon than
        # driving to the right place. At gain 15 that becomes the preferred
        # solution and the pair spins 21.9% of the time. Only a term tying the
        # heading to the *route* forbids it.
        'path_lane_yaw': '3.00',       # heading vs lane tangent: forbids spinning
        'path_progress': '0.95',       # IRL fitted 1.90; halved alongside cross_track
        # 0.53 with sigma = 0.133 is an effective gain of 4.0, the strength
        # every polar arm that worked actually ran at. The signed form is
        # selected automatically whenever this weight is non-zero.
        'rel_angle': '0.53',             # |along|/separation: pull toward abreast
        'rel_dist_end': '2.00',   # superseded by cross_track + along_track
    }
    arguments += [DeclareLaunchArgument(f'w_{n}', default_value=v)
                  for n, v in calibrated.items()]
    return LaunchDescription(arguments + [
        SetEnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE',
            os.path.join(share, 'config', 'fastdds_large_messages.xml')),
        # One BLAS thread per process. numpy's bundled OpenBLAS otherwise takes
        # every core for each operation, and with two planners and two laser
        # raycasters running at once the oversubscription costs more than the
        # parallelism wins on arrays this small: median tick time halves, from
        # 180 ms to 89 ms, and ticks over the 100 ms budget fall from 91% to
        # 34%.
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '1'),
        SetEnvironmentVariable('OMP_NUM_THREADS', '1'),
        SetEnvironmentVariable('MKL_NUM_THREADS', '1'),
        SetEnvironmentVariable('NUMEXPR_NUM_THREADS', '1'),
        OpaqueFunction(function=launch_setup),
    ])
