"""Runtime configs for kcare_robot — the robot's own numbers.

``ConfigManager`` resolves a config group in this order:

  1. ``configs/locations/<site>/skill_configs_override.json`` (per-site, and
     what the dashboard's Edit-then-Save writes)
  2. this module
  3. ``robot_agent.skill_config_defaults`` (now only neutral schema stubs)

An override replaces a whole group outright — there is no per-key merge — so
this module is what a site that has not overridden a group falls back to.
Values below are the kcare machine's, in METRES, lifted from the `kcare` site
override. They previously lived in robot_agent as millimetre-era defaults that
no running site ever reached.
"""

GRIP_CONFIGS = {'range': [0, 0.1]}

LIFT_CONFIGS = {'mobile_height': 0.43, 'range': [0.56, 1.1], 'home': {'front': 0.6, 'left': 0.8, 'right': 0.8}}

HEAD_CONFIGS = {   'ry_range': [-50, 10],
    'rz': {'front': 0, 'left': -70, 'right': 70},
    'ry': {'up': 10, 'straight': -40, 'down': -50}}

# Visual Q&A (`qa` in skills/hri.py). The vision model runs in Ollama on the dev PC
# (RTX 3090); `views` are HEAD_CONFIGS['ry'] names, photographed in order.
# Ollama on the 4xA6000 server, leaving the dev PC's 3090 to visionserve.
# 2026-09-30, 22 where-questions on dressroom / cabinet / dining photos:
# 32b-instruct 19/22, 2.4 s per answer (1.5 s server + network); 8b-instruct
# 18/22, 2.0 s; 30b-a3b-instruct 16/22 (mixes up left / right). Local 8b on
# the 3090 was 0.5 s. A LAN address instead of the Cloudflare hostname saves
# about a second.
# HRI skills (skills/hri.py: reply, ask, qa).
#   source: which mic by default — 'dashboard' (the browser the operator is at)
#           or 'robot' (the robot's own speaker + mic); a call's source= wins.
#   dashboard_stt: who turns the dashboard mic into text — 'whisper' (the
#           browser records, the 'stt' connection transcribes; good Korean) or
#           'browser' (the browser's own recogniser: Google in Chrome).
#   stt: the 'stt' connection id ('' = the first one).
#   stt_hint: words Whisper should expect, per language (its initial prompt).
HRI_CONFIGS = {   'source': 'dashboard',
    'dashboard_stt': 'whisper',
    'stt': '',
    'earcon': True,             # "your turn" / "got it" tones around each listen (robot speaker or browser)
    'earcon_volume': 0.5,       # 0..1
    'speech_rate': 1.5,         # how fast the dashboard speaks (1.0 = the browser's normal, 0.5..2.0); a call's rate= wins
    'speech_rate_en': 1.0,      # the same for English lines (English voices are faster at one rate)
    'stt_hint': {   'ko': '와인잔, 머그컵, 텀블러, 컵, 접시, 선반, 서랍, 옷장, 행거. 어디에 있어?',
                    'en': ''}}

# Emergency checks (skills/emergency.py, SIMULATED: nothing is dialled). check_fall / check_restroom ask, then
# listen in windows of reprompt_sec until timeout_sec is used up; no clear "괜찮아" -> call_119 (siren + notice).
# demo: True uses the demo_* times (5 s / 15 s) instead of 20 s / 60 s.
EMERGENCY_CONFIGS = {   'timeout_sec': 60,
    'reprompt_sec': 20,
    'demo': True,
    'demo_timeout_sec': 15,
    'demo_reprompt_sec': 5,
    'source': None,             # None = HRI_CONFIGS.source (kcare-3 has no robot speaker / mic wired)
    'energy_threshold': None,
    'detect_sec': 2,            # check_fall / check_restroom "look" this long after "확인하고 있습니다"
    'simulate': True,
    'alarm_volume': 0.6,
    'alarm_cycles': 3}

# vla_drawer_open (skills/vla.py): SmolVLA served by vla-serve (README: /remote_dir/kcare_vla_open_drawer_20261026).
#   docker run --gpus all -p 11450:11450 -v /path/to/ckpt:/ckpt mtbui2010/vlaserve:latest serve --host 0.0.0.0 /ckpt/<run>/pretrained_model
# execute False = dry run (reads, asks the policy, checks, sends nothing). Untested on the real robot (2026-10-26).
VLA_CONFIGS = {   'drawer_open': {   'url': '',          # '' = the url of the connection below (Connections panel)
                       'connection': 'vlaserve',
                       'model': None,
                       'execute': False,
                       'fps': 30,          # training speed (15 = half: the first real run reached ~60 % deep)
                       'seconds': 15,      # trained-motion seconds; the run lasts seconds * 30 / fps
                       'speed_limit': 1.5,
                       'max_jump': 0.6,    # 0.35 tripped at 30 fps: the real arm lags the commands
                       'ready_tol_deg': 5,
                       'head_tilt': -41,
                       'prompt': 'open drawer'}}

QA_CONFIGS = {   'url': 'https://ollama.aistations.org',
    'model': 'qwen3-vl:8b-instruct',          # 2026-10-09: 8b vs 32b on every LLM role (bench_llm): 8b as good, ~3x faster
    'num_ctx': 8192,
    'camera': 'head_rgb',
    'views': ['up', 'straight', 'down'],
    'idle_ask_sec': 60,         # qa: silence this long → "더 궁금하신 게 있으세요?"; 0 = never
    'idle_end_sec': 30}         # ... and this much more → "대답이 없으셔서 대화를 마칠게요." (ends)

ARM_CONFIGS = {   'calib_offset': [0, 0, -0.03],
    'wrist_cam_offset': [-0.02, -0.075, 0],
    'base_x': 0.1915,
    'range': {'x': [0.4415, 0.4915], 'y': [-1, 1], 'z': [-0.1, 1.5]},
    'tool_range': {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [0.05, 0.9]},
    # where along the heading the approach pose must be (base frame); outside it the base drives (mforward).
    # Standing objects are approached tilted, the pose 0.125 m ahead of the object, so a shorter band.
    # Sim sweep 2026-10-19 (kitchen counter, left mode). Code defaults if a site override leaves them out.
    'approach_x': {'standing': [0.15, 0.40], 'lying': [0.15, 0.70]},
    'place_x': [0.15, 0.35],
    'max_reach_y': 0.8,          # side modes: an object deeper than this into the table is refused (IK fails beyond)
    'wrist_angle': {'lying': 45, 'standing': 10},
    'approach_d': 0.12,
    'home': {   'front': [0, 15, 0, 15, -180, 0, 0],
                'left': [-90, 15, 0, 15, -180, 0, 0],
                'right': [90, 15, 0, 15, -180, 0, 0]},
    'ready': {   'front': [0, 15, 0, 15, -180, 90, 0],
                 'left': [-90, 15, 0, 15, -180, 90, 0],
                 'right': [90, 15, 0, 15, -180, 90, 0]},
    'fold': {   'front': [90, 0, 20, 12, -40, 97, -10],
                'left': [   82.75220873842167,
                            -26.259632150976163,
                            -32.94722746168656,
                            17.842784914127574,
                            -41.129295483552426,
                            104.7276115626965,
                            -14.829346433452967],
                'right': [   -76.25895963035481,
                             -35.633824315103126,
                             31.37629419786738,
                             21.483215977589133,
                             -313.3162407767658,
                             110.09322375955092,
                             23.492944499352024]},
    'unfold': {   'front': [0, -26.3, 18.8, 1.5, -160.5, 62.2, 0],
                  'right': [-11.8, -19.1, -25.7, 12.3, -324.7, 60.7, 32],
                  'left': [3.7, -20.6, 26.8, 8.3, -36.7, 56.2, -34.8]},
    'give': {   'front': [90, 0, 20, 12, -40, 97, -10],
                'left': [96.85, 18.01, -49.85, 14.47, -33.29, 53.71, -14.79],
                'right': [-78.91, 24.26, 45.87, 24.81, -318.36, 52.09, 20.36]},
    'approach_lying': {   'right': [89.6, -26.8, 0.3, 28.4, 0.9, 55, -179.9],
                          'left': [86, 23.3, -182.5, 37, -0.2, 60.4, -185.5],
                          # front (2026-10-09, sim IK): gripper down, tcp (0.50, 0, 0.73) m in base_footprint at lift 0.6
                          'front': [0.0, -4.91, 0.0, 48.86, -180.0, -53.77, 0.0]},
    'approach_standing': {   'right': [   36.430907682150696,
                                          33.89414098255503,
                                          -31.20457641194409,
                                          51.74252835934931,
                                          -295.6459403372958,
                                          99.96921684343455,
                                          13.196118429308688],
                             'left': [   -39.71862719505864,
                                         30.56703972462593,
                                         36.299706287476845,
                                         45.17457405723544,
                                         -58.420349669357414,
                                         99.99019918456483,
                                         -5.605114914535039],
                             # front (2026-10-09, sim IK): tcp (0.66, 0, 0.67), 15 deg down like left / right
                             'front': [0.0, 5.83, 0.0, 5.97, -180.0, 75.41, 0.0]},
    'approach_drawer': {   'right': [   -6.211779573691565,
                                        81.90505354547581,
                                        -15.779663096075398,
                                        80.38103343597086,
                                        -277.14880451766084,
                                        105.36503896698794,
                                        -3.078588880403007],
                           'left': [   3.834008869823475,
                                       82.74111651120957,
                                       17.51407523013154,
                                       81.91691758406412,
                                       -81.43637962500787,
                                       107.12427713112979,
                                       2.9979493191843094]},
    'pre_pick': {   'right': [   8.743469444279539,
                                 41.818468400995876,
                                 18.121398575656148,
                                 52.97783295754323,
                                 -294.22176393307814,
                                 68.28793762365318,
                                 16.83512027568642],
                    'left': [   14.24531232545081,
                                46.09995199165086,
                                -35.125928033326396,
                                47.90983942358294,
                                -51.51324573910115,
                                58.453530728345335,
                                -21.345445939388096]},
    # Pre-pick poses for objects on the floor / on high shelves (2026-10-09, from the sim URDF IK, self-collision
    # checked in MuJoCo). floor: gripper pointing down, tcp 0.30 m above the floor at lift 0.56 (lowest);
    # front tcp (0.60, 0, 0.30), left / right (0.25, -+0.55, 0.30). high: gripper horizontal, tcp 1.30 m at lift
    # 1.10 (highest); front tcp (0.70, 0, 1.30) facing forward, left / right (0.25, -+0.60, 1.30) facing out.
    'pre_pick_floor': {"front": [0.0, 64.64, 0.0, 79.37, -180.0, -14.72, 0.0], "left": [-33.54, 106.3, -70.37, 91.75, -253.32, 70.67, -37.49], "right": [33.54, 106.3, 70.37, 91.75, -106.68, 70.67, 37.49]},
    'pre_pick_high': {"front": [0.0, -13.39, 0.0, 6.35, -180.0, 70.27, 0.0], "left": [-44.63, 10.06, -40.33, 31.39, -174.03, 65.8, 4.06], "right": [45.28, 9.96, 39.7, 31.39, -185.93, 65.81, -3.93]},
    'j_arm_speed': 1.5,
    'j_arm_accel': 12,
    'l_arm_speed': 400,
    'l_arm_accel': 1200,
    'tool_radius': 0.05,
    'tool_length': 0.26,
    'wrist_tool_length': 0.17,
    'approach_wrist_angle': 45,
    'place_liftup': 0.13}

# strict_loc: a location name matching several ENV entries (e.g. `table` with
# table@kitchen and table@living room) fails the skill instead of taking the
# first one. ENV entries may list other names for the same spot in `aliases`.
MOBILE_CONFIGS = {'dforward': 0.2, 'dshift': -0.2, 'height': 0.405, 'max_shift': 0.6,
                  'shift_deadband': 0.03,      # base shifts under this are not driven (was 0.1, hard-coded)
                  'strict_loc': False}

FIND_CONFIGS = {   'detect_object': {   'model': 'rfdetr-gdino-sam-etri',
                         'min_size': 0,
                         'max_size': 50,
                         'box_threshold': 0.35,
                         'text_threshold': 0.3,
                         'target_distance': 0.9,
                         'distance_sigma': 0.01,
                         'near_point': 'center',
                         'weight': {'quality': 0.8, 'distance': 0.1, 'near': 0.1}},
    'detect_grasp': {   'model': 'rfdetr-gdino-sam-etri',
                        'min_size': 0,
                        'max_size': 50,
                        'box_threshold': 0.3,
                        'text_threshold': 0.15,
                        'target_distance': 0.3,
                        'distance_sigma': 0.01,
                        'near_point': 'center',
                        'weight': {'quality': 0.78, 'distance': 0.2, 'near': 0.02},
                        'gripper_min': 10,
                        'gripper_max': 500,
                        # tool-tip depth (_recognition_helpers._grasp_tip_depth): ratio of the gap to what is
                        # behind, within [min, max] mm; nothing seen behind -> no_back mm
                        'depth_ratio': 0.5,
                        'depth_min_mm': 15,
                        'depth_max_mm': 50,
                        'depth_no_back_mm': 40,
                        'depth_clearance_mm': 3},
    'detect_background': {   'use': True,
                             'stride': 20,
                             'avoid_model': 'grounding-dino',
                             'avoid_prompt': 'object.',
                             'avoid_conf': 0.15,
                             'avoid_pad': 20,
                             'avoid_max_area': 0.1,
                             'place_height_band': 0.04,
                             'place_margin_cells': 2,
                             'place_min_component_cells': 6,
                             'calib_tune': {'dx': 0.03, 'dy': 0.05}},
    'grasp_check_roi': [415, 365, 476, 390],
    'standing_obj_names': ['cup'],
    'lying_obj_names': [   'orange',
                           'apple',
                           'phone',
                           'fork',
                           'spoon',
                           'bread',
                           'banana',
                           'control'],
    'having_handle_obj_names': []}

ENV = {   'table@living room': {   'default_mode': 'left',
                             'placepose': {'dx': 0, 'dy': -0.1},
                             'height': 0.72,
                             'loc': {'x': 0.9, 'y': -1.67, 'rz': 0},
                             'dforward': 0},
    'wait@living room': {   'default_mode': 'front',
                            'height': 0.72,
                            'loc': {'x': 4.6, 'y': 0.48, 'rz': 0}},
    'drawer@living room': {   'default_mode': 'left',
                              'placepose': {'dx': 0, 'dy': 0},
                              'handle': 'gray handle',
                              'handle_height': 0.415,
                              'height': 0.565,
                              'loc': {'x': 0.763, 'y': -1.913, 'rz': -90},
                              'dshift': -0.25,
                              'dforward': -0.1},
    'drawer2@living room': {   'default_mode': 'left',
                               'placepose': {'dx': 0.1},
                               'handle': 'white handle',
                               'handle_height': 0.6,
                               'height': 0.72,
                               'loc': {'x': 0.763, 'y': -1.913, 'rz': -90},
                               'dshift': -0.25,
                               'dforward': -0.1},
    'drawer3@living room': {   'default_mode': 'right',
                               'placepose': {'dx': 0.1, 'dy': 0},
                               'handle': 'gray handle',
                               'handle_height': 0.415,
                               'height': 0.565,
                               'loc': {'x': 2.1, 'y': -0.25, 'rz': -90},
                               'dshift': -0.1,
                               'dforward': 0}}
