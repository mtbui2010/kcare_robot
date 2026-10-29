import math
import threading
import time

from robot_agent.skill_configs import VLA_CLIENTS, _ConfigProxy
from robot_agent.connect import init_detect_client
from robot_agent.utils import exception_handler
import numpy as np
from robot_agent.skills import log_data
from kcare_robot.skills.head import get_robot_mode

# vla_cfg = VLA_CLIENTS['openpi']
# vla_client = init_detect_client(vla_cfg['url'])


mean_action = lambda actions, weights: np.divide(np.sum(np.multiply(actions, weights), axis=-1), np.sum(weights, axis=-1))

@exception_handler
def vla(node, **kwargs):
    control_func = vla_cfg['control']
    obs_funcs = vla_cfg['obs']

    chunk_size=3
    stack= np.zeros((50,8,chunk_size), 'float32')
    while True:
        obs = {k: func(node) for k, func in obs_funcs.items()}
        obs['prompt'] = kwargs['inputs']
        actions = vla_client.send({'obs': obs})['actions']
        
        stack = np.roll(stack, shift=-1, axis=0)            # remove first row 
        stack[-1, ...] = 0                                  # set last row to 0s
        stack = np.roll(stack, shift=-1, axis=-1)           # roll stacks
        stack[..., -1] = actions                            # update last actions
        
        action = mean_action(stack[0,...], (stack[0,...]!=0)+1e-10)                # mean action of fist row
        control_func(node, obs['state'], action)

        # for i in range(vla_cfg['num_exec']):
        #     action = actions[i, :]
        #     control_func(node, obs['state'], action)
        

    aa = 1    
    
    


# ── vla_drawer_open: SmolVLA through vla-serve (2026-10-26) ─────────────────────────────────────────────────
# Port of kcare_vla_open_drawer_20261026/vla_drawer_node_real.py (the README there: model, prompts, risks) into
# the skill, so no separate ROS node is needed: `vla_drawer_open::drawer` drives to the ENV place, puts the robot
# in the pose the policy was trained from, and runs the policy served by vla-serve:
#     docker run --gpus all -p 11450:11450 -v /path/to/ckpt:/ckpt \
#       mtbui2010/vlaserve:latest serve --host 0.0.0.0 /ckpt/<run>/pretrained_model
# Actions (8-D absolute joint targets: joint1..7 [rad] + gripper width [m]) go to the forward controllers; the
# trajectory controllers are switched back afterwards — also on error and on the dashboard's Cancel (watched here:
# a forward controller is not an action, CustomNode.cancel_all cannot stop it).

VLA_CONFIGS = _ConfigProxy('VLA_CONFIGS', {})
DRAWER_DEFAULTS = {
    'url': '',                  # vla-serve; '' = the url of connection `connection` (Connections panel)
    'connection': 'vlaserve',   # an 'http' connection whose url is the vla-serve server (http://<gpu-host>:11450)
    'model': None,              # served model id (None = the first one loaded)
    'execute': False,           # False = dry run: read, ask the policy, check, log — send nothing (README step 1)
    'fps': 30,                  # the data is 30 Hz (training speed); 15 = half speed. The first real run at 15 reached
                                # ~60 % of the depth and closed on air (2026-10-26)
    'seconds': 15,              # TRAINED-motion seconds (the sim eval timeout); the run lasts seconds * 30 / fps — 15 s
                                # at 30 fps, 30 s at 15 (a fixed 15 s at half speed cut the first real run before the pull)
    'speed_limit': 1.5,         # rad/s clamp on the commanded joint speed (training data p99 1.3 at 30 fps)
    'max_jump': 0.6,            # stop when a target is this far from the arm [rad] (training data p99 0.05; 0.35
                                # tripped at 30 fps — the real arm lags the commands, 0.37 rad after 2.5 s, 2026-10-26)
    'ready_tol_deg': 5,         # arm vs approach_drawer
    'head_tilt': -41,           # head ry of the trained pose (pan = the mode's side)
    'prompt': 'open drawer',    # default prompt (the middle drawer); a place's ENV `vla_prompt` wins
    'client_path': '/remote_dir/kcare_vla_open_drawer_20261026/vlaserve_client',   # when vlaserve is not installed
    'head_topic': '/femto/color/image_raw',
    'wrist_topic': '/hand/camera/color/image_raw/compressed',
    'arm_cmd_topic': '/arm/xarm7_forward_controller/commands',
    'tool_cmd_topic': '/body/tool_forward_controller/commands',
}
_ARM = [f'joint{i}' for i in range(1, 8)]
_GRIP = 'virtual_gripper_joint'
_IMG_WH = (848, 480)                       # both cameras, as in the training data
_HEAD_PAN = {'left': -70.0, 'right': 70.0}
# ARM_CONFIGS approach_drawer (kcare_bucheon) when a site has none — the pose the policy was trained from [deg]
_READY_ARM_DEG = {'left': [3.834, 82.741, 17.514, 81.917, -81.436, 107.124, 2.998],
                  'right': [-6.212, 81.905, -15.78, 80.381, -277.149, 105.365, -3.079]}


def _drawer_cfg(over: dict) -> dict:
    cfg = dict(DRAWER_DEFAULTS)
    try:
        cfg.update(dict(VLA_CONFIGS.get('drawer_open') or {}))
    except Exception:
        pass
    for k in list(over):
        if k in cfg:
            cfg[k] = over.pop(k)
    return cfg


def _server_url(cfg) -> str:
    """vla-serve's url: VLA_CONFIGS / the call's `url`, else the `url` of connection cfg['connection']."""
    if str(cfg.get('url') or '').strip():
        return str(cfg['url']).strip().rstrip('/')
    try:
        from robot_agent.state import current
        conn = current().dm.saved_connect(cfg.get('connection') or 'vlaserve') or {}
        return str((conn.get('config') or {}).get('url') or '').strip().rstrip('/')
    except Exception:
        return ''


def _truthy(v) -> bool:
    return v.strip().lower() in ('1', 'true', 'yes', 'on') if isinstance(v, str) else bool(v)


def _vlaserve(client_path):
    """vlaserve.client (installed, or the copy next to the checkpoint — appended, so the robot's own httpx wins)."""
    try:
        from vlaserve.client import ChunkExecutor, PolicyClient
    except ImportError:
        import os, sys
        for p in (client_path, os.path.join(client_path, 'vlaclient_deps')):
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
        from vlaserve.client import ChunkExecutor, PolicyClient
    return ChunkExecutor, PolicyClient


class _DrawerIO:
    """Its own rclpy node + executor thread (the backend's node is not touched): head / wrist images, joint
    states, the forward-controller command topics and the controller switch services."""

    def __init__(self, cfg):
        import rclpy
        from controller_manager_msgs.srv import SwitchController
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from sensor_msgs.msg import CompressedImage, Image, JointState
        from std_msgs.msg import Float64MultiArray
        self.SwitchController, self.Float64MultiArray = SwitchController, Float64MultiArray
        self.node = rclpy.create_node('kcare_vla_drawer')
        cg = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.head = self.wrist = None
        self.q = {}
        n = self.node
        n.create_subscription(Image, cfg['head_topic'], self._on_head, 1, callback_group=cg)
        n.create_subscription(CompressedImage, cfg['wrist_topic'], self._on_wrist, 1, callback_group=cg)
        n.create_subscription(JointState, '/joint_states', self._on_js, 10, callback_group=cg)
        self.arm_pub = n.create_publisher(Float64MultiArray, cfg['arm_cmd_topic'], 10)
        self.tool_pub = n.create_publisher(Float64MultiArray, cfg['tool_cmd_topic'], 10)
        self.sw = {w: n.create_client(SwitchController, f'/{w}/controller_manager/switch_controller', callback_group=cg)
                   for w in ('arm', 'body')}
        self.ex = MultiThreadedExecutor(num_threads=4)
        self.ex.add_node(n)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    def _on_head(self, m):
        import cv2
        ch = 4 if m.encoding in ('rgba8', 'bgra8') else 3
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, ch)[:, :, :3]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if m.encoding.startswith('bgr') else im
        im = cv2.resize(im, _IMG_WH, interpolation=cv2.INTER_AREA)
        with self.lock:
            self.head = im

    def _on_wrist(self, m):
        import cv2
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if im.shape[1::-1] != _IMG_WH:
            im = cv2.resize(im, _IMG_WH, interpolation=cv2.INTER_AREA)
        with self.lock:
            self.wrist = im

    def _on_js(self, m):
        with self.lock:
            self.q.update(zip(m.name, m.position))

    def ready(self, wait=5.0):
        t = time.time()
        while time.time() - t < wait:
            with self.lock:
                if self.head is not None and self.wrist is not None and _GRIP in self.q and all(j in self.q for j in _ARM):
                    return True
            time.sleep(0.1)
        return False

    def state(self):
        with self.lock:
            return np.array([self.q[j] for j in _ARM] + [self.q[_GRIP]], np.float64)

    def observation(self):
        with self.lock:
            return {'images': {'head': self.head.copy(), 'wrist': self.wrist.copy()},
                    'state': np.array([self.q[j] for j in _ARM] + [self.q[_GRIP]], np.float32)}

    def switch(self, which, activate, deactivate):
        cli = self.sw[which]
        if not cli.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f'/{which}/controller_manager/switch_controller not available')
        req = self.SwitchController.Request()
        req.activate_controllers, req.deactivate_controllers = activate, deactivate
        req.strictness = self.SwitchController.Request.STRICT
        req.activate_asap = True
        ev, box = threading.Event(), {}
        cli.call_async(req).add_done_callback(lambda f: (box.update(r=f.result()), ev.set()))
        if not ev.wait(5.0) or not box.get('r') or not box['r'].ok:
            raise RuntimeError(f'switch {which}: activate {activate} / deactivate {deactivate} failed')

    def to_forward(self):
        self.switch('arm', ['xarm7_forward_controller'], ['xarm7_traj_controller'])
        self.switch('body', ['tool_forward_controller'], ['tool_controller'])

    def to_trajectory(self) -> str:
        errs = []
        for which, act, deact in (('arm', ['xarm7_traj_controller'], ['xarm7_forward_controller']),
                                  ('body', ['tool_controller'], ['tool_forward_controller'])):
            try:
                self.switch(which, act, deact)
            except Exception as e:
                errs.append(str(e))
        return '; '.join(errs)


_IO = {}


def _drawer_io(cfg) -> _DrawerIO:
    key = (cfg['head_topic'], cfg['wrist_topic'], cfg['arm_cmd_topic'], cfg['tool_cmd_topic'])
    if key not in _IO:
        _IO[key] = _DrawerIO(cfg)
    return _IO[key]


def _not_ready(io, mode, cfg) -> list:
    """Why the robot is not in the trained pose ([] = it is): arm vs ARM_CONFIGS approach_drawer, head, gripper."""
    from robot_agent.skill_configs import ARM_CONFIGS
    q = io.state()
    pose = (ARM_CONFIGS.get('approach_drawer') or {}).get(mode) or _READY_ARM_DEG[mode]
    want = np.radians(np.asarray(pose, float))
    d = np.degrees(np.abs(q[:7] - want)) % 360
    d = np.minimum(d, 360 - d)                       # joint5 of 'right' is given as -277 deg
    with io.lock:
        pan, tilt = math.degrees(io.q.get('head_joint1', 0.0)), math.degrees(io.q.get('head_joint2', 0.0))
    out = []
    if d.max() > float(cfg['ready_tol_deg']):
        out.append(f'arm {d.max():.1f} deg from approach_drawer (joint{int(d.argmax()) + 1})')
    if abs(pan - _HEAD_PAN[mode]) > 8 or abs(tilt - float(cfg['head_tilt'])) > 8:
        out.append(f'head at pan {pan:.0f} tilt {tilt:.0f} deg, expected {_HEAD_PAN[mode]:.0f} / {cfg["head_tilt"]:.0f}')
    if q[7] < 0.08:
        out.append(f'gripper at {q[7]:.3f} m, expected open (0.1)')
    return out


def _run_drawer_policy(io, cfg, prompt, mode, execute):
    """The policy loop (ChunkExecutor). Returns (ok, msg, info)."""
    from robot_agent.core.run_control import cancel_requested
    ChunkExecutor, PolicyClient = _vlaserve(cfg['client_path'])
    client = PolicyClient(cfg['url'], model=cfg['model'])
    if not client.model:
        import httpx
        client.model = httpx.get(f"{cfg['url'].rstrip('/')}/v1/models", timeout=10).json()['models'][0]['model']
    fps, max_jump = float(cfg['fps']), float(cfg['max_jump'])
    max_step = float(cfg['speed_limit']) / fps
    t0, switched, sent, worst = time.time(), False, [0], [0.0]
    stop, why = threading.Event(), []
    try:
        client.reset()
        cmd = io.state()                                 # last commanded target (starts at the pose)

        def send(action):
            tgt = np.asarray(action[:8], np.float64)
            # moving: a target far from the arm = the policy is off; dry run: the arm stands still, so the gap
            # grows by design — only reported
            jump = float(np.abs(tgt[:7] - io.state()[:7]).max())
            worst[0] = max(worst[0], jump)
            if execute and jump > max_jump:
                why.append(f'policy target {jump:.2f} rad from the arm (limit {max_jump})')
                stop.set()
                return
            cmd[:7] = cmd[:7] + np.clip(tgt[:7] - cmd[:7], -max_step, max_step)
            cmd[7] = float(np.clip(tgt[7], 0.0, 0.1))
            sent[0] += 1
            if execute:
                io.arm_pub.publish(io.Float64MultiArray(data=[float(v) for v in cmd[:7]]))
                io.tool_pub.publish(io.Float64MultiArray(data=[float(cmd[7])]))
            elif sent[0] % int(fps) == 1:
                log_data({'msg': 'vla_drawer_open DRY RUN would send ' + ' '.join(f'{v:+.3f}' for v in cmd)})

        def watch():                                     # the dashboard's Cancel
            while not stop.is_set():
                if cancel_requested():
                    why.append('cancelled'); stop.set()
                time.sleep(0.02)
        threading.Thread(target=watch, daemon=True).start()
        if execute:
            io.to_forward(); switched = True
        log_data({'msg': f'vla_drawer_open: policy {client.model} — "{prompt}" (mode {mode}, {fps:g} fps, '
                         f'{"EXECUTE" if execute else "dry run, nothing sent"})'})
        run_sec = float(cfg['seconds']) * 30.0 / fps    # `seconds` counts trained motion (30 Hz data)
        stats = ChunkExecutor(client, get_observation=io.observation, send_action=send,
                              fps=fps).run(task=prompt, seconds=run_sec, stop=stop)
        ok = not why
        msg = (f'policy ran {time.time() - t0:.1f} s, {sent[0]} actions{"" if execute else " (dry run, nothing sent)"}; '
               f'largest target-arm gap {worst[0]:.2f} rad' + (f'; stopped: {why[0]}' if why else '') +
               ('; check the drawer by eye' if execute and ok else ''))
        info = {'stats': stats.summary(), 'largest_gap_rad': round(worst[0], 3), 'actions': sent[0]}
    except Exception as e:
        ok, msg, info = False, f'vla_drawer_open: {type(e).__name__}: {e}', {}
    finally:
        stop.set()
        if switched:
            # Stop where the arm IS: the trajectory controller takes over the last forward COMMAND, not the arm's
            # position — the 30 fps run (2026-10-26) stopped with the command 0.37 rad ahead and the arm went on
            # 24 deg after the switch. So the arm's own position is sent as the last command first.
            time.sleep(0.05)
            try:
                here = io.state()
                for _ in range(5):
                    io.arm_pub.publish(io.Float64MultiArray(data=[float(v) for v in here[:7]]))
                    io.tool_pub.publish(io.Float64MultiArray(data=[float(here[7])]))
                    time.sleep(0.04)
            except Exception as e:
                log_data({'msg': f'vla_drawer_open: could not hold the arm before the switch ({e})'})
            err = io.to_trajectory()
            if err:
                msg += ('; RESTORE THE CONTROLLERS BY HAND (ros2 control switch_controllers -c /arm/controller_manager '
                        f'--activate xarm7_traj_controller --deactivate xarm7_forward_controller; same for /body): {err}')
                ok = False
        try:
            client.close()
        except Exception:
            pass
    return ok, msg, info


@exception_handler
def vla_drawer_open(node, **kwargs) -> dict:
    """Open a drawer with the SmolVLA policy served by vla-serve.

        vla_drawer_open::drawer                                  # the ENV place: drive there, ready pose, run
        vla_drawer_open::inputs='drawer', prompt='open the top drawer'
        vla_drawer_open::inputs='drawer', execute=True           # really move (default VLA_CONFIGS: dry run)
        vla_drawer_open::open the top drawer                     # not an ENV place: a prompt, from where it is

    The place's ENV entry: `default_mode` left / right (the side the arm opens to), optional `vla_prompt`
    (top / middle / bottom drawer: "open the top drawer" · "open drawer" · "open the bottom drawer", counted from
    the top). Server: the 'http' connection `vlaserve` (Connections panel; its url = vla-serve). Settings:
    VLA_CONFIGS.drawer_open (execute, fps, seconds, …; `url` overrides the connection); any of them per call. The robot must stand where it stood in training: the wrist camera ~0.445 m from the drawer front.
    Untested on the real robot (2026-10-26) — someone at the e-stop for --execute runs."""
    from kcare_robot.utils import get_env_specs, env_key
    from robot_agent.skill_configs import ENV
    inp = str(kwargs.pop('inputs', '') or '').strip()
    cfg = _drawer_cfg(kwargs)
    execute = _truthy(kwargs.pop('execute', cfg['execute']))
    stay_here = _truthy(kwargs.pop('stay_here', False))
    prompt = kwargs.pop('prompt', None)
    key = env_key(inp, ENV) if inp else None
    env = get_env_specs(inp, ENV) if key else {}
    if key is None and inp:
        prompt = prompt or inp                           # the old form: inputs = the prompt
    prompt = str(prompt or env.get('vla_prompt') or cfg['prompt'])
    mode = str(kwargs.pop('mode', None) or env.get('default_mode') or get_robot_mode(node) or 'left')

    cfg['url'] = _server_url(cfg)
    if not cfg['url']:
        if 'vla_drawer_open' in node.agents:
            # the sim (kcare-sim): its /drawer/open action (vla_drawer_node), no vla-serve connection there
            return node.agents['vla_drawer_open'].send({'mode': mode, 'prompt': prompt})
        return {'isdone': False, 'msg': f"vla_drawer_open: no vla-serve server — add an 'http' connection "
                f"\"{cfg['connection']}\" with url http://<gpu-host>:11450 (Connections panel), or set "
                f"VLA_CONFIGS.drawer_open.url; the server: docker run --gpus all -p 11450:11450 -v /path/to/ckpt:/ckpt "
                f"mtbui2010/vlaserve:latest serve --host 0.0.0.0 /ckpt/<run>/pretrained_model"}
    if mode not in _HEAD_PAN:
        return {'isdone': False, 'msg': f'vla_drawer_open: mode {mode!r} — the policy was trained for left / right '
                                        f'(set default_mode of "{key or inp}" in ENV)'}
    try:
        import httpx
        httpx.get(f"{cfg['url'].rstrip('/')}/health", timeout=3).raise_for_status()
    except Exception as e:
        return {'isdone': False, 'msg': f"vla_drawer_open: vla-serve at {cfg['url']} not reachable ({e})"}

    from robot_agent.utils import run_parallel_check
    from kcare_robot.skills.arm import movej
    from kcare_robot.skills.grip import grip
    from kcare_robot.skills.head import moveh
    from kcare_robot.skills.lift import lift
    from kcare_robot.skills.mobile import move
    if key and not stay_here:
        ret = move(node=node, inputs=inp, wait=True)
        if not ret.get('isdone'):
            return ret
    # the pose the policy was trained from (README §4): head to the side, tilted; lift lowest; approach_drawer; open
    ret = run_parallel_check(funcs=[
        lambda: moveh(node=node, inputs=mode, ry=float(cfg['head_tilt']), wait=True),
        lambda: lift(node=node, inputs='lowest', wait=True),
        lambda: grip(node=node, inputs='open', wait=True),
    ])
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='approach_drawer', mode=mode, wait=True)
    if not ret['isdone']:
        return ret

    io = _drawer_io(cfg)
    if not io.ready():
        return {'isdone': False, 'msg': f"vla_drawer_open: no head / wrist image or joint states "
                                        f"({cfg['head_topic']}, {cfg['wrist_topic']}, /joint_states)"}
    time.sleep(0.5)                                      # fresh joint states after the motion
    bad = _not_ready(io, mode, cfg)
    if bad:
        return {'isdone': False, 'msg': 'vla_drawer_open: not in the ready pose: ' + '; '.join(bad)}
    ok, msg, info = _run_drawer_policy(io, cfg, prompt, mode, execute)
    log_data({'msg': f'vla_drawer_open: {msg}'})
    return {'isdone': ok, 'msg': msg, 'prompt': prompt, 'mode': mode, 'executed': execute, **info}


def vla_pick_approach(node, **kwargs) -> dict:
    """PICK phase A with a VLA ("reach to view"): from the mode's start pose, arm + lift + a small base nudge
    bring the wrist camera to the object (sim: vla_pick_node, /pick/approach). The grasp itself is fine_move's.

    inputs: the object ('cup' -> prompt 'reach the cup'; a full prompt is passed through).
    mode:   'front' | 'left' | 'right' (default: the head's current side)."""
    obj = str(kwargs.get('inputs') or '').strip()
    prompt = obj if obj.split(' ')[0] in ('reach', 'go', 'look') else f'reach the {obj}'
    mode = kwargs.get('mode') or get_robot_mode(node) or 'right'
    ret = node.agents['vla_pick_approach'].send({'mode': mode, 'prompt': prompt})
    try:
        import json
        ret.update(json.loads(ret.get('msg') or '{}'))
    except Exception:
        pass
    return ret
