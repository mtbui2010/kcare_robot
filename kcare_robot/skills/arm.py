from robot_agent.utils import exception_handler, refine_inputs
from kcare_robot.utils import get_dtool_next_state
from robot_agent.utils import quaternion2deg, deg2quaternion
from kcare_robot.skills.head import get_robot_mode
from robot_agent.connect.helpers import update_dict, data_info
from robot_agent.skill_configs import ARM_CONFIGS
import numpy as np

@exception_handler
def arm_joints(node, **kwargs):
    ret =  node.agents['joint_states'].get()
    assert ret is not None, f'check joint_state connection'
    
    joints = ret['position'][3:10]
    return {'isdone': True, 'joints': [round(el*180./np.pi, 2) for el  in joints]}


@exception_handler
def arm_pose(node, **kwargs):
    ret =  node.agents['mobile_base_tool_pose'].get()
    assert ret is not None, f'check mobile_base_tool_pose connection'
    
    pos, ort = ret['pose'].position, ret['pose'].orientation
    rx, ry, rz= quaternion2deg(ort.x, ort.y,ort.z, ort.w)
    
    return {'isdone': True, 'pose': [round(el, 3) for el in (pos.x, pos.y, pos.z, rx, ry, rz)]}

def _qmul(a, b):
    """Hamilton product a * b of (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _qconj(q):
    """Inverse of a unit (x, y, z, w) quaternion."""
    return (-q[0], -q[1], -q[2], q[3])


@exception_handler
def movel(node, **kwargs):
    """Linear TCP move, sent RELATIVE to the current pose (``is_relative=True``):
    the goal carries the offset to move by — position (dx, dy, dz) and rotation
    (drx, dry, drz, degrees, as a quaternion) — not the target pose.

    Absolute x / y / z / rx / ry / rz are still accepted and turned into offsets
    from the current pose, so callers passing an approach pose keep working; an
    absolute and a relative value for one axis add up, as before.

    Rotation: the server applies the relative rotation about the TOOL axes
    (new = current * q_rel) — measured: from pre_pick, ``rz=80`` sent as a
    base-axis rotation ended at (89.9, -77.4, 90.5), which only tool-axis
    application explains (0.4° off). So:
      * absolute rx / ry / rz: q_rel = q_current^-1 * q_target;
      * drx / dry / drz: a turn about the BASE axes (like dx, dy, dz),
        q_rel = q_current^-1 * R(drx, dry, drz) * q_current.
    Quaternions throughout, so several angles at once and ry = ±90 (gimbal
    lock, e.g. ``movel::rx=90, ry=-90, rz=90``) are exact.
    """
    out = {}
    out['velocity_scale'] = kwargs.get('speed', 1.0)
    out['acceleration_scale'] = kwargs.get('acc', 0.35)

    dx, dy, dz  = kwargs.pop('dx', 0), kwargs.pop('dy', 0), kwargs.pop('dz', 0)
    drx, dry, drz  = kwargs.pop('drx', 0), kwargs.pop('dry', 0), kwargs.pop('drz', 0)
    q_rel = (0.0, 0.0, 0.0, 1.0)

    # Absolute positions → offsets from where the arm is now (only then is the
    # current pose needed).
    if any(k in kwargs for k in ('x', 'y', 'z')):
        x0, y0, z0 = arm_pose(node=node)['pose'][:3]
        dx += kwargs.pop('x', x0) - x0
        dy += kwargs.pop('y', y0) - y0
        dz += kwargs.pop('z', z0) - z0

    absolute_rot = any(k in kwargs for k in ('rx', 'ry', 'rz'))
    if absolute_rot or drx or dry or drz:
        # Exact current orientation (arm_pose rounds Euler degrees, which loses
        # it at gimbal lock).
        o = node.agents['mobile_base_tool_pose'].get()['pose'].orientation
        q_cur = (o.x, o.y, o.z, o.w)
        if absolute_rot:
            # An axis not given keeps its current angle; d* on top adds to it.
            rx0, ry0, rz0 = quaternion2deg(*q_cur)
            q_tgt = deg2quaternion(kwargs.pop('rx', rx0) + drx, kwargs.pop('ry', ry0) + dry,
                                   kwargs.pop('rz', rz0) + drz)
        else:
            q_tgt = _qmul(deg2quaternion(drx, dry, drz), q_cur)     # turn about the base axes
        q_rel = _qmul(_qconj(q_cur), q_tgt)

    qx, qy, qz, qw = q_rel

    # The message fields are float64: an untouched axis is the int 0 here
    # (the absolute version always added a float pose to it).
    out.update({k: float(v) for k, v in {'x': dx, 'y': dy, 'z': dz,
                                          'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw}.items()})
    out['velocity_scale'] = float(out['velocity_scale'])
    out['acceleration_scale'] = float(out['acceleration_scale'])

    out['base_frame'] = 'base_footprint'
    out['is_relative'] = True
    ret =  node.agents['arm_movel'].send(out)
    return ret

def movel_backup(node, **kwargs):
    out = {}
    out['velocity_scale'] = kwargs.get('speed', 1.0)
    out['acceleration_scale'] = kwargs.get('acc', 0.35)

    x0, y0, z0, rx0, ry0, rz0 = arm_pose(node=node)['pose']
    x, y, z  = kwargs.pop('x', x0), kwargs.pop('y', y0), kwargs.pop('z', z0)
    rx, ry, rz  = kwargs.pop('rx', rx0), kwargs.pop('ry', ry0), kwargs.pop('rz', rz0)
    dx, dy, dz  = kwargs.pop('dx', 0), kwargs.pop('dy', 0), kwargs.pop('dz', 0)
    drx, dry, drz  = kwargs.pop('drx', 0), kwargs.pop('dry', 0), kwargs.pop('drz', 0)

    x,y,z =  x+dx, y+dy, z+dz
    rx, ry, rz = rx+drx, ry+dry, rz+drz

    qx, qy, qz,qw = deg2quaternion(rx, ry, rz)
    
    out.update({'x': x, 'y': y, 'z': z, 
           'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw})
    

    out['base_frame'] = 'base_footprint'
    out['is_relative'] = False
    ret =  node.agents['arm_movel'].send(out)
    return ret
    

fix_angle = lambda angle: angle-360 if angle>=360 else angle+360 if angle<=-360 else angle
@exception_handler
def movej(**kwargs):
    last_state_only = kwargs.pop('last_state_only', False)
    inputs = {}
    inputs['velocity_scale'] = kwargs.get('speed', 1.0)
    inputs['acceleration_scale'] = kwargs.get('acc', 0.3)

    node = kwargs.pop('node', None)
    agents = node.agents
    # inputs = kwargs.pop('inputs')
    
    robot_mode = kwargs.get('mode', get_robot_mode(node=node))
    # robot_mode = "left"

    # inputs = {'relative': 'inputs' not in kwargs}
    # inputs = {}
    dangles = {f'dr{i}': kwargs.pop(f'dr{i}', 0.) for i in range(7)}
    if 'inputs' in kwargs:
        angles = kwargs['inputs']
        if isinstance(angles, str):
            angles_list = ARM_CONFIGS[angles][robot_mode]
            if isinstance(angles_list[0], (list, tuple)):
                if last_state_only:
                    angles_list = [angles_list[-1],]
                for angles in angles_list:
                    ret = movej(node=node, inputs= angles)
                    assert ret['isdone'], f'{ret}'
            else:
                angles = angles_list
        inputs['angles'] = [el for el in angles]
        inputs['angles'] = [a+b for a,b in zip(inputs['angles'], list(dangles.values()))]
    else:
        angles = dangles
        current_joints = arm_joints(node=node)
        assert current_joints is not None, 'arm_joints failed'
        
        current_angles = current_joints['joints']
        for i in range(7):
            if f'r{i}' in kwargs:
                angles[f'dr{i}'] = kwargs[f'r{i}'] - current_angles[i]

        # inputs['angles'] = list(angles.values())
        inputs['angles'] = [fix_angle(el0+el1) for el0, el1 in zip(current_angles, angles.values())]
        
        
    # inputs['relative'] = False
    #   
    # inputs['angles'] = [float(el) for el in inputs['angles']]
    # inputs['speed'] = ARM_CONFIGS['j_arm_speed'] * kwargs.get('speed', 1.0)
    # inputs['acc'] = ARM_CONFIGS['j_arm_accel'] * kwargs.get('acc', 1.0)
    # inputs['wait'] = kwargs.get('wait', True)

    # angles were kept in degrees throughout movej; convert to radians for the arm controller.
    inputs['target_joints'] = [float(el)*np.pi/180. for el in inputs['angles']]
    inputs.pop('angles', None)
    print(data_info(inputs))


    return agents['arm_movej'].send(inputs)


@exception_handler
def movet(node, **kwargs):
    inputs = {}
    inputs['velocity_scale'] = kwargs.get('speed', 1.0)
    inputs['acceleration_scale'] = kwargs.get('acc', 0.35)

    agents = node.agents
    
    # angle_list = ['rx', 'ry', 'rz']
    # out = {key: kwargs.get(key, 0.) if key not in angle_list else kwargs.get(key, 0.)*np.pi/180
    #          for key in ['dx', 'dy', 'dz', 'rx', 'ry', 'rz']}
    qx, qy, qz, qw = deg2quaternion(kwargs.pop('rx',0.), kwargs.pop('ry',0.), kwargs.pop('rz',0.))
    dx, dy, dz = float(kwargs.pop('dx', 0.)), float(kwargs.pop('dy', 0.)), float(kwargs.pop('dz', 0.))
    
    inputs.update({'dx': -dy, 'dy': dx,'dz': dz, 
                    'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw})

    return agents['arm_movet'].send(inputs)

def get_wrist_angle(node, **kwargs):
    ry = arm_pose(node=node)['pose'][4]
    return abs(90 + ry)


@exception_handler
def movelf(**kwargs):
    node = kwargs.pop('node', None)
    agents = node.agents
    kwargs['mode'] = kwargs.get('mode', get_robot_mode(node=node))
    
    pos_keys = ['x', 'y', 'z', 'dx', 'dy', 'dz']
    angle_keys = ['rx', 'ry', 'rz', 'drx', 'dry', 'drz']
    current_pose = agents['robot_pose'].get()['pose']
    rx, ry, rz = current_pose[3:]*180/np.pi
    
    movel_data = {k:v for k, v in kwargs.items() if k not  in angle_keys}
    movet_data = {k:v for k, v in kwargs.items() if k not  in pos_keys}
    for k in ['rx', 'ry', 'rz']:
        if k in movet_data:
            movet_data[f'd{k}'] = movet_data[k] - eval(k)
    dry, drz = movet_data.pop('drz', 0.), movet_data.pop('dry', 0.)
    movet_data['dry'], movet_data['drz'] = dry, drz
    
    if not movel(**movel_data, node=node)['isdone']:
        raise Exception('movel failed ...')
    
    if not movet(**movet_data, node=node)['isdone']:
        raise Exception('movet failed ...')
    
    return {'isdone': True}

def arm_exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            ret0 = func(*args, **kwargs)
        except Exception as e:
            ret0 = {'isdone': False, 'msg': f"'Exception in {func.__name__}': {e}"}
            # text2voice(f'{func.__name__} 실패 했습니다')

        if ret0['isdone']:
            return ret0
        
        node=kwargs.get('node', None)
        # ret = movel(node=node,dz=100, wait=True)
        # if not ret['isdone']:
        #     return ret
        
        # ret = movej(node=node, inputs='fold')
        # if not ret['isdone']:
        #     return ret

        return ret0
        
    return wrapper