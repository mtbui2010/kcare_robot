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
    return {'isdone': True, 'joints': [el*180./np.pi for el  in joints]}


@exception_handler
def arm_pose(node, **kwargs):
    ret =  node.agents['mobile_base_tool_pose'].get()
    assert ret is not None, f'check mobile_base_tool_pose connection'
    
    pos, ort = ret['pose'].position, ret['pose'].orientation
    rx, ry, rz= quaternion2deg(ort.x, ort.y,ort.z, ort.w)
    
    return {'isdone': True, 'pose': [pos.x, pos.y, pos.z, rx, ry, rz]}

@exception_handler
def movel(node, **kwargs):
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
    inputs = {}
    inputs['velocity_scale'] = kwargs.get('speed', 1.0)
    inputs['acceleration_scale'] = kwargs.get('acc', 0.4)

    node = kwargs.pop('node', None)
    agents = node.agents
    # inputs = kwargs.pop('inputs')
    
    robot_mode = kwargs.get('mode', get_robot_mode(node=node))
    # robot_mode = "left"

    # inputs = {'relative': 'inputs' not in kwargs}
    # inputs = {}
    dangles = {f'dr{i}': kwargs[f'dr{i}'] if f'dr{i}' in kwargs else 0. for i in range(7)}
    if 'inputs' in kwargs:
        angles = kwargs['inputs']
        if isinstance(angles, str):
            angles = ARM_CONFIGS[angles][robot_mode]
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
    qx, qy, qz, qw = deg2quaternion(kwargs.get('rx',0.), kwargs.get('ry',0.), kwargs.get('rz',0.))
    dx, dy, dz = float(kwargs.get('dx', 0.)), float(kwargs.get('dy', 0.)), float(kwargs.get('dz', 0.))
    
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
        
        ret = movej(node=node, inputs='fold')
        if not ret['isdone']:
            return ret

        return ret0
        
    return wrapper