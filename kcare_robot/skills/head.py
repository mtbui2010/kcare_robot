from robot_agent.utils import exception_handler
from robot_agent.skill_configs import HEAD_CONFIGS
import numpy as np



@exception_handler
def head_state(**kwargs):
    node = kwargs.pop('node', None)
    current_rz, current_ry = node.agents['joint_states'].get()['position'][1:3]
    return {'isdone': True, 'current_ry': current_ry*180/np.pi, 'current_rz': current_rz*180/np.pi}


def get_robot_mode(node=None, head_rz=None, env={}):
    if 'default_mode' in env:
        return env['default_mode']

    if node is None and head_rz is None:
        return None

    if head_rz is None:
        hstate = head_state(node=node)
        if hstate is None:
            return None
        head_rz = hstate['current_rz']

    head_position_dict = {-90: 'left', 0: 'front', 90:'right'}
    head_angles = np.array(list(head_position_dict.keys()))
    argmin = int(np.argmin(np.abs(head_angles-head_rz)))

    return head_position_dict[head_angles[argmin]]


@exception_handler
def moveh(**kwargs):
    node = kwargs.pop('node', None)
    agents = node.agents
    inp = kwargs.pop('inputs', None)
    
    if isinstance(inp, str):
        for el in inp.split(','):
            el = el.strip().lower()
            if el in HEAD_CONFIGS['ry']:
                kwargs['ry'] = HEAD_CONFIGS['ry'][el]
            elif el in HEAD_CONFIGS['rz']:
                kwargs['rz'] = HEAD_CONFIGS['rz'][el]
    
    hstate =  head_state(node=node)
    ry = kwargs.pop('ry', hstate['current_ry'] )
    ry = float(HEAD_CONFIGS['ry'][ry] if isinstance(ry, str) else ry)
    rz  =kwargs.get('rz', hstate['current_rz'] )
    rz = float(HEAD_CONFIGS['rz'][rz] if isinstance(rz, str) else rz)
    
    kwargs['head_joint2'], kwargs['head_joint1'] = ry*np.pi/180, rz*np.pi/180.

    return agents['head_move'].send(kwargs)



 