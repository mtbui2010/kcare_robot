from robot_agent.utils import exception_handler, refine_inputs
from kcare_robot.skills.head import get_robot_mode
from robot_agent.skill_configs import LIFT_CONFIGS
import numpy as np

@exception_handler
def lift(**kwargs):
    node = kwargs.pop('node', None)
    agents = node.agents
    inp = refine_inputs(kwargs.pop('inputs'))['inputs']

    
    robot_mode = get_robot_mode(node=node)
    # robot_mode = 'front'
    if isinstance(inp, str):
        if inp=='lowest':
            inp = LIFT_CONFIGS['range'][0]
        elif inp=='highest':
            inp = LIFT_CONFIGS['range'][1]
        elif inp=='home':
            inp = LIFT_CONFIGS['home'][robot_mode]
        else:
            NotImplementedError

    kwargs['target_height'] = float(np.clip(inp, LIFT_CONFIGS['range'][0], LIFT_CONFIGS['range'][1])) - LIFT_CONFIGS['mobile_height']
    # kwargs['until_complete'] =  kwargs.pop('wait', True)
    
    return agents['lift'].send(kwargs)

@exception_handler
def lift_state(**kwargs):
    node = kwargs.pop('node', None)
    lift_state = node.agents['joint_states'].get()['position'][0] + LIFT_CONFIGS['mobile_height']
    return {'isdone': True, 'current_position': lift_state}


@exception_handler
def dlift(**kwargs):
    node = kwargs.pop('node', None)
    inp = refine_inputs(kwargs.pop('inputs', 0))['inputs']

    lift_to = lift_state(node=node)['current_position'] + inp
    return  lift(inputs=lift_to, node=node)