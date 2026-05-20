from robot_agent.utils import exception_handler, refine_inputs
from robot_agent.skill_configs import GRIP_CONFIGS
import numpy as np


@exception_handler
def grip(**kwargs):
    node = kwargs.pop('node', None)
    agents = node.agents
    inputs = kwargs.pop('inputs')
    inputs = refine_inputs(inputs)
    target_pos = inputs.get('position', inputs['inputs'])
    

    target_pos = GRIP_CONFIGS['range'][1] if target_pos=='open' else GRIP_CONFIGS['range'][0] if target_pos=='close' else target_pos

    kwargs['position'] = float(np.clip(target_pos, GRIP_CONFIGS['range'][0], GRIP_CONFIGS['range'][1]))

    return  agents['grip'].send(kwargs)

    