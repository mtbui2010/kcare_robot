from robot_agent.skill_configs import VLA_CLIENTS
from pyconnect.utils import init_detect_client
from robot_agent.utils import exception_handler
import numpy as np

vla_cfg = VLA_CLIENTS['openpi']
vla_client = init_detect_client(vla_cfg['url'])


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
    
    