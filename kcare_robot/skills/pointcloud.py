
from robot_agent.utils import exception_handler
from robot_agent.skills import log_data

@exception_handler
def get3d(node, **kwargs):
    u, v = kwargs['x'], kwargs['y']
    ret =  node.agents['get3d'].send({'u':int(u), 'v': int(v), 'base_frame': 'base_footprint'})
    out = (ret.pop('x'), ret.pop('y'), ret.pop('z'))
    ret['pose'] = out
    log_data({'input': (u,v), 'output': out})
    
    return ret
