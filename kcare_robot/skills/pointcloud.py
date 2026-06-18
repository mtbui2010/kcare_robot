
from robot_agent.utils import exception_handler
from robot_agent.skills import log_data
import numpy as np

@exception_handler
def get3d(node, **kwargs):
    p2 = kwargs['points']
    u  = [int(p[0]) for p in p2]
    v  = [int(p[1]) for p in p2]
    out = {'u_array':u, 'v_array': v,  'base_frame': 'base_footprint'}
    if  len(p2[0])==3:
        out['d_array'] = [float(p[2]/1000.) for p in p2]

    ret =  node.agents['get3d'].send(out)
    ret['pose'] = np.stack(
        (np.asarray(ret.pop('x')),
        np.asarray(ret.pop('y')),
        np.asarray(ret.pop('z'))),axis=-1)
    
    # inds = ~np.isnan(P).any(axis=-1)
    # ret['pose'] = P[inds, ...]
    
    return ret
