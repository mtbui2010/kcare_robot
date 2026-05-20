"""Pattern 1 -- pyconnect NodeAgent.

The 'grip' device is registered once via POST /devices (conn_type=ros_service,
conn_name='grip'). robot_agent's DeviceManager attaches the rclpy client to the
shared CustomNode and wraps it in a ServiceClientAgent (encode/decode via
dict2str/str2dict, with logging hooks).

The skill just calls `.send(...)` -- everything else is handled by pyconnect.

Pros : shortest code, free logging, reuse of registered device.
Cons : payload bound to SendStringData (req/ret are strings); custom QoS or
       feedback streaming is not exposed.
"""

from robot_agent.utils import exception_handler, refine_inputs
from robot_agent.skill_configs import GRIP_CONFIGS
import numpy as np


@exception_handler
def grip(node, **kwargs):
    inputs = refine_inputs(kwargs.pop('inputs', ''))
    target_pos = inputs.get('position', inputs.get('inputs', 'open'))

    lo, hi = GRIP_CONFIGS['range']
    if target_pos == 'open':
        target_pos = hi
    elif target_pos == 'close':
        target_pos = lo

    kwargs['position'] = float(np.clip(target_pos, lo, hi))
    return node.agents['grip'].send(kwargs)
