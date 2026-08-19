"""Pattern 2 -- pure rclpy on top of the device registered in robot_agent.

The 'grip' device is still registered via POST /devices, but the skill bypasses
the NodeAgent.send() wrapper and works directly with the underlying
rclpy.Client (exposed as `agent.raw`). The shared MultiThreadedExecutor is
already spinning, so we just `call_async` and poll the future.

Pros : full rclpy control (timeout, retry, custom request shape, error fields
       beyond `isdone/msg`); no extra connection setup.
Cons : skill is responsible for future handling and timing out; bypasses
       the connect layer's log_msg.
"""

import time
import rclpy
from rosinterfaces.srv import SendStringData

from robot_agent.connect.serde import dict2str, str2dict
from robot_agent.utils import exception_handler, refine_inputs
from robot_agent.skill_configs import GRIP_CONFIGS
import numpy as np


@exception_handler
def grip(node,**kwargs):
    timeout = float(kwargs.pop('timeout', 10.0))

    inputs = refine_inputs(kwargs.pop('inputs', ''))
    target_pos = inputs.get('position', inputs.get('inputs', 'open'))

    lo, hi = GRIP_CONFIGS['range']
    if target_pos == 'open':
        target_pos = hi
    elif target_pos == 'close':
        target_pos = lo
    kwargs['position'] = float(np.clip(target_pos, lo, hi))

    agent = node.agents.get('grip')
    if agent is None or agent.raw is None:
        return {'isdone': False, 'msg': "device 'grip' not registered"}

    cli = agent.raw
    if not cli.wait_for_service(timeout_sec=2.0):
        return {'isdone': False, 'msg': "service 'grip' unavailable"}

    request = SendStringData.Request()
    request.req = dict2str(kwargs)

    future = cli.call_async(request)
    t0 = time.time()
    while rclpy.ok() and not future.done():
        if time.time() - t0 > timeout:
            return {'isdone': False, 'msg': f'timeout after {timeout}s'}
        time.sleep(0.02)

    return str2dict(future.result().ret)
