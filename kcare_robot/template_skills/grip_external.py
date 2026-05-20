"""Pattern 3 -- fully self-managed skill exposed over HTTP.

The skill runs in its own process (own rclpy.init, own Node, own lifecycle).
robot_agent only stores a URL and forwards POST /skill/grip there.

Two files in this template:
    1. The FastAPI server below (`uvicorn grip_external:app --port 9000`)
    2. A one-shot registration call (curl example at the bottom)

Pros : no robot_agent / pyconnect import, any language, any ROS_DOMAIN_ID,
       crash isolation, custom timeout/headers via SkillDef.
Cons : extra process, HTTP overhead (~ms), no streaming feedback.
"""

import threading
import time
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosinterfaces.srv import SendStringData
from fastapi import FastAPI
from pydantic import BaseModel

from pyconnect.utils import dict2str, str2dict


class GripNode(Node):
    """Owns its own rclpy.Client. Spun in a background thread."""

    def __init__(self):
        super().__init__('grip_external_node')
        self.cli = self.create_client(SendStringData, 'grip')

    def call(self, payload: dict, timeout: float = 10.0) -> dict:
        if not self.cli.wait_for_service(timeout_sec=2.0):
            return {'isdone': False, 'msg': "service 'grip' unavailable"}
        req = SendStringData.Request()
        req.req = dict2str(payload)
        future = self.cli.call_async(req)
        t0 = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - t0 > timeout:
                return {'isdone': False, 'msg': f'timeout after {timeout}s'}
            time.sleep(0.02)
        return str2dict(future.result().ret)


if not rclpy.ok():
    rclpy.init()
_node = GripNode()
_executor = MultiThreadedExecutor(num_threads=2)
_executor.add_node(_node)
threading.Thread(target=_executor.spin, daemon=True).start()


app = FastAPI(title='grip-external')


class GripParams(BaseModel):
    position: float | str = 'open'
    timeout: float = 10.0


@app.post('/grip')
def grip(params: GripParams):
    pos = params.position
    if pos == 'open':
        pos = 1.0
    elif pos == 'close':
        pos = 0.0
    return _node.call({'position': float(pos)}, timeout=params.timeout)


# Register this skill with robot_agent (run once):
#
#   curl -X POST http://localhost:8001/skills \
#        -H 'Content-Type: application/json' \
#        -d '{"name": "grip",
#             "type": "external",
#             "url":  "http://localhost:9000/grip",
#             "timeout": 15,
#             "method": "POST"}'
#
# Then call it like any other skill:
#
#   curl -X POST http://localhost:8001/skill/grip \
#        -H 'Content-Type: application/json' \
#        -d '{"position": "open"}'
