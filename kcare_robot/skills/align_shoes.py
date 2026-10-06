

from robot_agent.skills import log_data
from robot_agent.state import current
from kcare_robot.skills.recognition import find_grasp
from kcare_robot.skills.lift import lift, dlift
from kcare_robot.skills.arm import movet, movej, arm_exception_handler, movel, movelf, arm_joints



def align_shoes(node, **params) -> dict:
    ready_shoes(node)
    #ret = find_grasp(node=node, inputs="white shoes")
    #ret = find_arm(node=node, inputs="white shoes")
    ret= find_grasp(node=node, inputs="black shoes", max_instances="2")
    assert ret['isdone'], f'{ret}'

    

    return ret


def ready_shoes(node):
    lift(node=node, inputs=0.67, wait=True)
    movej(node=node, inputs=[0, 0, 0, 25, 0, 25, 180], speed=0.5, acc=0.3, wait=True)
