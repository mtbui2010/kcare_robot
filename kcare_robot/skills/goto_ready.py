from robot_agent.utils import run_parallel_check, exception_handler
from robot_agent.skill_configs import ARM_CONFIGS
from kcare_robot.skills.grip import grip
from kcare_robot.skills.head import moveh, get_robot_mode
from kcare_robot.skills.lift import lift
from kcare_robot.skills.arm import movej




@exception_handler
def init_arm(node, device_mode='', **kwargs):

    ret = node.agents['arm_enable'].send()
    if not ret['isdone']:
        return ret

    robot_mode = kwargs.pop('inputs', get_robot_mode(node=node))
    ret = run_parallel_check(funcs=[
        lambda : moveh(node=node, ry='straight', rz=robot_mode, wait=False),
        lambda : grip(node=node, inputs='close', wait=False),
        # lambda : node.agents['lift'].send({'dr': 20, 'current_position': node.agents['lm_state'].get()['current_position'], 'robot_params': RobotParam}),
    ])
    if not ret['isdone']:
        return ret
    
    ret = movej(node=node, inputs='fold', mode=robot_mode, wait=True)
    if not ret['isdone']:
        return ret
    
    ret = run_parallel_check(funcs=[
        # lambda : moveh(node=node, ry='straight', rz=robot_mode, wait=False),
        lambda : grip(node=node, inputs='open', wait=False),
        lambda : lift(node=node, inputs='home', mode=robot_mode),
    ])
    
    return ret


  