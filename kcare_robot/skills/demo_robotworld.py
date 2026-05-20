from robot_agent.utils import exception_handler
from kcare_robot.skills.place import place

@exception_handler
def demo_grasp(**kwargs):
    node = kwargs.pop('node')
    
    ret = place(node=node, inputs='spray bottle@table@source>>loc0@destination')
    if not ret['isdone']:
        return ret
    
    ret = place(node=node, inputs='drink@table@source>>loc1@destination ')
    if not ret['isdone']:
        return ret
    
    ret = place(node=node, inputs='cup@table@source>>loc2@destination ')
    if not ret['isdone']:
        return ret
    
    ret = place(node=node, inputs='control@table@source>>loc3@destination ')
    if not ret['isdone']:
        return ret
    
    return {'isdone': True}
    
    