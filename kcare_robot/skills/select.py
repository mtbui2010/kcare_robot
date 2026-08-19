from robot_agent.utils import text2voice
from kcare_robot.utils import correct_noun
from kcare_robot.skills.place import place

def select_response(**kwargs):
    node = kwargs.pop('node')
    inp = kwargs.pop('inputs')
    text2voice(f'{inp} 선택 확인', run_thread=False)    
    inp_en = correct_noun(inp, lang='english')
    
    # return {'isdone': True}
    
    return place(node=node, inputs=f'{inp_en}@shelf>>sopha table@living room')
    
