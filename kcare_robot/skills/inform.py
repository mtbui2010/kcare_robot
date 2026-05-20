import numpy as np
from robot_agent.utils import text2voice

def inform(**kwargs):
    
    node =kwargs.pop('node')
    inp = kwargs.pop('inputs').strip().lower()
    
    if 'light' in inp:
        rgb=None
        try:
            rgb = node.agents['head_rgb'].get()['im']
        except Exception as e:
            print(e)
            try:
                rgb = node.agents['wrist_raw'].get()['rgb']
            except Exception as e:
                print(e)
        mean_light = np.mean(rgb)
        msg = '몰라요' if rgb is None else '불이 켜져있어요'  if mean_light>kwargs.get('thresh', 125) else '불이 꺼져있어요'
        text2voice(msg, run_thread=False)
        return {'isdone': True, 'msg': msg}
    
    return {'isdone': False}