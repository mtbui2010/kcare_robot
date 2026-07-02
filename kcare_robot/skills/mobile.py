from robot_agent.utils import run_parallel_check, exception_handler, get_env_specs, get_lift_height, translate, text2voice, loc2text, correct_noun, correct_loc, announce_moving, announce_arrived
from kcare_robot.skills.arm import movej
from kcare_robot.skills.head import moveh, get_robot_mode
from kcare_robot.skills.lift import lift
from robot_agent.skill_configs import ENV, LIFT_CONFIGS, MOBILE_CONFIGS, HOME_LOC
from pyconnect.utils import update_dict
import numpy as np, threading, time
from robot_agent.utils import quaternion2deg, deg2quaternion

def check_current_loc(node, loc):
    try:
        x0, y0 = mobile_pose(node=node)['pose'][:2]
        x, y = loc['x'], loc['y']

        return np.linalg.norm([x-x0, y-y0]) < 0.25
    except Exception as e:
        print(e)
        return False
    

get_turn_deg = lambda robot_mode: -90 if robot_mode=='right' else 90 if robot_mode=='left' else 0

@exception_handler
def mobile_pose(node, **kwargs):
    ret =  node.agents['mobile_pose'].get()
    assert ret is not None, f'check mobile_pose connection'
    pos, ort = ret['pose'].position, ret['pose'].orientation
    rz= quaternion2deg(ort.x, ort.y,ort.z, ort.w)[-1]
    
    return {'isdone': True, 'pose': [float(pos.x), float(pos.y), float(rz)]}

@exception_handler
def moveb(node, **kwargs):
    wait = kwargs.pop('wait', True)

    x0, y0, rz0 = mobile_pose(node=node)['pose']
    x, y, rz = float(kwargs.pop('x', x0)), float(kwargs.pop('y', y0)), float(kwargs.pop('rz', rz0))
    qx, qy, qz, qw = deg2quaternion(0,0, rz)
    
    return node.agents['mobile_move'].send({'x': x, 'y':y, 'z':0., 'qx': qx, 'qy':qy, 'qz': qz, 'qw': qw, 'wait': wait})

@exception_handler
def turn(node, **kwargs):
    inp = float(kwargs.pop("inputs"))
    wait = kwargs.pop('wait', True)
    if abs(inp)<5:
        return {'isdone': True}
    
    return node.agents['mobile_turn'].send({'theta': inp*np.pi/180., 'wait': wait})

@exception_handler
def rotate(node, **kwargs):
    inp = float(kwargs.pop("inputs"))
    wait = kwargs.pop('wait', True)
    
    return node.agents['mobile_rotate'].send({'theta': inp*np.pi/180., 'wait': wait})
    # return node.agents['mobile_rotate'].send({'target_yaw': inp*np.pi/180.})
    
    

@exception_handler
def forward(node, **kwargs):
    inp = float(kwargs.pop("inputs"))
    wait = kwargs.pop('wait', True)

    if abs(inp)<0.02:
        return {'isdone': True}
    
    prev_rz = mobile_pose(node=node)['pose'][-1]
    
    ret = node.agents['mobile_forward'].send({'distance': inp, 'wait':wait})
    assert ret['isdone'], f'{ret}'

    return rotate(node=node, inputs=prev_rz)





@exception_handler
def move(node, **kwargs):
    agents = node.agents   
    env_name = kwargs.pop('inputs')
    
    if 'home' in env_name:
        moveh(node=node, inputs='front', wait=False)
        return moveb(node=node, wait=True, **HOME_LOC)   

    if 'base' in env_name:
        moveh(node=node, inputs='front', wait=False) 
        moveb(node=node, wait=True, x=0.3, y=0, theta=0) 
        return forward(node=node, inputs=-0.3)
    
    
    env = get_env_specs(env_name, ENV)
    
    threading.Thread(target=announce_moving, args=(env_name,), daemon=True).start()
    
    prev_robot_mode = get_robot_mode(node=node)
    back_turn_deg = - get_turn_deg(robot_mode=prev_robot_mode)
    
    # approach to new location
    env_loc = env.get('loc', None)
    robot_mode = env.get('default_mode', 'front') 
    dforward = MOBILE_CONFIGS['dforward'] + env.get('dforward', 0) 
    dshift = MOBILE_CONFIGS['dshift'] + env.get('dshift',  0) 
    robot_mode = kwargs.get('mode', robot_mode) 
    lift_height = get_lift_height(env, robot_mode)

    turn_deg = get_turn_deg(robot_mode=robot_mode)
    dforward =  0 if robot_mode=='front' else dforward
    dshift =  0 if robot_mode=='front' else dshift
    
    is_current_loc = False
    if env_loc is None:
        kwargs['inputs']=env.get('label', env_name)
    else:
        robot_ori, turn_ori = env_loc['rz']*np.pi/180,  (env_loc['rz']+turn_deg)*np.pi/180
        env_loc['x'] +=  dshift*np.cos(turn_ori)
        env_loc['y'] +=  dshift*np.sin(turn_ori)
        kwargs.update(env_loc)
        
        # is_current_loc =  check_current_loc(node, env_loc)
        is_current_loc =  check_current_loc(node, {
            'x': env_loc['x'] + dforward*np.cos(robot_ori),
            'y': env_loc['y'] + dforward*np.sin(robot_ori),
        })
            
    
    if is_current_loc:
        return run_parallel_check(funcs=[
            # lambda : agents['turn'].send({'inputs':env_loc['rz'] + turn_deg -agents['mobile_pose'].get()['rz'], 'wait':True}),
            lambda : turn(node=node, inputs=env_loc['rz'] + turn_deg -mobile_pose(node=node)['pose'][-1]),
            lambda : moveh(node=node, ry='straight', rz=robot_mode, wait=True)
        ])

    # backward 
    if prev_robot_mode!='front':
        # ret = run_parallel_check(funcs=[
        #     lambda : moveh(node=node, inputs='straight,front', wait=True),
        #     # lambda : node.agents['turn'].send({'inputs': back_turn_deg, 'wait': True}),
        #     lambda : turn(node=node, inputs=back_turn_deg),
        # ])
        # if not ret['isdone']:
        #     return ret
        
        # ret = run_parallel_check(funcs=[
        #     lambda : forward(node=node, inputs=-0.3, wait=True),
        # ])
        # if not ret['isdone']:
        #     return ret

        ret = moveh(node=node, inputs='straight,front', wait=True)
        assert  ret['isdone'], f'{ret}'
    
    # move
    kwargs.update({'wait':True})
    if not run_parallel_check(funcs=[
        lambda : lift(node=node, inputs='home', mode='front'),
        # lambda : movej(node=node, inputs='fold', mode=robot_mode),
        lambda : (time.sleep(3), movej(node=node, inputs='fold', mode='front'))[-1],
        lambda : moveb(node=node, **kwargs) ,
    ]) ['isdone']:
        raise  Exception('moveh/lift/moveb failed ...')
    
    ret = lift(node=node, inputs=lift_height, mode=robot_mode, wait=True)
    assert ret['isdone'], f'{ret}'

    ret = forward(node=node, inputs=dforward, wait=True)
    assert ret['isdone'], f'{ret}'
    
    # text2voice(translate(f'I am arrived at {loc2text(env_name)}', to_language='korean'))
    announce_arrived()
    

    ret =  run_parallel_check(funcs=[
        lambda: turn(node=node, inputs=turn_deg),
        lambda : moveh(node=node, ry='straight', rz=robot_mode, wait=True),
        # lambda : movej(node=node, inputs='fold', mode=robot_mode, wait=True),
        lambda : movej(node=node, inputs='give', mode=robot_mode, wait=True),
    ])
    assert ret['isdone'], f'{ret}'
    
    kwargs['isdone'] = True
    return kwargs

    





