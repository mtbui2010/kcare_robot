from robot_agent.utils import run_parallel_check, exception_handler, get_env_specs, get_lift_height, translate, text2voice, loc2text, correct_noun, correct_loc, announce_moving, announce_arrived
from kcare_robot.skills.arm import movej
from kcare_robot.skills.head import moveh, get_robot_mode
from kcare_robot.skills.lift import lift
from robot_agent.skill_configs import ENV, LIFT_CONFIGS, MOBILE_CONFIGS, HOME_LOC
from pyconnect.utils import update_dict
import numpy as np, threading
from robot_agent.utils import quaternion2deg, deg2quaternion

def check_current_loc(node, loc):
    try:
        loc0 = node.agents['mobile_pose'].get()
        return np.linalg.norm([loc['x'] - loc0['x'], loc['y'] - loc0['y']]) < 50
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
    
    return {'isdone': True, 'pose': [pos.x, pos.y, rz]}

@exception_handler
def moveb(node, **kwargs):


    x0, y0, rz0 = mobile_pose(node=node)['pose']
    x, y, rz = kwargs.pop('x', x0), kwargs.pop('y', y0), kwargs.pop('rz', rz0)
    qx, qy, qz, qw = deg2quaternion(0,0, rz)
    
    return node.agents['mobile_move'].send({'x': x, 'y':y, 'z':0., 'qx': qx, 'qy':qy, 'qz': qz, 'qw': qw})
    
    

@exception_handler
def forward(node, **kwargs):
    inp = float(kwargs.pop("inputs"))
    if abs(inp)<0.02:
        return {'isdone': True}
    
    
    return node.agents['mobile_forward'].send({'distance': inp})
    # return node.agents['mobile_forward'].send({'x': inp})


@exception_handler
def turn(node, **kwargs):
    inp = float(kwargs.pop("inputs"))
    if abs(inp)<5:
        return {'isdone': True}
    
    return node.agents['mobile_turn'].send({'theta': inp*np.pi/180.})

@exception_handler
def rotate(node, **kwargs):
    inp = float(kwargs.pop("inputs"))
    
    return node.agents['mobile_rotate'].send({'theta': inp*np.pi/180.})
    # return node.agents['mobile_rotate'].send({'target_yaw': inp*np.pi/180.})


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
    dforward = env.get('dforward', MOBILE_CONFIGS['dforward']) 
    dshift = env.get('dshift',  MOBILE_CONFIGS['dshift']) 
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
            lambda : agents['turn'].send({'inputs':env_loc['rz'] + turn_deg -agents['mobile_pose'].get()['rz'], 'wait':True}),
            lambda : moveh(node=node, ry='straight', rz=robot_mode, wait=True)
        ])

    # backward 
    if prev_robot_mode!='front':
        ret = run_parallel_check(funcs=[
            lambda : moveh(node=node, inputs='straight,front', wait=True),
            lambda : node.agents['turn'].send({'inputs': back_turn_deg, 'wait': True}),
            # lambda : movej(node=node, inputs='fold', mode='front'),
        ])
        if not ret['isdone']:
            return ret
        
        ret = run_parallel_check(funcs=[
            lambda : forward(node=node, inputs=-0.3, wait=True),
        ])
        if not ret['isdone']:
            return ret
    
    # move
    kwargs.update({'wait':True})
    if not run_parallel_check(funcs=[
        lambda : lift(node=node, inputs='home', mode='front'),
        # lambda : movej(node=node, inputs='fold', mode=robot_mode),
        lambda : moveb(node=node, **kwargs) ,
    ]) ['isdone']:
        raise  Exception('moveh/lift/moveb failed ...')
    
    ret = run_parallel_check(funcs=[
        # lambda : lift(node=node, inputs=lift_height, mode=robot_mode, wait=True),
        lambda : lift(node=node, inputs=lift_height, mode=robot_mode, wait=True),
        # lambda : movej(node=node, inputs='fold', mode=robot_mode, wait=True),
    ])
    if not ret['isdone']:
        return ret

    # forward
    ret = run_parallel_check(funcs=[
        # lambda : lift(node=node, inputs=lift_height, mode=robot_mode, wait=True),
        lambda : forward(node=node, inputs=dforward, wait=True),
    ])
    if not ret['isdone']:
        return ret
    
    # text2voice(translate(f'I am arrived at {loc2text(env_name)}', to_language='korean'))
    announce_arrived()
    
    return run_parallel_check(funcs=[
        lambda : turn(node=node, inputs=turn_deg),
        lambda : moveh(node=node, ry='straight', rz=robot_mode, wait=True),
        lambda : movej(node=node, inputs='fold', mode=robot_mode, wait=True),
    ])
    





