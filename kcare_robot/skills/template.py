
from robot_agent.connect.ros.node import CustomNode
from robot_agent.connect.ros.configs import get_default_service_server, get_default_pub, get_default_service_client
from robot_agent.skill_configs import PROMPT_CONFIGS, OBSERVATION_NODE_CONFIGS, DEVICE_CLIENT_CONFIGS, GUIDE
from robot_agent.utils import run_parallel_check
import threading, time


def run_node_using_api():
    node = CustomNode('shift')

    # this agent to acquire device states from task manager
    node.add_agent(**get_default_service_client(), conn_name='device_states')

    # this server agent for receiving msg and response
    def response_func(data):
        print(data)
        return node.agents['device_states'].send({'device_name': 'femto_rgb'}, show_info=True)
    node.add_agent(**get_default_service_server(), conn_name='find', response_func=response_func)

    node.spin()


def run_node():
    import rclpy, threading
    from rclpy.node import Node
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rosinterfaces.srv import SendStringData

    from robot_agent.connect.serde import str2dict, dict2str
    from robot_agent.connect.helpers import data_info
    # rclpy.init()

    class SkillNode(Node):
        def __init__(self, name='shift', conn_name='shift', data_interface=SendStringData):
            super().__init__(name)

            # this server agent for receiving msg and response
            self.create_service(data_interface,  conn_name, self.callback, callback_group=ReentrantCallbackGroup())
            

            # this agent to acquire device states from task manager
            device_conn_name = 'device_states'
            self.exector = self.create_client(data_interface, device_conn_name, callback_group=ReentrantCallbackGroup())
            def wait_service():
                print(f'Waiting for {device_conn_name} service...')
                while not self.exector.wait_for_service(timeout_sec=1.0):
                    # print(f'Waiting for {conn_name} service...')
                    pass
                print(f'Service {device_conn_name} connected ...')
                self.request = data_interface.Request()
            threading.Thread(target=wait_service, daemon=True).start()

        def callback(self, request, response):
            
            # receive data
            rev_data = str2dict(request.req)
            print(data_info(rev_data))

            # acquire needed data
            params = self.send({'device_name': 'femto_rgb'})

            # return the response
            response.ret = dict2str(params)
            return response
                

        def send(self, data={}, wait_until_done=True, **kwargs):

            try:
                self.request.req= dict2str(data)
            except Exception as e:
                print(f'Error: {e}. Stopped ...')
                return {'isdone': False} 

            future = self.exector.call_async(self.request)
            while rclpy.ok() and not future.done() and wait_until_done:
                time.sleep(0.1)
            result = future.result()

            ret_data = str2dict(result.ret)
            return ret_data
    
    # Create client and run in thread
    node = SkillNode()
    try:

        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__=='__main__':
    run_node()

        
        