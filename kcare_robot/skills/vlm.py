

from robot_agent.utils import exception_handler
from robot_agent.state import current


@exception_handler
def llm(node, **kwargs):
    llm_client = current().dm.get_client('llm')
    if llm_client is None:
        raise RuntimeError("LLM client 'llm' not registered — add it via Connections panel or switch_llm()")
    ret_msg = llm_client.chat(prompt=kwargs['inputs'])
    return {'isdone': True, 'msg': ret_msg}
