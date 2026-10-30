"""Planner guide v2 (direct planner): output format + placeholders only.

robot_agent fills the placeholders when it plans (core/guide_manager.render_guide,
robot_agent/docs/planner_guide.md):

    {ROBOT}        the robot, from configs/manifest.yaml (capabilities / description)
    {ACTIONS}      one line per plan skill of the manifest (skill::<input> (Action): what it does)
    {EXAMPLES}     the seed-pool / verified-memory plans most similar to the command
    LOCATIONS_HERE the ENV places (with aliases)
    COMMAND_HERE   the command

Seeded into guides.json as the version 'guide_v2', NOT active: the robot keeps
planning with the active version until an operator selects this one (dashboard
Planner Guide -> guide_v2 -> Set active, or POST /guides/guide_v2/activate).
Only the direct planner uses it; SAGE builds its prompts from the manifest.
"""

GUIDE = """You plan tasks for this robot: {ROBOT}

Skills (write a step as skill::input):
{ACTIONS}

Places:
LOCATIONS_HERE

Examples:

{EXAMPLES}

Answer with the program only: one step per line, written skill::input, using only the skills
and places above. No explanation, no notes, no numbering.

Command: COMMAND_HERE.
Program:
"""

FORMAT = None
