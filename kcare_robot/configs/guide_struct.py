GUIDE= """
The robot is a mobile manipulator consisting of a 6-degree-of-freedom collaborative robot arm mounted on a 2-wheel mobile base. The arm is equipped with a wrist camera, a two-finger gripper with suction capability, an additional robotic arm, and a wide-area observation camera.

The robot's capabilities include:

place: Position a held object at a designated location.

Using these skills, develop a task plan to execute provided commands.

For example:

Command: give me pringles and cola

Program:
target_objects: pringles, cola
target_locations: None, None
destination_locations: me, me 



Command: Put cup into wooden dish. 
Program:
target_objects: cup
target_locations: None
destination_locations: wooden dish


Command: Place food on the dish  and give me drink
Program:
target_objects: food, drink
target_locations: None, None
destination_locations: dish, me

Command: Move cup on the shelf into wooden dish on the desk. 
Program:
target_objects: cup
target_locations: shelf
destination_locations: wooden dish@desk


Command: Give me remote control on the bed
Program:
target_objects: remote control
target_locations: bed
destination_locations: me

Command: I am thirsty
Program:
target_objects: drink
target_locations: None
destination_locations: me

Command: I am thirsty. Drink in the fridge
Program:
target_objects: drink
target_locations: fridge
destination_locations: me

Command: I am hungry
Program:
target_objects: food
target_locations: None
destination_locations: me

Command: I am hungry. Food on the fridge
Program:
target_objects: food
target_locations: fridge
destination_locations: me

Command: I am hungry and thirsty
Program:
target_objects: food, drink
target_locations: None, None
destination_locations: me, me

Command: Set up the dinning table. Food and dink on the shelf. 
Program:
target_objects: food, drink
target_locations: shelf, shelf
destination_locations: dinning table, dinning table

Command: place trash on table in living room  into trash bin in kitchen
Program:
target_objects: trash
target_locations: table@living room
destination_locations: trash bin@kitchen

Command: put bread and cheese in kitchen fridge on dining table
Program:
target_objects: bread, cheese
target_locations: kitchen fridge, kitchen fridge
destination_locations: dining table, dining table

Command: set up dinning table. Food and drink on the shelf
Program:
target_objects: food, drink
target_locations: shelf, shelf
destination_locations: table, table

Command: Bring me a towel. I'm in the bathroom.
Program:
target_objects: towel
target_locations: None
destination_locations: bathroom

Command: Bring me a towel. I'm in the bathroom.
Program:
target_objects: towel
target_locations: None
destination_locations: bathroom

Command: Wipe up the drink I spilled on the kitchen table.
target_objects: towel
target_locations: None
destination_locations: spill@kitchen table




Complete the following without any explanation and note:

Command: COMMAND_HERE.
Program:

"""


from pydantic import BaseModel
from typing import List

class TaskPlan(BaseModel):
  target_objects: List[str]
  target_locations: List[str]
  destination_locations: List[str] 
  
FORMAT = TaskPlan.model_json_schema()