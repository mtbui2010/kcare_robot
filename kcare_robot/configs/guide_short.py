GUIDE= """
The robot is a mobile manipulator consisting of a 6-degree-of-freedom collaborative robot arm mounted on a 2-wheel mobile base. The arm is equipped with a wrist camera, a two-finger gripper with suction capability, an additional robotic arm, and a wide-area observation camera.

The robot's capabilities include:

pick: Grasp an object using the robot arm and gripper.
place: Position a held object at a designated location.

Known Information:

Shelf: Primarily contains necessary items.

Using these skills, develop a task plan to execute provided commands.

For example:

Command: give me pringles and cola

Program:
place::pringles>>me
place::cola>>me


Command: give me food and drink

Program:
place::food>>me
place::drink>>me

Command: Put cup into wooden dish. 
Program:
place::cup>>wooden dish

Command: Place food on the dish 
Program:
place::food>>dish

Command: Place food on the dish  and give me drink
Program:
place::food>>dish
place::drink>>me

Command: Put cup into wooden dish on the desk. 
Program:
place::cup@wooden dish>>desk

Command: Move cup on the shelft into wooden dish on the desk. 
Program:
place::cup@shelf>>dish@desk


Command: Give me remote control on the bed
Program:
place::remote control@bed>>me

Command: I am thirsty
Program:
place::drink>>me

Command: I am thirsty. Drink in the fridge
Program:
place::drink@fridge>>me

Command: I am hungry
Program:
place::food>>me

Command: I am hungry. Food on the fridge
Program:
place::food@fridge>>me

Command: I am hungry and thirsty
Program:
place::food>>me
place::drink>>me

Command: I am thirsty and hungry
Program:
place::drink>>me
place::food>>me

Command: I am hungry and thirsty. Food and drink in the fridge
Program:
place::food@fridge>>me
place::drink@fridge>>me

Command: pick remote control
Program:
pick::remote control

Command: Set up the dinning table. Food and dink on the shelf. 
Program:
place::food@shelf>>dinning table
place::drink@shelf>>dinning table

Command: Set up the dinning table.
Program:
place::food>>dinning table
place::drink>>dinning table

Command: place trash on table in living room  into trash bin in kitchen
Program:
place::trash@table@living room>>trash bin@kitchen

Command: place trash living room table  into kitchen trash bin
Program:
place::trash@table@living room>>trash bin@kitchen

Command: put bread and cheese in kitchen fridge on dining table
Program:
place::bread@fridge@kitchen>>dining table
place::cheese@fridge@kitchen>>dining table

Command: put bread and cheese in kitchen fridge on living room table
Program:
place::bread@fridge@kitchen>>table@living room
place::cheese@fridge@kitchen>>table@living room

Command: move cup on the desk to living room table 
Program:
place::cup@desk>>table@living room

Command: Give me snack on the desk
Program:
place::snack@desk>>me

Command: Give me snack
Program:
place::snack>>me

Command: Give me phone on the desk
Program:
place::phone@desk>>me

Command: Give me phone
Program:
place::phone>>me

Complete the following without any explanation and note:

Command: COMMAND_HERE.
Program:

"""

FORMAT = None