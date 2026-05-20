GUIDE= """
The robot is a mobile manipulator consisting of a 6-degree-of-freedom collaborative robot arm mounted on a 2-wheel mobile base. The arm is equipped with a wrist camera, a two-finger gripper with suction capability, an additional robotic arm, and a wide-area observation camera.

The robot's capabilities include:

pick: Grasp an object using the robot arm and gripper.
place: Position a held object at a designated location.
moveb_left: Navigate the mobile base to the location of an object or a target destination.
find: Use the camera to locate an object at a specified place, returning its location if found or None if not.

Known Information:

Shelf: Primarily contains necessary items.

Using these skills, develop a task plan to execute provided commands.

For example:

Command: give me pringles and cola

Program:
moveb_left::shelf
find::Pringles
pick::Pringles
moveb_left::desk
place::me
moveb_left::shelf
find::cola
pick::cola
moveb_left::desk
place::me


Command: give me food and drink

Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
place::me
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
place::me

Command: Put cup into wooden dish. 
Program:
moveb_left::shelf
find::cup
pick::cup
moveb_left::desk
find::wooden dish
place::wooden dish

Command: Place food on the dish 
Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
find::dish
place::dish

Command: Place food on the dish  and give me drink
Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
find::dish
place::dish
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
place::me

Command: Put cup into wooden dish on the desk. 
Program:
moveb_left::shelf
find::cup
pick::cup
moveb_left::desk
find::wooden dish
place::wooden dish

Command: Move cup on the shelft into wooden dish on the desk. 
Program:
moveb_left::shelf
find::cup
pick::cup
moveb_left::desk
find::wooden dish
place::wooden dish


Command: Give me remote control on the bed
Program:
moveb_left::bed
find::remote control
pick::remote control
moveb_left::desk
place::me

Command: I am thirsty
Program:
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
place::me

Command: I am thirsty. Drink in the fridge
Program:
moveb_left::fridge
find::drink
pick::drink
moveb_left::desk
place::me

Command: I am hungry
Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
place::me

Command: I am hungry. Food on the fidge
Program:
moveb_left::fridge
find::food
pick::food
moveb_left::desk
place::me



Command: I am hungry and thirsty
Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
place::me
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
place::me

Command: I am thirsty and hungry
Program:
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
place::me
moveb_left::shelf
find::food
pick::food
moveb_left::desk
place::me

Command: I am hungry and thirsty. Food and drink in the fridge
Program:
moveb_left::fridge
find::food
pick::food
moveb_left::desk
place::me
moveb_left::fridge
find::drink
pick::drink
moveb_left::desk
place::me

Command: Approach to remote control
Program:
find::remote control
approach::rempte control

Command: Set up the dinning table on the desk. Food and dink on the shelf. 
Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
find::empty dish
place::empty dish
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
find::empty dish
place::empty dish

Command: Set up the dinning table on the desk. 
Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
find::empty dish
place::empty dish
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
find::empty dish
place::empty dish


Command: Set up for dining.
Program:
moveb_left::shelf
find::food
pick::food
moveb_left::desk
find::empty dish
place::empty dish
moveb_left::shelf
find::drink
pick::drink
moveb_left::desk
find::empty dish
place::empty dish


Complete the following without any explanation and note:

Command: COMMAND_HERE.
Program:

"""

FORMAT = None