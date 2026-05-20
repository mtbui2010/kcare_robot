GUIDE= """
이 로봇은 2륜 이동식 베이스에 장착된 6자유도 협동 로봇 팔로 구성된 이동식 매니퓰레이터입니다. 팔에는 손목 카메라, 흡입 기능이 있는 두 손가락 그리퍼, 추가 로봇 팔, 그리고 광역 관찰 카메라가 장착되어 있습니다.

로봇의 기능은 다음과 같습니다.

위치: 지정된 위치에 물체를 놓습니다.

이러한 기술을 사용하여 제공된 명령을 실행하는 작업 계획을 개발합니다.

예:

명령: 프링글스와 콜라를 줘

프로그램:
대상 객체: 프링글스, 콜라
대상 위치: None, None
목적지 위치: 나, 나

명령: 컵을 나무 접시에 넣어
프로그램:
target_objects: cup
target_locations: None
destination_locations: 나무 접시

명령: 접시에 음식을 놓고 음료를 주세요
프로그램:
target_objects: food, drink
target_locations: None, None
destination_locations: dish, me

명령: 선반 위의 컵을 책상 위의 나무 접시로 옮겨라.
프로그램:
target_objects: 컵
target_locations: 선반
destination_locations: 나무 접시@책상

명령: 침대 위의 리모컨을 줘
프로그램:
target_objects: 리모컨
target_locations: 침대
destination_locations: 나

명령어: 목마르다
프로그램:
target_objects: 음료
target_locations: None
destination_locations: 나

명령: 목마르다. 냉장고에 음료가 있어
프로그램:
target_objects: 음료
target_locations: 냉장고
destination_locations: 나

명령: 배고프다
프로그램:
target_objects: 음식
target_locations: None
destination_locations: 나

명령: 배고파. 냉장고에 음식이 있어.
프로그램:
target_objects: 음식
target_locations: 냉장고
destination_locations: 나

명령: 배고프고 목마르다.
프로그램:
target_objects: 음식, 음료
target_locations: None, None
destination_locations: 나, 나

명령: 식탁을 차려. 음식과 음료는 선반에 있어.
프로그램:
target_objects: 음식, 음료
target_locations: 선반, 선반
destination_locations: 식탁, 식탁

명령: 식탁 위 주방 냉장고에 빵과 치즈를 넣으세요
프로그램:
target_objects: 빵, 치즈
target_locations: 주방 냉장고, 주방 냉장고
destination_locations: 식탁, 식탁

명령: 식탁을 차려주세요. 선반에 음식과 음료를 놓으세요
프로그램:
target_objects: 음식, 음료
target_locations: 선반, 선반
destination_locations: 테이블, 테이블


설명이나 메모 없이 다음을 완성하세요.

명령: COMMAND_HERE.
프로그램:

"""


from pydantic import BaseModel
from typing import List

class TaskPlan(BaseModel):
  target_objects: List[str]
  target_locations: List[str]
  destination_locations: List[str] 
  
FORMAT = TaskPlan.model_json_schema()