import numpy as np
import cv2
import copy
import os,json
from robot_agent.skill_configs import CALIB_PARAMS
class Head2BaseCalibration():
    def __init__(self):
        self.fx=CALIB_PARAMS['fx']
        self.ppx=CALIB_PARAMS['ppx']
        self.fy=CALIB_PARAMS['fy']
        self.ppy=CALIB_PARAMS['ppy']

        # 딕셔너리 형태로 로드된 데이터 출력
        # print(CALIB_PARAMS)
        self.init_angle=CALIB_PARAMS['init_angle']
        self.link2base_robot=np.reshape(CALIB_PARAMS['link2base_robot'],[4,4])
        self.base2LMbase_robot=np.reshape(CALIB_PARAMS['base2LMbase_robot'],[4,4])
        self.error_linear=np.reshape(CALIB_PARAMS['error_linear'],[3,2])

        self.error_linear_front=np.reshape(CALIB_PARAMS['error_linear_front'],[3,2])
        self.error_linear_left=np.reshape(CALIB_PARAMS['error_linear_left'],[3,2])
        self.error_linear_right=np.reshape(CALIB_PARAMS['error_linear_right'],[3,2])

        self.tune_xyz=CALIB_PARAMS['tune_xyz']
        self.lm_max_value=CALIB_PARAMS['lm_max_value']

    def set_intrinsic(self,k):
        self.fx, self.fy, self.ppx, self.ppy=k

    def rotate_x(self,theta, point):
        # Convert theta to radians
        theta_rad = np.deg2rad(theta)

        # Rotation matrix for X-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta_rad), -np.sin(theta_rad)],
            [0, np.sin(theta_rad), np.cos(theta_rad)]
        ])

        # Apply the rotation
        rotated_point = np.matmul(rotation_matrix, point)

        return rotated_point
    def rotate_y(self,theta, point):
        # Convert theta to radians
        theta_rad = np.deg2rad(theta)

        # Rotation matrix for X-axis
        rotation_matrix = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])

        # Apply the rotation
        rotated_point = np.matmul(rotation_matrix, point)

        return rotated_point
    def convert_sensor_to_link(self,point,rotate_y,rotate_z,robot_mode,flag_tune=True):
        T_1=[32,-(20.13+55),36.9+24]
        T_2=[0,0,28]
        if type(robot_mode)==type(None):
            if abs(rotate_z)<3:
                mode="front"
            else:
                if rotate_z<0:
                    mode="left"
                else:
                    mode="right"
        else:
            mode =robot_mode
        print(f"calibration mode : {mode}_mode")
        rotate_point_x = self.rotate_x(rotate_y,point) + T_1
        rotate_point_y = self.rotate_y(-rotate_z,rotate_point_x) + T_2
        if flag_tune:
            ret_tune=self.tune_y_anlge_linear(rotate_point_y,rotate_y,mode)
            final_ret=copy.deepcopy(ret_tune)
        else:
            final_ret=copy.deepcopy(rotate_point_y)
        return final_ret
    def tune_y_anlge_linear(self,point,cur_yangle,mode="left"):
        # self.error_linear_front=np.reshape(CALIB_PARAMS['error_linear_front'],[3,2])
        # self.error_linear_left=np.reshape(CALIB_PARAMS['error_linear_left'],[3,2])
        # self.error_linear_right=np.reshape(CALIB_PARAMS['error_linear_right'],[3,2])
        if  mode=="front":
            a=np.array(self.error_linear_front)[:,0]
            b=np.array(self.error_linear_front)[:, 1]
        elif  mode=="left":
            a=np.array(self.error_linear_left)[:,0]
            b=np.array(self.error_linear_left)[:, 1]
        elif  mode=="right":            
            a=np.array(self.error_linear_right)[:,0]
            b=np.array(self.error_linear_right)[:, 1]
        conf_error = a*cur_yangle+b
        ret_point=point-conf_error
        return ret_point
    def convert_femto_2dto3d(self,X_2d,Y_2d,depth):
        fx=self.fx
        fy=self.fy
        ppx=self.ppx
        ppy=self.ppy
        target_point=[(X_2d-ppx)/fx*depth,(Y_2d-ppy)/fy*depth,depth]
        return target_point
    def convert_femto_2dtoBase3d(self,pose,rotate_y, rotate_z):
        fx=self.fx
        fy=self.fy
        ppx=self.ppx
        ppy=self.ppy

        X_2d = (pose[0]+pose[2])/2
        Y_2d = (pose[1]+pose[3])/2
        depth = pose[4]

        target_point=[(X_2d-ppx)/fx*depth,(Y_2d-ppy)/fy*depth,depth]

        link_point=self.convert_sensor_to_link(target_point,rotate_y,rotate_z,flag_tune=False)
        point_x,point_y,point_z=link_point

        #링크에서 카메라좌표계에서 로봇좌표계로 변환
        if self.init_angle:
            robot_x=point_z
            robot_y=-point_x
            robot_z=-point_y
        else:
            robot_x=point_x
            robot_y=point_z
            robot_z=-point_y

        base_femto_point=np.dot((self.link2base_robot),[robot_x,robot_y,robot_z,1])
        return base_femto_point

    def convert_femto_object2Height(self,pose,head_state):
        fx=self.fx
        fy=self.fy
        ppx=self.ppx
        ppy=self.ppy

        mid_X_2d = (pose[0]+pose[2])/2
        mid_Y_2d = (pose[1]+pose[3])/2
        mid_depth = pose[4]

        mid_target_point=[(mid_X_2d-ppx)/fx*mid_depth,(mid_Y_2d-ppy)/fy*mid_depth,mid_depth]
        mid_link_point=self.convert_sensor_to_link(mid_target_point,head_state[0],head_state[1])
        mid_point_x,mid_point_y,mid_point_z=mid_link_point
        #링크에서 카메라좌표계에서 로봇좌표계로 변환
        if self.init_angle:
            m_robot_x=mid_point_z
            m_robot_y=-mid_point_x
            m_robot_z=-mid_point_y
        else:

            m_robot_x=mid_point_x
            m_robot_y=mid_point_z
            m_robot_z=-mid_point_y


        base_X_2d = (pose[0]+pose[2])/2
        base_Y_2d = pose[3]
        base_depth = pose[4]

        base_target_point=[(base_X_2d-ppx)/fx*base_depth,(base_Y_2d-ppy)/fy*base_depth,base_depth]
        base_link_point=self.convert_sensor_to_link(base_target_point,head_state[0],head_state[1])
        base_point_x,base_point_y,base_point_z=base_link_point
	#링크에서 카메라좌표계에서 로봇좌표계로 변환
        if self.init_angle:
            b_robot_x=base_point_z
            b_robot_y=-base_point_x
            b_robot_z=-base_point_y
        else:

            b_robot_x=base_point_x
            b_robot_y=base_point_z
            b_robot_z=-base_point_y

        grasp_height = m_robot_z-b_robot_z
        return grasp_height




    def convert_femto_to_arm_range(self,X_2d,Y_2d,depth,cur_lift_position_mm,cur_robot,current_ry,current_rz):
        # self.load_config()
        print(f"current ry rz:{current_ry},{current_rz}")
        print(f"X_2d,Y_2d,depth:{X_2d},{Y_2d},{depth}")
        print(f"cur_lift_position_mm:{cur_lift_position_mm}")
        # convert 2d point to 3d point
        target_point=self.convert_femto_2dto3d(X_2d,Y_2d,depth)
        print(f"target_point: {target_point}")
        # mm단위로 변경
        if cur_lift_position_mm<1:
            cur_lift_position=cur_lift_position_mm*1000
        else:
            cur_lift_position=cur_lift_position_mm

        #femto 카메라에서 링크까지 변환
        link_point=self.convert_sensor_to_link(target_point,current_ry,current_rz)
        point_x,point_y,point_z=link_point
        print(f"link_point: {link_point}")

        #링크에서 카메라좌표계에서 로봇좌표계로 변환
        robot_x=point_z
        robot_y=-point_x
        robot_z=-point_y
        print(f"robot_xyz: {robot_x},{robot_y},{robot_z}")
        femto2base_point=np.dot((self.link2base_robot),[robot_x,robot_y,robot_z,1])

        #현재 리프트 위치에서 base 2 LMbase 관계 계산
        current_base2LMbase=copy.deepcopy(self.base2LMbase_robot)
        current_base2LMbase[2,3]=cur_lift_position
        #base 2 armrobot
        current_armrobot2base=np.matmul(current_base2LMbase,cur_robot[0:3].tolist()+[1])

        # distance : z_target_object-z_cur_arm
        print(f"base_femto_point[2]: {femto2base_point[2]}")
        print(f"base_cur_robot[2]: {current_armrobot2base[2]}")
        trans_x,trans_y,trans_z_position,trans_t=femto2base_point-current_armrobot2base

        # trans_z_position = move_lift+move_robot_z
        # 가장 낮은 lm 위치
        lm_min_value=self.base2LMbase_robot[2,3]
        lm_max_value=1000.0

        robot_z_min=268.5
        robot_z_max=510.0

        move_trans_lift=0
        move_trans_robot_z=0

        robot_max_range=250.0

        # calculate move_trans_lift / move_trans_robot_z
        if trans_z_position<0:
            move_trans_lift=trans_z_position
            move_trans_robot_z=0
        elif trans_z_position<robot_max_range:
            move_trans_lift=0
            move_trans_robot_z=trans_z_position
        elif robot_max_range<trans_z_position:
            move_trans_lift=trans_z_position-robot_max_range+10
            move_trans_robot_z=trans_z_position-move_trans_lift

        trans_lift_position_meter=move_trans_lift
        trans_z=move_trans_robot_z

        trans_x=trans_x+self.tune_xyz[0]
        trans_y=trans_y+self.tune_xyz[1]
        trans_z=trans_z+self.tune_xyz[2]

        return trans_lift_position_meter,[trans_x,trans_y,trans_z]
    def convert_head_to_base(self,X_2d,Y_2d,depth,cur_lift_position_mm,cur_robot,current_ry,current_rz):
        # self.load_config()
        print(f"current ry rz:{current_ry},{current_rz}")
        print(f"X_2d,Y_2d,depth:{X_2d},{Y_2d},{depth}")
        print(f"cur_lift_position_mm:{cur_lift_position_mm}")
        # convert 2d point to 3d point
        target_point=self.convert_femto_2dto3d(X_2d,Y_2d,depth)
        # mm단위로 변경
        if cur_lift_position_mm<1:
            cur_lift_position=cur_lift_position_mm*1000
        else:
            cur_lift_position=cur_lift_position_mm

        #femto 카메라에서 링크까지 변환
        link_point=self.convert_sensor_to_link(target_point,current_ry,current_rz)
        point_x,point_y,point_z=link_point

        #링크에서 카메라좌표계에서 로봇좌표계로 변환
        robot_x=point_z
        robot_y=-point_x
        robot_z=-point_y
        base_femto_point=np.dot((self.link2base_robot),[robot_x,robot_y,robot_z,1])
        print(f"base_femto_point: {base_femto_point}")
        #현재 리프트 위치에서 base 2 LMbase 관계 계산
        current_base2LMbase=copy.deepcopy(self.base2LMbase_robot)
        current_base2LMbase[2,3]=cur_lift_position
        #base 2 armrobot
        current_base2armrobot=np.matmul(current_base2LMbase,cur_robot[0:3].tolist()+[1])

        # distance : z_target_object-z_cur_arm
        print(f"base_femto_point[2]: {base_femto_point[2]}")
        print(f"base_cur_robot[2]: {current_base2armrobot[2]}")
        trans_z_position=(base_femto_point[2]-current_base2armrobot[2])

        # trans_z_position = move_lift+move_robot_z
        #리프트 이동 후, arm로봇의 이동거리 계산
        target_base2LMbase_robot=copy.deepcopy(current_base2LMbase)

        # 가장 낮은 lm 위치
        lm_min_value=self.base2LMbase_robot[2,3]
        lm_max_value=self.lm_max_value

        # calculate move Z
        move_lift_position=cur_lift_position+trans_z_position
        if move_lift_position<=lm_min_value:
            # lift position이 가장 낮은 위치보다 낮을 때
            # lift position은 0으로 두고
            trans_move_lift=0
        elif lm_max_value<=move_lift_position:
            # lift position이 가장 높은 위치보다 높을 때
            # lift position은 lm_max_value으로 두고
            trans_move_lift=lm_max_value-lm_min_value
        else:
            trans_move_lift=trans_z_position

        target_base2LMbase_robot[2,3]+=trans_move_lift

        #리프트 이동 후 베이스 기준 현재 로봇의 위치
        target_cur_robot=np.matmul(target_base2LMbase_robot,cur_robot[0:3].tolist()+[1])


        print(f"total trans_z_position_meter:{trans_z_position}")
        print(f"trans_lift_position:{trans_move_lift}")

        trans_x,trans_y,trans_z,trans_t=base_femto_point-target_cur_robot
        # print(f"trans_x:{trans_x},trans_y:{trans_y},trans_z:{trans_z}")
        trans_x=trans_x+self.tune_xyz[0]
        trans_y=trans_y+self.tune_xyz[1]
        trans_z=trans_z+self.tune_xyz[2]

        # print(f"modi_trans_x:{trans_x},modi_trans_y:{trans_y},modi_trans_z:{trans_z}")
        # print(f"cur_pose,{cur_robot[0]+trans_x} {cur_robot[2]+trans_y}  {cur_robot[2]+trans_z} ")
        return trans_move_lift,[trans_x,trans_y,trans_z]
    def convert_head_to_base_point(self,robot_mode,X_2d,Y_2d,depth,current_ry,current_rz):
        # self.load_config()
        print(f"current ry rz:{current_ry},{current_rz}")
        print(f"X_2d,Y_2d,depth:{X_2d},{Y_2d},{depth}")

        # convert 2d point to 3d point
        camera_point=self.convert_femto_2dto3d(X_2d,Y_2d,depth)
        print(f"target_point: {camera_point}")

        #femto 카메라에서 링크까지 변환
        link_point=self.convert_sensor_to_link(camera_point,current_ry,current_rz,robot_mode,flag_tune=False)
        point_x,point_y,point_z=link_point
        print(f"link_point: {link_point}")

        #링크에서 카메라좌표계에서 base 로봇좌표계로 변환
        matrix=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        link_to_head_point=np.dot(matrix,[point_x,point_y,point_z])
        base_femto_point=np.dot((self.link2base_robot),link_to_head_point.tolist()+[1])
        return base_femto_point
    def convert_head_to_base_point_with_joint0(self,X_2d,Y_2d,depth,current_ry,current_rz,arm_base_angle):
        # self.load_config()
        print(f"current ry rz:{current_ry},{current_rz}")
        print(f"X_2d,Y_2d,depth:{X_2d},{Y_2d},{depth}")

        # convert 2d point to 3d point
        camera_point=self.convert_femto_2dto3d(X_2d,Y_2d,depth)
        print(f"target_point: {camera_point}")

        #femto 카메라에서 링크까지 변환
        link_point=self.convert_sensor_to_link(camera_point,current_ry,current_rz,flag_tune=False)
        point_x,point_y,point_z=link_point
        print(f"link_point: {link_point}")

        #링크에서 카메라좌표계에서 base 로봇좌표계로 변환
        matrix=self.Matrix_arm_z_angle(current_rz)
        link_to_head_point=np.dot(matrix,[point_x,point_y,point_z])
        # matrix_4x4=np.eye(4)
        # matrix_4x4[0:3,0:3]=matrix
        # rot_link2base_robot=np.matmul(matrix_4x4,self.link2base_robot)
        base_femto_point=np.dot(self.link2base_robot,link_to_head_point.tolist()+[1])
        return base_femto_point

    def Matrix_arm_z_angle(self,rotate_rz=90):
        # base_point_x,base_point_y,base_point_z=camera_points
        if abs(rotate_rz)<1:#front
            # b_robot_x=base_point_z
            # b_robot_y=-base_point_x
            # b_robot_z=-base_point_y
            matrix=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        elif abs(rotate_rz-(90))<1:#left
            # b_robot_x=base_point_x
            # b_robot_y=base_point_z
            # b_robot_z=-base_point_y
            matrix=np.array([[1,0,0],[0,1,0],[0,0,-1]])
            
        elif abs(rotate_rz-(-90))<1:#right
            # b_robot_x=-base_point_x
            # b_robot_y=-base_point_z
            # b_robot_z=base_point_y
            matrix=np.array([[-1,0,0],[0,0,-1],[0,1,0]])
        else:
            rad=np.deg2rad((rotate_rz))
            front_matrix=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
            rot_matrix =np.array([[np.cos(rad),-np.sin(rad),0],[np.cos(rad),np.sin(rad),0],[0,0,1]])
            matrix=np.matmul(rot_matrix,front_matrix)
        return matrix