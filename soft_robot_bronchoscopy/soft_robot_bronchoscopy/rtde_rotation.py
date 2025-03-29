import rtde_control
import rtde_receive


class RobotArm:
    def __init__(self):
        self.rtde_frequency = 500.0
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.0.100",self.rtde_frequency, rtde_control.RTDEControlInterface.FLAG_CUSTOM_SCRIPT)
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.100")
        self.init_q = self.rtde_r.getActualQ()
        self.velocity = 0.05
        self.accel = 0.2
        # Servo specific variables
        self.dt = 1.0/500
        self.lookahead_time = 0.1
        self.gain = 300

    def rotate_wrist(self, prev_q, angle):
        # Rotates wrist of robot arm (about tool z-axis)
        new_q = prev_q[:]
        new_q[5] += angle

        self.rtde_c.moveJ(new_q, self.velocity, self.accel, True)  # Async movement in joint space, False is sync and blocks
        # Stop the movement
        #rtde_c.stopJ(0.5)
        return new_q

    def rotate_abt_axis(self, angle, axis, in_velocity=0.1):
        # Rotates tool on UR5e robot arm about specified tool axis

        self.init_TCP = self.rtde_r.getActualTCPPose()
        if axis != 3 and axis != 4 and axis != 5:
            print('Axis must be 3, 4, or 5 corresponding to respective x- y- z-')
            return self.init_TCP
        p_from_to = [0, 0, 0, 0, 0, 0]
        p_from_to[axis] = angle
        resulting_pose = self.rtde_c.poseTrans(self.init_TCP, p_from_to)
        self.rtde_c.moveL(resulting_pose, self.velocity, in_velocity)
        pose = self.rtde_r.getActualTCPPose()

        return pose

    def translate_axis(self, dists, axis):
        # incrementally translate in specific tool axis

        self.init_TCP = self.rtde_r.getActualTCPPose()
        p_from_to = [0, 0, 0, 0, 0, 0]
        if len(axis) > 1:
            c = 0
            for e in axis:
                if e != 0 and e != 1 and e != 2:
                    print('Axis must be 0, 1, or 2 corresponding to respective x- y- z-')
                    return self.init_TCP
                p_from_to[e] += dists[c]
                c += 1
            resulting_pose_T = self.rtde_c.poseTrans(self.init_TCP, p_from_to)
            self.rtde_c.moveL(resulting_pose_T, self.velocity, self.accel)
        else:
            if axis != 0 and axis != 1 and axis != 2:
                print('Axis must be 0, 1, or 2 corresponding to respective x- y- z-')
                return self.init_TCP
            p_from_to = [0, 0, 0, 0, 0, 0]
            p_from_to[axis] += dists
            resulting_pose_T = self.rtde_c.poseTrans(self.init_TCP, p_from_to)
            self.rtde_c.moveL(resulting_pose_T, self.velocity, self.accel)

        return resulting_pose_T

    def robot_return(self):
        # Move back to initial position
        self.rtde_c.moveJ(self.init_q)

    def robot_close(self):
        self.rtde_c.stopScript()
