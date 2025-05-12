import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath, LQuaternionf, TextNode, InputDevice, loadPrcFileData, Vec3
from direct.actor.Actor import Actor
from direct.task import Task
import numpy as np
from panda3d.core import Quat
from bronchoscopy_interfaces.msg import DaqOutput
from std_msgs.msg import Bool
import scipy
from panda3d.core import LineSegs, LPoint3f

confVars = """
win-size 1280 720
window-title Yo
"""

loadPrcFileData("", confVars)

confVars2 = """
win-size 1280 720
window-title hi
"""
loadPrcFileData("", confVars2)


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    """
    return np.array((-quaternion[0], -quaternion[1],
                     -quaternion[2], quaternion[3]), dtype=np.float64)

def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> numpy.allclose(quaternion_multiply(q0, q1), [0, 0, 0, 1])
    True

    """
    return quaternion_conjugate(quaternion) / np.dot(quaternion, quaternion)

def rotate_vector_by_q(q, vec):
    """
    Rotate a 3D vector by the quaternion, q
    """
    if len(vec) == 3:
        p = np.append(np.array(vec), 0)
    else:
        p = vec
    q_pr = [-q[0], -q[1], -q[2], q[3]]
    H1 = [q[3]*p[0] + q[0]*p[3] + q[1]*p[2] - q[2]*p[1],
          q[3]*p[1] - q[0]*p[2] + q[1]*p[3] + q[2]*p[0],
          q[3]*p[2] + q[0]*p[1] - q[1]*p[0] + q[2]*p[3],
          q[3]*p[3] - q[0]*p[0] - q[1]*p[1] - q[2]*p[2]]

    H2 = [H1[3]*q_pr[0] + H1[0]*q_pr[3] + H1[1]*q_pr[2] - H1[2]*q_pr[1],
          H1[3]*q_pr[1] - H1[0]*q_pr[2] + H1[1]*q_pr[3] + H1[2]*q_pr[0],
          H1[3]*q_pr[2] + H1[0]*q_pr[1] - H1[1]*q_pr[0] + H1[2]*q_pr[3],
          H1[3]*q_pr[3] - H1[0]*q_pr[0] - H1[1]*q_pr[1] - H1[2]*q_pr[2]]

    if len(vec) == 3:
        p_pr = [H2[0], H2[1], H2[2]]
    else:
        p_pr = [H2[0], H2[1], H2[2], H2[3]]

    return p_pr


# Quaternion for 180 degree rotation of the path
# Aurora and Panda nodes are based off stl file which is rotated from original paths found in MATLAB
q_orient = [np.sin(np.pi/2)*0, np.sin(np.pi/2)*0, np.sin(np.pi/2)*1, np.cos(np.pi/2)]

class MyGame(ShowBase):
    def __init__(self):
        super().__init__()
        self.camera.setPosHpr(0,200,-500,90,0,0)

        matlabmat = scipy.io.loadmat('C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\newPaths1.mat', mat_dtype=True)
        self.matpath = matlabmat['xPath_Left5']
        path = LineSegs()
        path.setThickness(5)
        path.setColor(255,0,0,0.75)
        path_origin = rotate_vector_by_q(q_orient,[self.matpath[0][0],self.matpath[0][1],self.matpath[0][2]])  #[self.matpath[0][0], self.matpath[0][1], self.matpath[0][2]]  #
        path.moveTo(path_origin[0],path_origin[1],path_origin[2])
        for point in self.matpath:
            point = rotate_vector_by_q(q_orient, point)
            path.drawTo(point[0],point[1],point[2])
        pathnode = path.create(True)
        pathnodes = NodePath(pathnode)
        pathnodes.reparentTo(self.render)

        myTexture = self.loader.loadTexture("texture3.png")

        # how to create a animation and an actor
        self.baseT = LineSegs()
        self.baseT.setThickness(2)
        self.baseT.setColor(0,0,255,0.75)
        self.baseT.moveTo(0,0,0)
        self.baseT.drawTo(1,1,1)
        self.baseTnode = self.baseT.create(True)
        self.baseTnodes = NodePath(self.baseTnode)
        self.baseTnodes.reparentTo(self.render)
        self.base_x = LineSegs()
        self.base_x.setThickness(2)
        self.base_x.setColor(255,0,0,0.75)
        self.base_x.moveTo(0,0,0)
        self.base_x.drawTo(1,1,1)
        self.base_xnode = self.base_x.create(True)
        self.base_xnodes = NodePath(self.base_xnode)
        self.base_xnodes.reparentTo(self.render)
        self.base_y = LineSegs()
        self.base_y.setThickness(2)
        self.base_y.setColor(0,255,0,0.75)
        self.base_y.moveTo(0,0,0)
        self.base_y.drawTo(1,1,1)
        self.base_ynode = self.base_y.create(True)
        self.base_ynodes = NodePath(self.base_ynode)
        self.base_ynodes.reparentTo(self.render)

        self.tip = LineSegs()
        self.tip.setThickness(2)
        self.tip.setColor(0,0,255,0.75)
        self.tip.moveTo(0,0,0)
        self.tip.drawTo(1,1,1)
        self.tipnode = self.tip.create(True)
        self.tipnodes = NodePath(self.tipnode)
        self.tipnodes.reparentTo(self.render)
        self.tip_x = LineSegs()
        self.tip_x.setThickness(2)
        self.tip_x.setColor(255,0,0,0.75)
        self.tip_x.moveTo(0,0,0)
        self.tip_x.drawTo(1,1,1)
        self.tip_xnode = self.tip_x.create(True)
        self.tip_xnodes = NodePath(self.tip_xnode)
        self.tip_xnodes.reparentTo(self.render)
        self.tip_y = LineSegs()
        self.tip_y.setThickness(2)
        self.tip_y.setColor(0,255,0,0.75)
        self.tip_y.moveTo(0,0,0)
        self.tip_y.drawTo(1,1,1)
        self.tip_ynode = self.tip_y.create(True)
        self.tip_ynodes = NodePath(self.tip_ynode)
        self.tip_ynodes.reparentTo(self.render)
        self.Cath = self.loader.loadModel("Robot")  #Curv2", {"anim1": "Curv2-A", "anim2": "Curv2-A2"})
        self.Tip = self.loader.loadModel("Robot")
        self.Cath.setTexture(myTexture)
        #self.render.clearLight(plnp)
        Lung = self.loader.loadModel("Lung")
        Lung.setPos(0,0,0)
        rotatedlung = LQuaternionf(0,0,0,1)
        #Lung.setQuat(rotatedlung)
        Lung.setTransparency(1)
        Lung.setColorScale(1, 1, 1, .5)
        Lung.reparentTo(self.render)

        self.Cath.reparentTo(self.render)
        self.Tip.reparentTo(self.render)
        #self.Cath.setHpr(90, 0, 0)
        #self.Cath.setScale(.4,.4,.4)

        #self.Cath.loop("anim1")
        #self.Cath.pose('anim1', 0)

        self.x = 0
        self.z = 0
        self.taskMgr.add(self.update, "update")

        self.gamepad = None
        devices = self.devices.getDevices(InputDevice.DeviceClass.gamepad)
        if devices:
            self.connect(devices[0])

        # Accept device dis-/connection events
        self.accept("connect-device", self.connect)
        self.accept("disconnect-device", self.disconnect)

        self.accept("escape", exit)

        # Disable the default mouse-camera controls since we need to handle
        # our own camera controls.

    def connect(self, device):
        """Event handler that is called when a device is discovered."""

        # We're only interested if this is a gamepad and we don't have a
        # gamepad yet.
        if device.device_class == InputDevice.DeviceClass.gamepad and not self.gamepad:
            print("Found %s" % (device))
            self.gamepad = device

            # Enable this device to ShowBase so that we can receive events.
            # We set up the events with a prefix of "gamepad-".
            self.attachInputDevice(device, prefix="gamepad")

            # Hide the warning that we have no devices.
            #self.lblWarning.hide()

    def disconnect(self, device):
        """Event handler that is called when a device is removed."""

        if self.gamepad != device:
            # We don't care since it's not our gamepad.
            return

        # Tell ShowBase that the device is no longer needed.
        print("Disconnected %s" % (device))
        self.detachInputDevice(device)
        self.gamepad = None

        # Do we have any other gamepads?  Attach the first other gamepad.
        devices = self.devices.getDevices(InputDevice.DeviceClass.gamepad)
        if devices:
            self.connect(devices[0])
        #else:
            # No devices.  Show the warning.
            #self.lblWarning.show()

    def reset(self):
        """Reset the camera to the initial position."""

        self.camera.setPosHpr(0, -200, 10, 0, 0, 0)

    def moveTask(self, task):
        dt = base.clock.dt

        if not self.gamepad:
            return task.cont

        strafe_speed = 85
        vert_speed = 50
        turn_speed = 100

        # If the left stick is pressed, we will go faster.
        lstick = self.gamepad.findButton("lstick")
        if lstick.pressed:
            strafe_speed *= 2.0

        # we will use the first found gamepad
        # Move the camera left/right
        strafe = Vec3(0)
        left_x = self.gamepad.findAxis(InputDevice.Axis.left_x)
        left_y = self.gamepad.findAxis(InputDevice.Axis.left_y)
        strafe.x = left_x.value
        strafe.y = left_y.value

        # Apply some deadzone, since the sticks don't center exactly at 0
        if strafe.lengthSquared() >= 0.01:
            self.camera.setPos(self.camera, strafe * strafe_speed * dt)

        # Use the triggers for the vertical position.
        trigger_l = self.gamepad.findAxis(InputDevice.Axis.left_trigger)
        trigger_r = self.gamepad.findAxis(InputDevice.Axis.right_trigger)
        lift = trigger_r.value - trigger_l.value
        self.camera.setZ(self.camera.getZ() + (lift * vert_speed * dt))

        # Move the camera forward/backward
        right_x = self.gamepad.findAxis(InputDevice.Axis.right_x)
        right_y = self.gamepad.findAxis(InputDevice.Axis.right_y)

        # Again, some deadzone
        if abs(right_x.value) >= 0.1 or abs(right_y.value) >= 0.1:
            self.camera.setH(self.camera, turn_speed * dt * -right_x.value)
            self.camera.setP(self.camera, turn_speed * dt * right_y.value)

            # Reset the roll so that the camera remains upright.
            self.camera.setR(0)

        return task.cont

    def update(self, task):
        return task.cont

    def update_probe(self, position, orientation, angle, tip_position, tip_ori):
        self.Cath.setPos(position[0], position[1], position[2])
        quat = LQuaternionf(orientation[3], orientation[0], orientation[1], orientation[2])
        z_base = rotate_vector_by_q(orientation, [0,0,5])
        x_base = rotate_vector_by_q(orientation, [5,0,0])
        y_base = rotate_vector_by_q(orientation, [0,5,0])
        #q = LQuaternionf(np.sin(2.79/2)*1, np.sin(np.pi/2)*0, np.sin(np.pi/2)*0,np.cos(2.79/2))
        self.Cath.setQuat(quat)
        self.baseT.moveTo(position[0], position[1], position[2])
        self.baseT.drawTo(position[0]+z_base[0], position[1]+z_base[1], position[2]+z_base[2])
        self.baseTnode = self.baseT.create(True)
        self.baseTnodes = NodePath(self.baseTnode)
        self.baseTnodes.reparentTo(self.render)
        self.base_x.moveTo(position[0], position[1], position[2])
        self.base_x.drawTo(position[0]+x_base[0], position[1]+x_base[1], position[2]+x_base[2])
        self.base_xnode = self.base_x.create(True)
        self.base_xnodes = NodePath(self.base_xnode)
        self.base_xnodes.reparentTo(self.render)
        self.base_y.moveTo(position[0], position[1], position[2])
        self.base_y.drawTo(position[0]+y_base[0], position[1]+y_base[1], position[2]+y_base[2])
        self.base_ynode = self.base_y.create(True)
        self.base_ynodes = NodePath(self.base_ynode)
        self.base_ynodes.reparentTo(self.render)

        self.Tip.setPos(tip_position[0], tip_position[1], tip_position[2])
        tipquat = LQuaternionf(tip_ori[3], tip_ori[0], tip_ori[1], tip_ori[2])
        z_tip = rotate_vector_by_q(tip_ori, [0,0,3])
        x_tip = rotate_vector_by_q(tip_ori, [3,0,0])
        y_tip = rotate_vector_by_q(tip_ori, [0,3,0])
        #q = LQuaternionf(np.sin(2.79/2)*1, np.sin(np.pi/2)*0, np.sin(np.pi/2)*0,np.cos(2.79/2))
        self.Tip.setQuat(tipquat)
        self.tip.moveTo(tip_position[0], tip_position[1], tip_position[2])
        self.tip.drawTo(tip_position[0]+z_tip[0], tip_position[1]+z_tip[1], tip_position[2]+z_tip[2])
        self.tipnode = self.tip.create(True)
        self.tipnodes = NodePath(self.tipnode)
        self.tipnodes.reparentTo(self.render)
        self.tip_x.moveTo(tip_position[0], tip_position[1], tip_position[2])
        self.tip_x.drawTo(tip_position[0]+x_tip[0], tip_position[1]+x_tip[1], tip_position[2]+x_tip[2])
        self.tip_xnode = self.tip_x.create(True)
        self.tip_xnodes = NodePath(self.tip_xnode)
        self.tip_xnodes.reparentTo(self.render)
        self.tip_y.moveTo(tip_position[0], tip_position[1], tip_position[2])
        self.tip_y.drawTo(tip_position[0]+y_tip[0], tip_position[1]+y_tip[1], tip_position[2]+y_tip[2])
        self.tip_ynode = self.tip_y.create(True)
        self.tip_ynodes = NodePath(self.tip_ynode)
        self.tip_ynodes.reparentTo(self.render)

        #Deg = Pressurec1+c2
        frame = 0.2 * angle
        #self.Cath.pose('anim1', frame)

    def update_inputs(self, task):
        # Update input devices
        self.device_manager.update()

        # Optionally, you can add code to update mouse position here
        self.update_mouse_position()

        return task.cont

class PandaSubscriber(Node):
    def __init__(self):
        super().__init__('panda_subscriber')
        self.subscription = self.create_subscription(
            PoseArray,
            'aurora_sensor',
            self.panda_callback,
            10)
        #self.publisher = self.create_publisher(Twist, 'UR5e_motion', 10)
        self.subscription2 = self.create_subscription(
            DaqOutput,
            'pressure',
            self.angle_callback,
            10)
        self.subscription3 = self.create_subscription(Bool, 'cameracontrol', self.toggle_camcontrol, 10)
        self.angle = 0
        self.pressure = 0
        self.activated2 = False
        self.simulation = MyGame()  # prevent unused variable warning

    def panda_callback(self, msg):
        base_pos_pframe = [msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z]
        q_base2patient = [msg.poses[0].orientation.x,
                          msg.poses[0].orientation.y,
                          msg.poses[0].orientation.z,
                          msg.poses[0].orientation.w]

        tip_pos_pframe = [msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z]
        q_patient2tip = [msg.poses[1].orientation.x,
                         msg.poses[1].orientation.y,
                         msg.poses[1].orientation.z,
                         msg.poses[1].orientation.w]

        self.simulation.update_probe(base_pos_pframe, q_base2patient, self.angle, tip_pos_pframe, quaternion_inverse(q_patient2tip))
        #self.get_logger().info('x="%f", y="%f", z="%f"' % (base_pos_pframe[0], base_pos_pframe[1], base_pos_pframe[2]))

    def angle_callback(self, msg):
        self.angle = msg.bend_angle
        self.pressure = msg.pressure
        if self.pressure < 40:
            self.angle = 0
        else:
            self.angle = 0.0007*self.pressure**2 + (-0.0527) * self.pressure + 0.8351
            self.get_logger().info('x="%f"' % (self.angle))

    def toggle_camcontrol(self, msg):
        self.activated2 = msg.data
        if self.activated2:
            self.simulation.disableMouse()
            self.simulation.taskMgr.add(self.simulation.moveTask, "movement update task")
            self.simulation.accept("gamepad-face_x", self.simulation.reset)
        else:
            self.simulation.taskMgr.remove("movement update task")


def main(args=None):
    rclpy.init(args=args)
    panda_subscriber = PandaSubscriber()

    try:
        # Run ROS in a separate thread
        ros_thread = threading.Thread(target=rclpy.spin, args=(panda_subscriber,))
        ros_thread.start()
        # Run Panda3D simulation
        panda_subscriber.simulation.run()
    except KeyboardInterrupt:
        pass
    finally:
        panda_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
