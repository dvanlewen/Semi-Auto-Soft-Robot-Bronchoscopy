import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from direct.showbase.ShowBase import ShowBase, WindowControls, exitfunc
from panda3d.core import NodePath, LQuaternionf, TextNode, InputDevice, loadPrcFileData, Vec3, WindowProperties, MouseWatcher, MouseAndKeyboard, ButtonThrower, Camera, DirectionalLight, PointLight, PerspectiveLens, OrthographicLens, DrawableRegion
from direct.actor.Actor import Actor
from direct.task import Task
import time
import numpy as np
import math
from panda3d.core import Quat
from panda3d.core import LQuaternionf, CollisionSphere, CollisionNode, CollisionHandlerPusher
from bronchoscopy_interfaces.msg import DaqOutput
from std_msgs.msg import Bool
import scipy
from panda3d.core import DisplayRegion, LPoint3f, LVecBase3f, GraphicsOutput, LineSegs, Filename, PNMImage, Texture, Filename, VBase4, LVector3, Spotlight, Texture
from direct.directtools.DirectManipulation import DirectManipulationControl, ObjectHandles, drawBox
import os

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
        self.counter = 0
        self.camera.setPosHpr(0,200,-500,90,0,0)

        matlabmat = scipy.io.loadmat('C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\newPaths1.mat', mat_dtype=True)
        self.matpath = matlabmat['xPath_Left2']
        path = LineSegs()
        path.setThickness(5)
        path.setColor(255,0,0,0.75)
        path_origin = rotate_vector_by_q(q_orient,[self.matpath[0][0],self.matpath[0][1],self.matpath[0][2]])
        path.moveTo(path_origin[0],path_origin[1],path_origin[2])
        for point in self.matpath:
            point = rotate_vector_by_q(q_orient, point)
            path.drawTo(point[0],point[1],point[2])
        pathnode = path.create(True)
        pathnodes = NodePath(pathnode)
        pathnodes.reparentTo(self.render)

        #create second path for second viewpoint
        self.render2 = NodePath("render2")
        path2 = LineSegs()
        path2.setThickness(5)
        path2.setColor(255,0,0,0.75)
        path_origin2 = rotate_vector_by_q(q_orient,[self.matpath[0][0],self.matpath[0][1],self.matpath[0][2]])
        path2.moveTo(path_origin2[0],path_origin2[1],path_origin2[2])
        for point in self.matpath:
            point = rotate_vector_by_q(q_orient, point)
            path2.drawTo(point[0],point[1],point[2])
        pathnode2 = path2.create(True)
        pathnodes2 = NodePath(pathnode2)
        pathnodes2.reparentTo(self.render2)

        myTexture = self.loader.loadTexture("texture3.png")
        myTexture2 = self.loader.loadTexture("texture3.png")

        # how to create a animation and an actor
        self.Cath = Actor("Curv2", {"anim1": "Curv2-A", "anim2": "Curv2-A2"})
        self.Cath.setTexture(myTexture)
        Lung = self.loader.loadModel("Lung")
        Lung.setPos(0,0,0)
        Lung.setTransparency(1)
        Lung.setColorScale(1, 1, 1, .5)
        Lung.reparentTo(self.render)

        self.Cath.reparentTo(self.render)
        self.Cath.setHpr(90, 0, 0)
        self.Cath.setScale(.4,.4,.4)

        #self.Cath.loop("anim1"
        self.Cath.pose('anim1', 0)



        # second view how to create a animation and an actor
        self.Cath2 = Actor("Curv2", {"anim1": "Curv2-A", "anim2": "Curv2-A2"})
        self.Cath2.setTexture(myTexture2)
        Lung2 = self.loader.loadModel("Lung")
        Lung2.setPos(0,0,0)
        # Lung2.setTransparency(1)
        Lung2.setColorScale(1, 1, 1, .5)
        Lung2.reparentTo(self.render2)
        self.Cath2.reparentTo(self.render2)
        self.Cath2.setHpr(90, 0, 0)
        self.Cath2.setScale(.4,.4,.4)
        self.Cath2.pose('anim1', 0)

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

        # Accept button events of the first connected gamepad
        #self.accept("gamepad-back", exit)
        #self.accept("gamepad-start", exit)
        #self.accept("gamepad-face_a", self.action, extraArgs=["face_a"])
        #self.accept("gamepad-face_a-up", self.actionUp)
        #self.accept("gamepad-face_b", self.action, extraArgs=["face_b"])
        #self.accept("gamepad-face_b-up", self.actionUp)
        #self.accept("gamepad-face_y", self.action, extraArgs=["face_y"])
        #self.accept("gamepad-face_y-up", self.actionUp)

        # Disable the default mouse-camera controls since we need to handle
        # our own camera controls.

        # dlight = DirectionalLight('dlight')
        # dlight.setColor((1, 1, 1, 1))
        # dlnp = render.attachNewNode(dlight)
        # dlnp.setHpr(0, -60, 0)
        # render.setLight(dlnp)

        base.cam.setPosHpr(150,-600, -120,0,0,0) # default camera is called base.cam; setting starting position of model in the window
        model2 = self.Cath2
        my_cam2 = Camera("cam2")
        self.my_camera2 = self.render2.attachNewNode(my_cam2)
        self.my_camera2.setName("camera2")
        lens = PerspectiveLens()
        lens.setFilmSize(10,10)
        # near_dist = 5
        # far_dist = 60
        # lens.setNearFar(near_dist, far_dist)
        lens.setFov(60)
        self.my_camera2.node().setLens(lens)
        self.my_camera2.setPosHpr(0,0, -100,0,-90,0) # second camera view AKA POV of cath; setting starting position of model in the window
        dr = base.camNode.getDisplayRegion(0)

        # # mouse control in the left section but only controls left side
        myDR = base.win.makeDisplayRegion(0, 0.5, 0, 1)
        base.mouseWatcherNode.setDisplayRegion(myDR)
        myMouseWatcher = MouseWatcher()
        mw = base.mouseWatcher.getParent().attachNewNode(myMouseWatcher)
        myMouseWatcher.setDisplayRegion(myDR)
        btn_thrower = ButtonThrower("my_btn_thrower")
        mw.attachNewNode(btn_thrower)
        if myMouseWatcher.hasMouse():
            mpos = myMouseWatcher.getMouse()
            # print(mpos)

        # dr.setActive(0)  # Or leave it (dr.setActive(1))
        self.window = dr.getWindow()
        self.window.setClearColor((0.2, 0.2, 0.2, 0))  #set left display region background color to black
        self.window.setClearColorActive(True)
        props = WindowProperties()
        props.setSize(1800, 900)
        self.window.requestProperties(props)
        self.dr1 = self.window.makeDisplayRegion(0, 0.5, 0, 1)  #create left display window region
        self.dr1.setSort(dr.getSort())
        self.dr2 = self.window.makeDisplayRegion(0.5, 1, 0, 1)  #create right display window region
        self.dr2.setClearColor((0, 0, 0.5, 0))  #set left display region background color to blue
        self.dr2.setClearColorActive(True)
        self.dr2.setSort(dr.getSort())
        self.dr2.setCamera(self.my_camera2)


        #Collision Nodes for Contact between robot and lung wall
        # self.colliderNode = CollisionNode("player")   # In the "__init__" method:
        # base.pusher = CollisionHandlerPusher()         # In the "__init__" method:

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

        #self.Tip.setPos(tip_position[0], tip_position[1], tip_position[2])
        tipquat = LQuaternionf(tip_ori[3], tip_ori[0], tip_ori[1], tip_ori[2])
        z_tip = rotate_vector_by_q(tip_ori, [0,0,3])
        x_tip = rotate_vector_by_q(tip_ori, [3,0,0])
        y_tip = rotate_vector_by_q(tip_ori, [0,3,0])
        #q = LQuaternionf(np.sin(2.79/2)*1, np.sin(np.pi/2)*0, np.sin(np.pi/2)*0,np.cos(2.79/2))
        #self.Tip.setQuat(tipquat)

        #Deg = Pressurec1+c2
        frame = 0.2 * angle

        fwdvec = self.Cath2.getQuat().getForward()
        upvec = self.Cath2.getQuat().getUp()
        self.my_camera2.setPos(position[0], position[1], position[2])  # cath POV
        self.my_camera2.setQuat(quat)  # cath POV
        self.my_camera2.lookAt(fwdvec, upvec)
        self.my_camera2.setHpr(0, 270, 180)

        # # # point light for ROBOT
        # plight = PointLight('plight')
        # plight.setColor((0.15, 0.15, 0.15, 0.6))
        # plight.attenuation = (1, 0, 0.00000001)
        # # plight.setColorTemperature(5000) 
        # plnp = self.my_camera2.attachNewNode(plight)
        # plnp.setPos(-position[0], -position[1], position[2])
        # plnp.setQuat(quat)  # cath POV
        # plnp.setHpr(0, 90, 0)
        # plnp.lookAt(fwdvec, upvec)
        # self.render2.setLight(plnp)

        slight = Spotlight('slight')
        slight.setColor((0.05, 0.05, 0.05, 0.1))
        # # slight.attenuation = (1, 0, 0.0009)   #use this for base data
        slight.attenuation = (1, 0, 0.005)  #use this for tip data
        slight.setColorTemperature(4500)
        lens = PerspectiveLens()
        slight.setLens(lens)
        lens.setFilmSize(100,100)
        # lens.setFov(5000)  # use this for base data 
        lens.setFov(800)  # use this for tip data 
        slnp = self.render2.attachNewNode(slight)
        slnp.setPos(-position[0], -position[1], position[2])
        slnp.setQuat(quat)  # cath POV
        slnp.setHpr(0, 90, 0)
        slnp.lookAt(fwdvec, upvec)
        self.render2.setLight(slnp)


        # used to take and save screenshots at every desired second(s), saves it to the C:\dev\ros2_ws
        given_sampling_rate = 0.1  #excel publishes data every 0.1 seconds
        want_sampling_rate = 5 # want to take a screenshot every 10 seconds
        step_scale = want_sampling_rate/given_sampling_rate
        if self.counter % step_scale == 0:
            # cwd = os.getcwd()
            # print(cwd)
            # base.movie(namePrefix='image' + str(self.counter), duration=1, fps=1, format='png', source=self.window)
          
            # texture = self.dr1.getScreenshot()
            # filename = self.dr1.makeScreenshotFilename('test' + str(self.counter))
            # self.dr1.saveScreenshot(filename)

            # base.screenshot(namePrefix='screenshot', defaultFilename=False, source=self.window)  why doesnt this save i am sad 
            
            print("counter:", self.counter)
        self.counter += 1

        # self.colliderNode.addSolid(CollisionSphere(-position[0], -position[1], position[2], 30))  # Add a collision-sphere centred on (0, 0, 0), and with a radius of 0.3
        # collider_cath1 = self.Cath.attachNewNode(self.colliderNode)
        # collider_cath1.show()
        # The pusher wants a collider, and a NodePath that should be moved by that collider's collisions.
        # In this case, we want our player-Actor to be moved.
        # base.pusher.addCollider(collider_cath1, self.Cath)
        # The traverser wants a collider, and a handler that responds to that collider's collisions
        # base.cTrav.addCollider(collider_cath1, base.pusher)

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

        self.simulation.update_probe(base_pos_pframe, q_base2patient, self.angle, tip_pos_pframe,quaternion_inverse(q_patient2tip))
        self.get_logger().info('x="%f", y="%f", z="%f"' % (base_pos_pframe[0], base_pos_pframe[1]+200, base_pos_pframe[2]))

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

    #panda_simulation = MyGame()

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
#    try:
#        panda_subscriber.simulation.run()
#        rclpy.spin(panda_subscriber)
#    except KeyboardInterrupt:
#        pass
#    finally:
#        panda_subscriber.destroy_node()
#        rclpy.shutdown()
if __name__ == '__main__':
    main()
