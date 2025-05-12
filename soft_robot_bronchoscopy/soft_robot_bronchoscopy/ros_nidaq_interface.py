"""
Node for interfacing with stepper motor and pressure regulator connected to NI DAQ
"""
import rclpy
from rclpy.node import Node
import nidaqmx
from nidaqmx.constants import LineGrouping
import time
from bronchoscopy_interfaces.msg import DaqInput
from bronchoscopy_interfaces.srv import SensePressure
from bronchoscopy_interfaces.msg import DaqOutput
import numpy as np

class Stepper:

    def __init__(self):
        self.stepper_out = nidaqmx.Task()
        port = 'Dev1/port1/line0:1'
        self.stepper_out.do_channels.add_do_chan(port, line_grouping=LineGrouping.CHAN_PER_LINE)
        self.stepper_out.start()
        self.stepper_pos = 0

    def step_backward(self):
        """
        Steps a stepper motor forward one step and updates the position
        """

        self.stepper_out.write([True, True])
        time.sleep(0.01)
        self.stepper_out.write([False, True])
        time.sleep(0.1)
        self.stepper_pos += 1

    def step_forward(self):
        """
        Steps a stepper motor backward one step and updates the position
        """

        self.stepper_out.write([True, False])
        time.sleep(0.1)
        self.stepper_out.write([False, False])
        time.sleep(0.1)
        self.stepper_pos -= 1

    def stepper_close(self):
        self.stepper_out.stop()
        self.stepper_out.close()


UNIT_PRESSURE = 0.025  # in Volts, corresponds to 1 kpa (for SMC ITV0030-2S regulator)
class pressure_pumps:

    def __init__(self):
        self.press_readout = nidaqmx.Task()
        self.press_in = nidaqmx.Task()
        port_out = 'Dev1/ai0'
        port_in = 'Dev1/ao0'
        self.press_readout.ai_channels.add_ai_voltage_chan(port_out, terminal_config=nidaqmx.constants.TerminalConfiguration(10083))
        self.press_in.ao_channels.add_ao_voltage_chan(port_in)

        # From ITV0030-2S Datasheet
        self.vMin = 0
        self.vMax = 5
        self.readvMin = 1
        self.readvMax = 5
        self.pMax = 500
        self.volt_in = 0

    def increase_pressure(self):
        """
        Increases pressure by a unit value
        """

        self.volt_in += 2*UNIT_PRESSURE
        if self.volt_in > self.vMax - 20 * UNIT_PRESSURE:
            raise ValueError('VOLTAGE IS TOO HIGH')
        self.press_in.write(self.volt_in)

    def decrease_pressure(self):
        """
        Decreases pressure by a unit value
        """

        self.volt_in -= UNIT_PRESSURE
        if self.volt_in < self.vMin + UNIT_PRESSURE:
            raise ValueError('VOLTAGE IS TOO LOW')
        self.press_in.write(self.volt_in)

    def set_pressure(self, pressure_input):
        """
        Inputs a set pressure to regulator
        """
        self.volt_in = pressure_input * (self.vMax - self.vMin)/(self.pMax - 0)

        if self.volt_in < 0:
            raise ValueError('VOLTAGE IS TOO LOW')
        if self.volt_in > self.vMax - 50 * UNIT_PRESSURE:
            raise ValueError('VOLTAGE IS TOO HIGH')
        self.press_in.write(self.volt_in)

    def log_pressure(self):
        v_sense = self.press_readout.read()
        p_sense = ((self.pMax - 0) / (self.readvMax - self.readvMin)) * (v_sense - self.readvMin)
        return v_sense, p_sense

    def pressure_pumps_close(self):
        self.press_in.write(0)
        self.press_readout.close()
        self.press_in.close()

def cc_predict(total_pressure):
    """
    Calculates bending angle and tip position relative to base using constant curvature assumption
    """
    L = 35
    if total_pressure > 40:
        total_bending_angle = 0.6675*np.exp(0.02116*total_pressure)
    else:
        total_bending_angle = 0
    total_bending_angle = np.radians(total_bending_angle)
    if total_bending_angle == 0:
        r_o_c = 0
        tip_z = L
    else:
        r_o_c = L / total_bending_angle
        tip_z = r_o_c * np.sin(total_bending_angle)
    tip_y = r_o_c * (1 - np.cos(total_bending_angle))
    tip_pos_bframe = np.array([0, tip_y, tip_z])
    return tip_pos_bframe, total_bending_angle

class DaqInterface(Node):
    def __init__(self):
        super().__init__('nidaq_interface')
        self.subscriber = self.create_subscription(DaqInput, 'daq_input', self.regulator_callback, 10)
        self.publisher = self.create_publisher(DaqOutput, 'pressure', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.publish_pressure)
        self.stepper = Stepper()
        self.regulator = pressure_pumps()
        self.srv = self.create_service(SensePressure, 'sense_pressure', self.send_pressure)
        self.get_logger().info("NI DAQ Connected")
        self.total_steps = 0
        self.cont_dist = [900, 900]

    def regulator_callback(self, msg):
        """
        Subscriber callback for inputs into pressure regulator (steering DOF)
        and stepper motor (insertion DOF)
        """
        pressure = msg.pressure
        steps = msg.steps
        apply_pressure = True
        self.get_logger().info(str(msg.contour_distance))
        self.cont_dist[0] = msg.contour_distance[0]
        self.cont_dist[1] = msg.contour_distance[1]
        if self.cont_dist != [0,0] and pressure == 0:  # Maintains constant bend angle while inserting
            _, sensed_p = self.regulator.log_pressure()
            pressure = sensed_p
            apply_pressure = False

        forward = True
        if steps < 0:
            forward = False
        else:
            forward = True
        for i in range(0, abs(steps)):
            self.get_logger().info('Stepping')
            if forward:
                self.stepper.step_forward()
            else:
                self.stepper.step_backward()
        self.total_steps += steps
        if steps == 0 and apply_pressure:
            self.get_logger().info('Pressure input: ' + str(pressure))
            self.regulator.set_pressure(pressure)
            _, sensed_p = self.regulator.log_pressure()
            self.get_logger().info('Pressurizing to ' + str(sensed_p) + 'kPa')
            self.publish_pressure()  # publish pressure to data logger node after it is set

    def publish_pressure(self):
        """
        Publisher function for pressure and tip estimation data
        """
        msg = DaqOutput()
        v_sense, p_sense = self.regulator.log_pressure()
        msg.pressure = p_sense
        tip_pos_bframe, bend_angle = cc_predict(p_sense)
        msg.tip_position_baseframe = tip_pos_bframe
        #msg.tip_orientation_baseframe = tip_ori_bframe
        msg.bend_angle = bend_angle
        msg.steps = self.total_steps
        msg.contour_distance = [self.cont_dist[0], self.cont_dist[1]]
        self.publisher.publish(msg)

    def send_pressure(self, request, response):
        """
        Service callback used in ros_steering action node
        """
        v_sense, p_sense = self.regulator.log_pressure()
        response.pressure = p_sense
        tip_pos_bframe, bend_angle = cc_predict(p_sense)
        response.tip_position_baseframe = tip_pos_bframe
        #response.tip_orientation_baseframe = tip_ori_bframe
        response.bend_angle = bend_angle
        return response


def main(args=None):
    try:
        rclpy.init(args=args)
        daqnode = DaqInterface()
        rclpy.spin(daqnode)
    except KeyboardInterrupt:
        daqnode.regulator.pressure_pumps_close()
        daqnode.stepper.stepper_close()
    daqnode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
