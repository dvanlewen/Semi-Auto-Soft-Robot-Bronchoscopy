import nidaqmx
import numpy as np
import time
import matplotlib.pyplot as plt

pressure_dir = True  # True when increase is needed, false when decrease is needed
UNIT_PRESSURE = 0.025  # in V, corresponds to 1 kPa
pressure_counter = 0
RADIANS_PER_UNIT_PRESSURE = 0.1  # Update after characterizing
CAMCENTER_X = 200
CAMCENTER_Y = 200
CAMCENTER = np.array([CAMCENTER_X, CAMCENTER_Y])
lost_counter = 0
L = 35  # length of tip in mm

debug = False
#if debug:
    # Debugging
    #ax = plt.axes()
    #ax.set_xlim(0, 400)
    #ax.set_ylim(0, 400)

class pressure_pumps:

    def __init__(self):
        self.press_readout = nidaqmx.Task()
        self.press_in = nidaqmx.Task()
        port_out = 'Dev1/ai0'
        port_in = 'Dev1/ao0'
        self.press_readout.ai_channels.add_ai_voltage_chan(port_out, terminal_config=nidaqmx.constants.TerminalConfiguration(10083))
        self.press_in.ao_channels.add_ao_voltage_chan(port_in)
        #self.press_in.start()
        #self.press_readout.start()

        # From ITV0030-2N Datasheet
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
        Inputs a set pressure to regulators
        """
        self.volt_in = pressure_input * (self.vMax - self.vMin)/(self.pMax - 0)
        if self.volt_in < 0:  #self.vMin + UNIT_PRESSURE:
            raise ValueError('VOLTAGE IS TOO LOW')
        if self.volt_in > self.vMax - 20 * UNIT_PRESSURE:
            raise ValueError('VOLTAGE IS TOO HIGH')
        self.press_in.write(self.volt_in)

    def log_pressure(self):
        v_sense = self.press_readout.read()
        #pressure_voltage.append(v_sense)
        p_sense = ((self.pMax - 0) / (self.readvMax - self.readvMin)) * (v_sense - self.readvMin)
        #pressure_data.append(p_sense)
        return v_sense, p_sense

    def pressure_pumps_close(self):
        self.press_in.write(0)
        self.press_readout.close()
        self.press_in.close()

def distance(x_1, y_1, x_2, y_2):
    """
    Calculates the distance between two points
    """

    dist_between_2_points = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    return dist_between_2_points


if not debug:
    pressure_pumps = pressure_pumps()

def track_steering(contour_list):
    global found
    distance_to_cont = 0
    found = False
    contour_ori = np.array([CAMCENTER_X, CAMCENTER_Y])
    contour = contour_list.get()
    #for contour in contour_list:
    if contour.tracking == 1:
        found = True
        contour_ori = np.array([contour.center_x, contour.center_y])
        distance_to_cont = distance(CAMCENTER_X, CAMCENTER_Y, contour.center_x, contour.center_y)
        if debug:
            # Virtual update for Debugging
            contour.center[0] = 200
            #contour.center[1] = contour.center[1] + 205
            contour.center = contour.center - np.array([0, 5])  # simulates bending towards contour
            contour.center_x = contour.center[0]
            contour.center_y = contour.center[1]
            print(contour.center)
            #ax.scatter(contour.center_x, contour.center_y)
        #break
    return distance_to_cont, found, contour_ori

def pressure_control(distance_to_cont, contour_list, contour_ori):
    # ADD: some control (Adaptive?) code here that takes in path+contour pressure calcs
    # to get pressure input signal that should be applied
    # Currently uses a visual servo method
    global pressure_counter, lost_counter
    from __main__ import Video
    if not debug:
        v_sense = pressure_pumps.press_readout.read()
        p_sense = ((pressure_pumps.pMax - 0) / (pressure_pumps.readvMax - pressure_pumps.readvMin)) * (v_sense - pressure_pumps.readvMin)
        print('Pressure:', p_sense)
    else:
        p_sense = 0
    while contour_ori[1] > 210:
        found = False
        print(contour_ori[1])
        if (abs(contour_ori[1]) > CAMCENTER_Y + 10) & (p_sense < 300):  #pressure_dir: this was never updated
            if not debug:
                pressure_pumps.increase_pressure()
                time.sleep(0.2)
                v_sense = pressure_pumps.press_readout.read()
                p_sense = ((pressure_pumps.pMax - 0) / (pressure_pumps.readvMax - pressure_pumps.readvMin)) * (v_sense - pressure_pumps.readvMin)
                print('Pressure:', p_sense)
            else:
                pressure_counter += 1
                p_sense = pressure_counter
            print('increasing')
            print(distance_to_cont)
            #distance_to_cont -= 1  # remove when camera can move
            if not Video.video:
                break
        else:
            break
        # Recheck after pressurization
        distance_to_cont, found, contour_ori = track_steering(contour_list)
        # Decrease pressure if it went too far
        if (abs(contour_ori[1]) < CAMCENTER_Y - 10) & (p_sense > 20):
            if not debug:
                pressure_pumps.decrease_pressure()
                #time.sleep(0.2)
                v_sense = pressure_pumps.press_readout.read()
                p_sense = ((pressure_pumps.pMax - 0) / (pressure_pumps.readvMax - pressure_pumps.readvMin)) * (v_sense - pressure_pumps.readvMin)
            else:
                pressure_counter -= 1
                p_sense = pressure_counter
            print('decreasing')
        distance_to_cont, found, contour_ori = track_steering(contour_list)
        if not found:
            lost_counter += 1
            break
        if not Video.video:
            pressure_pumps.press_in.write(0)
            break
    total_pressure = p_sense
    #if debug:
        #plt.show()
    return total_pressure

def estimate_tip(total_pressure):
    global L
    from __main__ import Video
    phi_tip = np.array([0, 1, 0])
    z_tip = np.array([0, 0, 1])
    #v_sense = pressure_pumps.press_readout.read()
    #p_sense = ((pressure_pumps.pMax - 0) / (pressure_pumps.readvMax - pressure_pumps.readvMin)) * (v_sense - pressure_pumps.readvMin)
    #total_pressure = p_sense  #UNIT_PRESSURE * pressure_counter
    #total_pressure = 0
    if total_pressure > 100:
        total_bending_angle = 0.0004*total_pressure**2 + 0.033 * total_pressure - 7.2437
    else:
        total_bending_angle = 0
    # Rotate tip frame in base frame by calculated bending angle
    R_tip2base = np.array([[1, 0, 0], [0, np.cos(total_bending_angle), np.sin(total_bending_angle)], [0, -np.sin(total_bending_angle), np.cos(total_bending_angle)]])
    phi_hat_baseframe = R_tip2base @ phi_tip
    #phi_hat_pframe = R_base2patient @ phi_hat_baseframe  # np.linalg.inv(R_patient2tip)
    tip_dir_baseframe = R_tip2base @ z_tip
    #tip_dir_pframe = R_base2patient @ tip_dir_baseframe

    #phi_hat = phi_hat_pframe
    #print('Tip Direction:', tip_dir_pframe)
    if total_bending_angle == 0:
        r_o_c = 0
        tip_z = L
    else:
        r_o_c = L / total_bending_angle
        tip_z = r_o_c * np.sin(total_bending_angle)
    tip_y = r_o_c * (1 - np.cos(total_bending_angle))
    if total_bending_angle == 0:
        tip_z = L
    tip_pos_bframe = np.array([0, tip_y, tip_z])
    #tip_pos_pframe = R_base2patient @ tip_pos_bframe + base_pos_pframe

    if not Video.video:
        pressure_pumps.press_in.write(0)

    return tip_pos_bframe, tip_dir_baseframe, total_bending_angle

def plot_tip(R_base2patient, total_bending_angle, base_pos_pframe):
    global L
    # Constant Curvature Model to determine tip position and plot
    body_angle = np.linspace(0, total_bending_angle, 5)
    if total_bending_angle == 0:
        r_o_c = 0
        body_z = L * np.ones(len(body_angle))
    else:
        r_o_c = L / total_bending_angle
        body_z = r_o_c * np.sin(body_angle)
    body_y = r_o_c * (1 - np.cos(body_angle))
    body_x = np.zeros(5, dtype=float)
    body_pos_baseframe = np.concatenate(([body_x], [body_y], [body_z]), axis=0)
    rotated = R_base2patient @ body_pos_baseframe
    body_pos_x = rotated[0, :] + base_pos_pframe[0]
    body_pos_y = rotated[1, :] + base_pos_pframe[1]
    body_pos_z = rotated[2, :] + base_pos_pframe[2]

    return body_pos_x, body_pos_y, body_pos_z
