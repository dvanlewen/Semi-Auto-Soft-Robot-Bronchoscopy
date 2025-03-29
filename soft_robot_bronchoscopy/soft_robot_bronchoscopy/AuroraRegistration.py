
from sksurgerynditracker.nditracker import NDITracker
import time
import numpy as np

class EMTrack:
    #Aurora Setup (Values needed to init Device)
    def __init__(self):
        SETTINGS = {
            "tracker type": "aurora", "ports to use": [3, 4]
            # "tracker type": "aurora", "ports to use": [1]
                }
        self.TRACKER = NDITracker(SETTINGS)

        self.TRACKER.start_tracking()
        port_handles, timestamps, framenumbers, tracking, quality = self.TRACKER.get_frame()

        # initialize tracker with delay to avoid nan for the start future frame collection
        time.sleep(0.300333)
        M = [self.TRACKER.get_frame()]
        time.sleep(0.300333)

    def getpose(self):
        # This gets a frame of Data for every loop
        M = [self.TRACKER.get_frame()]
        time.sleep(0.015)
        # Opening up the data
        F = M[0]
        # # Opening up the data in the specific quad
        D = F[3]
        # # Opening up which device to use in this case 0 = device 1 (ref disk)
        K = D[0]
        # Same with device 2 (probe)
        K2 = D[1]
        # IN the matrix the 4th element is the cartesian coordinate X
        X1 = K[0][3]
        X2 = K2[0][3]
        # IN the matrix the 4th element is the cartesian coordinate Y
        Y1 = K[1][3]
        Y2 = K2[1][3]
        # IN the matrix the 4th element is the cartesian coordinate Z
        Z1 = K[2][3]
        Z2 = K2[2][3]

        ROT1 = K[0:3]
        ROT2 = K2[0:3]
        # This Turns the data in the format needed to conduct the rotation matrix
        ROTT1A = np.array([ROT1[0][0:3], ROT1[1][0:3], ROT1[2][0:3]])
        ROTT2A = np.array([ROT2[0][0:3], ROT2[1][0:3], ROT2[2][0:3]])
        # This puts all the X,Y,Z values into a list
        XYZ1 = np.array([[X1, Y1, Z1]])

        TXYZ1 = XYZ1.transpose()

        XYZ2 = np.array([[X2, Y2, Z2]])

        TXYZ2 = XYZ2.transpose()

        # Subtract the two x, y, z vaues getting relative distance b/w probe and ref disk in global (field generator) frame
        XYZ_Sub = np.subtract(TXYZ1,TXYZ2)
        # invert the rotation matrix
        InvROT1 = np.linalg.inv(ROTT1A)
        # apply it to the relative x, y, z to rotate into ref disk frame
        XYZ_in_Ref_Disc = InvROT1 @ XYZ_Sub
        self.XYZ_ROT_in_Ref_Disc = np.dot(InvROT1, ROTT2A)  # rotation from ref disk frame to probe frame through global frame
        # Test setup specific parameters
        # this is the distance between the origin of the reference disc and the origin of the lung
        DeltaXYZLung = np.array([[-39.46, 40.96, -6.97]])
        #DeltaXYZLung = np.array([[98.44, 14.46, -18.23]])

        DeltaXYZLungT = DeltaXYZLung.transpose()
        XYZ_Delta_Lung = np.subtract(DeltaXYZLungT, XYZ_in_Ref_Disc)  # translation of probe pos into patient frame
        XLungtemp1 = XYZ_Delta_Lung[0]
        YLungtemp1 = XYZ_Delta_Lung[1]
        ZLungtemp1 = XYZ_Delta_Lung[2]

        XYZ_Lung_Temp = np.array([[XLungtemp1, YLungtemp1, ZLungtemp1]])
        XYZ_Lung_TempT = XYZ_Lung_Temp.transpose()

        # Rotating system 20 deg around x axis
        #ROTXM = np.array([[1, 0, 0], [0, -0.34202014332, 0.93969262078], [0, -0.93969262078, -0.34202014332]])
        # rotating the system into the frame of the patient based on test setup
        ROTX1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        ROTY1 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        ROTz1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rotx2 = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])

        ROTXM = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(-20.70527726)), -np.sin(np.deg2rad(-20.70527726))], [0, np.sin(np.deg2rad(-20.70527726)), np.cos(np.deg2rad(-20.70527726))]])

        self.ROTXM2 = ROTXM @ ROTY1 @ ROTX1 @ ROTz1

        # ROTX1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        # ROTY1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        # ROTXM = np.array([[1, 0, 0], [0, 0.93969262078, -0.34202014332], [0, 0.34202014332, 0.93969262078]])

        # self.ROTXM2 = ROTXM @ ROTY1 @ ROTX1

        XYZ_LUNG_PROBE = np.dot(self.ROTXM2, XYZ_Lung_TempT)  # rotation of probe pos into patient frame

        XProbe = XYZ_LUNG_PROBE[0]
        YProbe = XYZ_LUNG_PROBE[1]
        ZProbe = XYZ_LUNG_PROBE[2]

        probe_error = F[-1]

        #    in referrence disk frame  ROTATION 1 =   Rot2 @ InvROT1
        #                     From refer to lungg =   ROTXM2  @ ROTATION 1 
        return XYZ_LUNG_PROBE, probe_error  #, rotAngles
