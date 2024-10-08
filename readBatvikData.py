##################################################
## readBatvikData.py
## A tool for reading the data in the Båtvik seasonal dataset
##################################################
## Author: Jouko Kinnari 2024
## License: MIT License
##################################################

import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as ScipyRot
from scipy.interpolate import interp1d
from pyproj import Transformer

def getBatvikData(pathToData, removeAltOffsetWrtMin=False):

    # Various pieces of data are stored in different csv files.
    pathToGimbalData = os.path.join(pathToData,"gimbal/realtime_data_custom.csv")
    pathToMavlinkData = os.path.join(pathToData,"mavlink/AHRS3.csv")
    pathToRangefinderData = os.path.join(pathToData,"mavlink/RANGEFINDER.csv")
    pathToCameraFrameData = os.path.join(pathToData,"camera/image_timestamps.csv")
    pathToServoOutputs = os.path.join(pathToData,"mavlink/SERVO_OUTPUT_RAW.csv")
    pathToPressure = os.path.join(pathToData,"mavlink/SCALED_PRESSURE.csv")
    pathToPressure2 = os.path.join(pathToData,"mavlink/SCALED_PRESSURE2.csv")

    # Read data from csv files using pandas.
    servo_outputs = pd.read_csv(pathToServoOutputs, sep=';')
    gimbal_df=pd.read_csv(pathToGimbalData, sep=';')
    mavlink_df = pd.read_csv(pathToMavlinkData, sep=';')
    rangefinder_df = pd.read_csv(pathToRangefinderData, sep=';')
    pressures_df  = pd.read_csv(pathToPressure, sep=';')
    pressure = pressures_df['press_abs'].to_numpy()
    pressure_ts = pressures_df['timestamp [s]'].to_numpy()
    pressures2_df = pd.read_csv(pathToPressure2, sep=';')
    pressure2 = pressures2_df['press_abs'].to_numpy()
    pressure2_ts = pressures2_df['timestamp [s]'].to_numpy()
    image_timestamps_df = pd.read_csv(pathToCameraFrameData, sep=';')

    # Convert to numpy arrays.
    image_ts = image_timestamps_df["timestamp [s]"].to_numpy()
    mavlink_ts = mavlink_df['timestamp [s]'].to_numpy()
    mavlink_roll = mavlink_df['roll'].to_numpy()
    mavlink_pitch = mavlink_df['pitch'].to_numpy()
    mavlink_yaw = mavlink_df['yaw'].to_numpy()
    mavlink_alt = mavlink_df['altitude'].to_numpy()
    mavlink_lat = mavlink_df['lat'].to_numpy()
    mavlink_lon = mavlink_df['lng'].to_numpy()

    # Read filenames.
    image_filenames = image_timestamps_df["filename"]

    # Remove altitude offset s.t. lowest altitude is recorded as zero.
    if (removeAltOffsetWrtMin):
        mavlink_alt = mavlink_alt-np.min(mavlink_alt)

    rangefinder_ts = rangefinder_df['timestamp [s]'].to_numpy()
    rangefinder_distance = rangefinder_df['distance'].to_numpy()

    # Read gimbal data.
    gimbal_ts = gimbal_df['ts [s]'].to_numpy()
    gimbal_microcontroller_ts = gimbal_df['bgcTimestamp [s]'].to_numpy()
    gimbal_imuAngleRoll = gimbal_df['imuAngleRoll [deg]'].to_numpy()/180.0*np.pi
    gimbal_imuAnglePitch = gimbal_df['imuAnglePitch [deg]'].to_numpy()/180.0*np.pi
    gimbal_imuAngleYaw = gimbal_df['imuAngleYaw [deg]'].to_numpy()/180.0*np.pi*(-1)
    gimbal_statorRotorAngleRoll = (gimbal_df['statorRotorAngleRoll [deg]'].to_numpy())/180.0*np.pi
    gimbal_statorRotorAnglePitch = (gimbal_df['statorRotorAnglePitch [deg]'].to_numpy())/180.0*np.pi
    gimbal_statorRotorAngleYaw = (gimbal_df['statorRotorAngleYaw [deg]'].to_numpy())/180.0*np.pi*(-1)
    gimbal_gyroX = gimbal_df['gyroX [deg/s]'].to_numpy()
    gimbal_gyroY = gimbal_df['gyroY [deg/s]'].to_numpy()
    gimbal_gyroZ = gimbal_df['gyroZ [deg/s]'].to_numpy()
    gimbal_accX = gimbal_df['accX [m/s^2]'].to_numpy()
    gimbal_accY = gimbal_df['accY [m/s^2]'].to_numpy()
    gimbal_accZ = gimbal_df['accZ [m/s^2]'].to_numpy()
    gimbal_zVect_x = gimbal_df['zVect_x [1]'].to_numpy()
    gimbal_zVect_y = gimbal_df['zVect_y [1]'].to_numpy()
    gimbal_zVect_z = gimbal_df['zVect_z [1]'].to_numpy()
    gimbal_hVect_x = gimbal_df['hVect_x [1]'].to_numpy()
    gimbal_hVect_y = gimbal_df['hVect_y [1]'].to_numpy()
    gimbal_hVect_z = gimbal_df['hVect_z [1]'].to_numpy()

    # Read motor control outputs.
    servo_ts = servo_outputs['timestamp [s]'].to_numpy()
    servo_1 = servo_outputs['servo1_raw'].to_numpy()
    servo_2 = servo_outputs['servo2_raw'].to_numpy()
    servo_3 = servo_outputs['servo3_raw'].to_numpy()
    servo_4 = servo_outputs['servo4_raw'].to_numpy()
    servo_5 = servo_outputs['servo5_raw'].to_numpy()
    servo_6 = servo_outputs['servo6_raw'].to_numpy()

    servos = np.vstack((servo_1,servo_2,servo_3,servo_4,servo_5,servo_6)).T

    # Store all values into a dictionary.
    retval = dict()

    retval["pressure_ts"] = pressure_ts
    retval["pressure"]= pressure
    retval["pressure2_ts"]= pressure2_ts
    retval["pressure2"]= pressure2
    retval["image_ts"]= image_ts
    retval["servo_outputs"]=servo_outputs
    retval["mavlink_ts"]=mavlink_ts
    retval["mavlink_roll"]=mavlink_roll
    retval["mavlink_pitch"]=mavlink_pitch
    retval["mavlink_yaw"]=mavlink_yaw
    retval["mavlink_alt"]=mavlink_alt
    retval["mavlink_lat"]=mavlink_lat
    retval["mavlink_lon"]=mavlink_lon
    retval["rangefinder_ts"]=rangefinder_ts
    retval["rangefinder_distance"]=rangefinder_distance
    retval["gimbal_ts"]=gimbal_ts
    retval["gimbal_microcontroller_ts"]=gimbal_microcontroller_ts
    retval["gimbal_imuAngleRoll"]=gimbal_imuAngleRoll
    retval["gimbal_imuAnglePitch"]=gimbal_imuAnglePitch
    retval["gimbal_imuAngleYaw"]=gimbal_imuAngleYaw
    retval["gimbal_statorRotorAngleRoll"]=gimbal_statorRotorAngleRoll
    retval["gimbal_statorRotorAnglePitch"]=gimbal_statorRotorAnglePitch
    retval["gimbal_statorRotorAngleYaw"]=gimbal_statorRotorAngleYaw
    retval["gimbal_gyroX"]=gimbal_gyroX
    retval["gimbal_gyroY"]=gimbal_gyroY
    retval["gimbal_gyroZ"]=gimbal_gyroZ
    retval["gimbal_accX"]=gimbal_accX
    retval["gimbal_accY"]=gimbal_accY
    retval["gimbal_accZ"]=gimbal_accZ
    retval["gimbal_zVect_x"]=gimbal_zVect_x
    retval["gimbal_zVect_y"]=gimbal_zVect_y
    retval["gimbal_zVect_z"]=gimbal_zVect_z
    retval["gimbal_hVect_x"]=gimbal_hVect_x
    retval["gimbal_hVect_y"]=gimbal_hVect_y
    retval["gimbal_hVect_z"]=gimbal_hVect_z
    retval["servo_ts"]=servo_ts
    retval["servos"]=servos
    retval["image_filenames"] = image_filenames

    # From Euler angles, compute rotation matrices and orientation quaternions for drone body with respect to navigation frame
    Rs_nav_body = eulerVectsToR(mavlink_roll, mavlink_pitch, mavlink_yaw, True)

    # Also compute camera orientation with respect to body frame
    Rs_cam_body = eulerVectsToR(gimbal_statorRotorAngleRoll, gimbal_statorRotorAngleRoll, gimbal_statorRotorAngleYaw, extern=False)

    retval["Rs_nav_body"] = Rs_nav_body
    retval["Rs_cam_body"] = Rs_cam_body

    R_gimbal_transf = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    retval["R_gimbal_transf"] = R_gimbal_transf

    # Convert lon, lat, alt to Cartesian coordinates, use 
    sourceEpsg = 'EPSG:4326' # this is WGS84 - World Geodetic System 1984, used in GPS
    targetEpsg = 'EPSG:3067' # this is ETRS89 / TM35FIN(E,N)

    coordtransformer_dronecontroller_to_map = Transformer.from_crs(sourceEpsg, targetEpsg)
    xs_nav, ys_nav, zs_nav = coordtransformer_dronecontroller_to_map.transform(mavlink_lat/1E7, mavlink_lon/1E7, mavlink_alt)

    retval["xs_nav"] = xs_nav
    retval["ys_nav"] = ys_nav
    retval["zs_nav"] = zs_nav

    # Interpolate the positions of the drone at the times when images are acquired.
    nav_xyz = np.vstack((xs_nav,ys_nav,zs_nav))
    xyz_interpolator = interp1d(mavlink_ts,nav_xyz,bounds_error=False)
    xyz_at_image_times = xyz_interpolator(image_ts)

    x_at_image_times = xyz_at_image_times[0,:]
    y_at_image_times = xyz_at_image_times[1,:]
    z_at_image_times = xyz_at_image_times[2,:]

    retval["x_at_image_times"] = x_at_image_times
    retval["y_at_image_times"] = y_at_image_times
    retval["z_at_image_times"] = z_at_image_times

    #### Interpolate body orientation at the times when images are acquired.

    # Before interpolating Euler angles, get rid of points where angle overflows from -pi to pi or vice versa to avoid interpolation artefacts.
    # We do this by going from angle truncated to interval [-pi, pi] to a continuous angle.
    mavlink_roll_continuous = fromTruncatedAngleToContinuousAngle(mavlink_roll)
    mavlink_pitch_continuous = fromTruncatedAngleToContinuousAngle(mavlink_pitch)
    mavlink_yaw_continuous = fromTruncatedAngleToContinuousAngle(mavlink_yaw)

    # Now, interpolate and return all angles to interval [-pi, pi]
    body_roll_at_image_times = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, mavlink_ts, mavlink_roll_continuous))
    body_pitch_at_image_times = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, mavlink_ts, mavlink_pitch_continuous))
    body_yaw_at_image_times = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, mavlink_ts, mavlink_yaw_continuous))

    retval["body_roll_at_image_times"] = body_roll_at_image_times
    retval["body_pitch_at_image_times"] = body_pitch_at_image_times
    retval["body_yaw_at_image_times"] = body_yaw_at_image_times

    # From Euler angles, compute rotation matrices for drone body with respect to navigation frame
    Rs_nav_body_at_image_times = eulerVectsToR(body_roll_at_image_times, body_pitch_at_image_times, body_yaw_at_image_times, True)
    retval["Rs_nav_body_at_image_times"] = Rs_nav_body_at_image_times

    #### Interpolate camera orientation with respect to body at imaging times
    gimbal_roll_continuous = fromTruncatedAngleToContinuousAngle(gimbal_statorRotorAngleRoll)
    gimbal_pitch_continuous = fromTruncatedAngleToContinuousAngle(gimbal_statorRotorAnglePitch)
    gimbal_yaw_continuous = fromTruncatedAngleToContinuousAngle(gimbal_statorRotorAngleYaw)

    gimbal_roll_at_image_times = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, gimbal_ts, gimbal_roll_continuous))
    gimbal_pitch_at_image_times = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, gimbal_ts, gimbal_pitch_continuous))
    gimbal_yaw_at_image_times = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, gimbal_ts, gimbal_yaw_continuous))

    retval["gimbal_roll_at_image_times"] = gimbal_roll_at_image_times
    retval["gimbal_pitch_at_image_times"] = gimbal_pitch_at_image_times
    retval["gimbal_yaw_at_image_times"] = gimbal_yaw_at_image_times
    
    Rs_cam_body_at_image_times = eulerVectsToR(gimbal_roll_at_image_times, gimbal_pitch_at_image_times, gimbal_yaw_at_image_times, False, True)
    retval["Rs_cam_body_at_image_times"] = Rs_cam_body_at_image_times

    # Also compute what is the orientation of the camera in the navigation frame.
    numImageFrames = len(image_ts)

    Rs_cam_nav_at_image_times = []

    R_90deg_around_z = ScipyRot.from_euler("z",90,True).as_matrix()

    for idx in range(numImageFrames):
        R_cam = Rs_cam_body_at_image_times[idx] @ R_gimbal_transf
        R_nav_body = Rs_nav_body_at_image_times[idx]
        
        R = R_90deg_around_z @ R_nav_body @ R_cam

        Rs_cam_nav_at_image_times.append(R)
    
    retval["Rs_cam_nav_at_image_times"] = Rs_cam_nav_at_image_times


    # Using imuAngleRoll and imuAnglePitch, define a rotation matrix between world and camera.
    
    # Find offset between imuAngleYaw and world coordinate system.
    bodyYawInit = mavlink_yaw[0]
    offsetInit = gimbal_imuAngleYaw[0] - bodyYawInit

    gimbal_roll_continuous_imu = fromTruncatedAngleToContinuousAngle(gimbal_imuAngleRoll)
    gimbal_pitch_continuous_imu = fromTruncatedAngleToContinuousAngle(gimbal_imuAnglePitch)
    gimbal_yaw_continuous_imu = fromTruncatedAngleToContinuousAngle(gimbal_imuAngleYaw)

    gimbal_roll_at_image_times_imu = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, gimbal_ts, gimbal_roll_continuous_imu))
    gimbal_pitch_at_image_times_imu = fromContinuousAngleToTruncatedAngle(np.interp(image_ts, gimbal_ts, gimbal_pitch_continuous_imu))

    gimbal_yaw_at_image_times_rotor = gimbal_yaw_at_image_times - body_yaw_at_image_times

    Rs_cam_nav_imu_unrotated = eulerVectsToR(gimbal_roll_at_image_times_imu, gimbal_pitch_at_image_times_imu, gimbal_yaw_at_image_times_rotor, False, True)

    Rs_cam_nav_at_image_times_imu = []

    for idx in range(numImageFrames):
        R_cam = Rs_cam_nav_imu_unrotated[idx] @ R_gimbal_transf
        
        R = R_90deg_around_z @ R_cam

        Rs_cam_nav_at_image_times_imu.append(R)

    retval["Rs_cam_nav_at_image_times_imu"] = Rs_cam_nav_at_image_times_imu

    return retval

def fromContinuousAngleToTruncatedAngle(angle_vect):
    N = len(angle_vect)

    angle_truncated = np.zeros_like(angle_vect)

    for n in np.arange(N):
        x = angle_vect[n]
        angle_truncated[n] = np.arctan2(np.sin(x),np.cos(x))
    
    return angle_truncated

def fromTruncatedAngleToContinuousAngle(angle_vect, tol=45*np.pi/180):
    # Find zero crossings where the value of angle goes from near zero to near 2*pi or vice versa and remove jumps to get a continuous angle.
    N = len(angle_vect)

    angle_cont = np.zeros_like(angle_vect)
    angle_cont[0] = angle_vect[0]

    for n in np.arange(0,N-1):
        angle_curr = angle_vect[n]
        angle_next = angle_vect[n+1]

        overflow_compensation = 0
        if (angle_curr > np.pi - tol and angle_next < -np.pi + tol):
            overflow_compensation = 2*np.pi
        elif (angle_curr < -np.pi + tol and angle_next > np.pi - tol):
            overflow_compensation = -2*np.pi
    
        angle_cont[n+1] = angle_cont[n] + angle_next - angle_curr + overflow_compensation

    return angle_cont

def eulerVectsToR(rolls,pitches,yaws,transpose=False,extern=True):
    Rs = []
    
    if (extern):
        sequence="ZYX"
    else:
        sequence="zyx"

    for r, p, y in zip(rolls,pitches,yaws):
        SR = ScipyRot.from_euler(sequence,[y,p,r])
        R = SR.as_matrix()

        if (transpose):
            R = R.T
        
        Rs.append(R)
    
    return Rs