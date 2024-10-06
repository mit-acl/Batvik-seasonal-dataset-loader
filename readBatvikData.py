import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as ScipyRot
from scipy.interpolate import interp1d
from pyproj import Transformer
import argparse

def getAllData(pathToData, removeAltOffsetWrtMin=False):

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

    # By default, remove altitude offset s.t. lowest altitude is recorded as zero.
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
    # We do this by going from angle truncated to interval [-pi, pi] to an unbound angle.
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

def visualizePathOnMap(data, pathToMapFile, ax, zoomToPath=True):
    from OrthoImageLoader import OrthoImageLoader
    
    x = data["xs_nav"]
    y = data["ys_nav"]
    z = data["zs_nav"]

    oil = OrthoImageLoader(pathToMapFile)
    oil.plotMap(ax)

    ax.plot(x,y,'r')

    if (zoomToPath):
        ax.set_xlim(np.min(x)-200,np.max(x)+200)
        ax.set_ylim(np.min(y)-200,np.max(y)+200)
    
    # plot start location
    ax.plot(x[0],y[0],'r*')

def visualizePoses(data, everyNth=100):
    from Visualizer import Visualizer

    vis = Visualizer()

    xs = data["xs_nav"]
    ys = data["ys_nav"]
    zs = data["zs_nav"]
    Rs_body = data["Rs_nav_body"]
    Rs_cam = data["Rs_cam_body"]

    for idx, (x,y,z,R_n_b,R_c_b) in enumerate(zip(xs,ys,zs,Rs_body,Rs_cam)):
        if (idx % everyNth == 0):
            
            t = np.array([x,y,z])
            vis.addCamera(R_n_b,t,f"{idx}",imagePlaneDepth=25, arrowLength=50)
    
    vis.visualize()

    return vis.getFigAndAx()

def visualizeGimbalRotation(data, idx):
    from Visualizer import Visualizer

    vis = Visualizer()

    Rs_cam = data["Rs_cam_body"]
    R_gimbal_transf = data["R_gimbal_transf"]


    R = Rs_cam[idx] @ R_gimbal_transf

    t = np.array([0,0,0])
    t2 = np.array([1,0,0])

    vis.addCamera(R, t, f"{idx}")
    vis.visualize()

    return vis.getFigAndAx()


def visualizeCameraPoseAtImagingTimes(data, everyNth=100):
    from Visualizer import Visualizer

    xs = data["x_at_image_times"]
    ys = data["y_at_image_times"]
    zs = data["z_at_image_times"]

    vis = Visualizer()

    for idx in range(0,len(xs),everyNth):

        x = xs[idx]
        y = ys[idx]
        z = zs[idx]

        R = data["Rs_cam_nav_at_image_times"][idx]

        t = np.array([x,y,z])

        vis.addCamera(R,t,f"{idx}",imagePlaneDepth=50, arrowLength=50)
    
    vis.visualize()

    return vis.getFigAndAx()


def plotData(data, titleText, mapFile=None):
    
    fig1, ax1 = plt.subplots()
    ax1.plot(data["mavlink_lon"],data["mavlink_lat"])
    fig1.suptitle(titleText)

    fig2, axs2 = plt.subplots(2,sharex=True)
    axs2[0].plot(data["mavlink_ts"],data["mavlink_alt"], label="altitude (mavlink)")
    axs2[0].set_ylabel("altitude (m)")
    axs2[1].plot(data["pressure_ts"],data["pressure"], label="pressure")
    axs2[1].plot(data["pressure2_ts"],data["pressure2"], label="pressure2")
    axs2[1].set_ylabel("pressure (Pa)")
    axs2[0].plot(data["rangefinder_ts"],data["rangefinder_distance"],label="rangefinder distance")
    axs2[0].legend()
    axs2[1].legend()
    fig2.suptitle(titleText)

    fig3, axs3 = plt.subplots(3,sharex=True)
    axs3[0].plot(data["mavlink_ts"],data["mavlink_roll"], label="roll (mavlink)")
    axs3[1].plot(data["mavlink_ts"],data["mavlink_pitch"], label="pitch (mavlink)")
    axs3[2].plot(data["mavlink_ts"],data["mavlink_yaw"], label="yaw (mavlink)")
    axs3[0].legend()
    axs3[1].legend()
    axs3[2].legend()
    fig3.suptitle(titleText)

    fig4, axs4 = plt.subplots(3,sharex=True)
    axs4[0].plot(data["gimbal_ts"],data["gimbal_statorRotorAngleRoll"], label="roll (gimbal)")
    axs4[1].plot(data["gimbal_ts"],data["gimbal_statorRotorAnglePitch"], label="pitch (gimbal)")
    axs4[2].plot(data["gimbal_ts"],data["gimbal_statorRotorAngleYaw"], label="yaw (gimbal)")
    axs4[0].legend()
    axs4[1].legend()
    axs4[2].legend()
    fig4.suptitle(titleText)

    fig5, axs5 = plt.subplots(6,sharex=True)
    axs5[0].plot(data["gimbal_ts"],data["gimbal_gyroX"], label="gyro x (gimbal)")
    axs5[1].plot(data["gimbal_ts"],data["gimbal_gyroY"], label="gyro y (gimbal)")
    axs5[2].plot(data["gimbal_ts"],data["gimbal_gyroZ"], label="gyro z (gimbal)")
    axs5[0].legend()
    axs5[1].legend()
    axs5[2].legend()
    axs5[3].plot(data["gimbal_ts"],data["gimbal_accX"], label="acceleration x (gimbal)")
    axs5[4].plot(data["gimbal_ts"],data["gimbal_accY"], label="acceleration y (gimbal)")
    axs5[5].plot(data["gimbal_ts"],data["gimbal_accZ"], label="acceleration z (gimbal)")
    axs5[3].legend()
    axs5[4].legend()
    axs5[5].legend()
    fig5.suptitle(titleText)

    fig6, ax6 = plt.subplots()
    ax6.plot(data["servo_ts"],data["servos"])
    ax6.set_ylabel("servo outputs")
    fig6.suptitle(titleText)


    tdiff_range_min = 0.1
    tdiff_range_max = 0.3

    image_timestamps_diff = np.diff(data["image_ts"])
    image_timestamps_diff_padded = np.hstack((image_timestamps_diff,np.nan))

    fig7, axs7 = plt.subplots(1,2,figsize=(15,5))
    fig7.suptitle(titleText)
    axs7[0].hist(image_timestamps_diff,bins=100,range=(tdiff_range_min,tdiff_range_max))
    axs7[1].plot(data["image_ts"],image_timestamps_diff_padded)
    axs7[0].set_xlabel("time difference between camera frames (s)")
    axs7[0].set_ylabel("frequency")
    axs7[1].set_ylim([tdiff_range_min,tdiff_range_max])
    axs7[1].set_xlabel("time since start of recording (s)")
    axs7[1].set_ylabel("time difference between camera frames (s)")

    fig8, ax8 = plt.subplots()
    fig8.suptitle(titleText)

    ax8.plot(data["xs_nav"],data["ys_nav"])

    fig9, ax9 = plt.subplots(figsize=(10,10))
    if (mapFile is not None):
        visualizePathOnMap(data, mapFile, ax9)

    fig10, ax10 = visualizePoses(data)
    fig10.suptitle("Drone frame poses")

    fig11, ax11 = visualizeGimbalRotation(data, 1000)
    fig11.suptitle("Gimbal rotation")

    fig12, ax12 = visualizeCameraPoseAtImagingTimes(data)
    fig12.suptitle("Camera poses at imaging times")

    figs = (fig1,fig2,fig3,fig4,fig5,fig6,fig7, fig8, fig9, fig10, fig11, fig12)
    axs =  (ax1, axs2,axs3,axs4,axs5,ax6,axs7,ax8, ax9, ax10, ax11, ax12)

    return (figs, axs)


if __name__=="__main__":
    #pathToData = "/mnt/ntfsdisk/Datasets/Flights_recorded_on_drone/40"
    #startIdx = 3550
    #endIdx = 3700

    pathToData = "../batvik-seasonal-dataset/40"

    data = getAllData(pathToData)

    import matplotlib.pyplot as plt

    plotData(data, "example")

    plt.show()

