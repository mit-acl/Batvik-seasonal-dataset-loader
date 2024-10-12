##################################################
## plotBatvikData.py
## A tool for visualizing the data in the Båtvik seasonal dataset
##################################################
## Author: Jouko Kinnari 2024
## License: MIT License
##################################################

import matplotlib.pyplot as plt
from readBatvikData import getBatvikData
import argparse
from Visualizer import Visualizer
import numpy as np

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

    vis = Visualizer()

    xs = data["xs_nav"]
    ys = data["ys_nav"]
    zs = data["zs_nav"]
    Rs_body = data["Rs_nav_body"]
    Rs_cam = data["Rs_cam_body"]

    for idx, (x,y,z,R_n_b,R_c_b) in enumerate(zip(xs,ys,zs,Rs_body,Rs_cam)):
        if (idx % everyNth == 0):
            
            t = np.array([x,y,z])
            vis.addCamera(R_n_b,t,f"{idx}",imagePlaneDepth=0, arrowLength=50)
    
    vis.visualize()

    return vis.getFigAndAx()

def visualizeGimbalRotation(data, idx):

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


def plotData(data, mapFile=None):
    
    fig1, ax1 = plt.subplots()
    ax1.plot(data["mavlink_lon"],data["mavlink_lat"])
    fig1.suptitle("Lon, lat")

    fig2, axs2 = plt.subplots(2,sharex=True)
    axs2[0].plot(data["mavlink_ts"],data["mavlink_alt"], label="altitude (mavlink)")
    axs2[0].set_ylabel("altitude (m)")
    axs2[1].plot(data["pressure_ts"],data["pressure"], label="pressure")
    axs2[1].plot(data["pressure2_ts"],data["pressure2"], label="pressure2")
    axs2[1].set_ylabel("pressure (Pa)")
    axs2[0].plot(data["rangefinder_ts"],data["rangefinder_distance"],label="rangefinder distance")
    axs2[0].legend()
    axs2[1].legend()
    fig2.suptitle("Altitude, rangefinder, and pressure sensor readings")

    fig3, axs3 = plt.subplots(3,sharex=True)
    axs3[0].plot(data["mavlink_ts"],data["mavlink_roll"], label="roll (mavlink)")
    axs3[1].plot(data["mavlink_ts"],data["mavlink_pitch"], label="pitch (mavlink)")
    axs3[2].plot(data["mavlink_ts"],data["mavlink_yaw"], label="yaw (mavlink)")
    axs3[0].legend()
    axs3[1].legend()
    axs3[2].legend()
    fig3.suptitle("Drone frame roll, pitch, yaw")

    fig4, axs4 = plt.subplots(3,sharex=True)
    axs4[0].plot(data["gimbal_ts"],data["gimbal_statorRotorAngleRoll"], label="roll (gimbal)")
    axs4[1].plot(data["gimbal_ts"],data["gimbal_statorRotorAnglePitch"], label="pitch (gimbal)")
    axs4[2].plot(data["gimbal_ts"],data["gimbal_statorRotorAngleYaw"], label="yaw (gimbal)")
    axs4[0].legend()
    axs4[1].legend()
    axs4[2].legend()
    fig4.suptitle("Gimbal roll, pitch, yaw")

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
    fig5.suptitle("Measured gimbal accelerations and rotational velocities")

    fig6, ax6 = plt.subplots()
    ax6.plot(data["servo_ts"],data["servos"])
    ax6.set_ylabel("servo output")
    fig6.suptitle("Motor speed control reference")


    tdiff_range_min = 0.1
    tdiff_range_max = 0.3

    image_timestamps_diff = np.diff(data["image_ts"])
    image_timestamps_diff_padded = np.hstack((image_timestamps_diff,np.nan))

    fig7, axs7 = plt.subplots(1,2,figsize=(15,5))
    fig7.suptitle("Image acquisition timing statistics")
    axs7[0].hist(image_timestamps_diff,bins=100,range=(tdiff_range_min,tdiff_range_max))
    axs7[1].plot(data["image_ts"],image_timestamps_diff_padded)
    axs7[0].set_xlabel("time difference between camera frames (s)")
    axs7[0].set_ylabel("frequency")
    axs7[1].set_ylim([tdiff_range_min,tdiff_range_max])
    axs7[1].set_xlabel("time since start of recording (s)")
    axs7[1].set_ylabel("time difference between camera frames (s)")

    fig8, ax8 = plt.subplots()
    fig8.suptitle("Easting and Northing in ETRS89 / TM35FIN")
    ax8.plot(data["xs_nav"],data["ys_nav"])

    fig9, ax9 = plt.subplots(figsize=(10,10))
    fig9.suptitle("Trajectory of the drone overlaid on a map")
    if (mapFile is not None):
        visualizePathOnMap(data, mapFile, ax9)

    fig10, ax10 = visualizePoses(data)
    fig10.suptitle("Drone frame poses")

    fig12, ax12 = visualizeCameraPoseAtImagingTimes(data)
    fig12.suptitle("Camera poses at imaging times")

    figs = (fig1,fig2,fig3,fig4,fig5,fig6,fig7, fig8, fig9, fig10, fig12)
    axs =  (ax1, axs2,axs3,axs4,axs5,ax6,axs7,ax8, ax9, ax10, ax12)

    return (figs, axs)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the folder containing data for one flight", required=True)
    parser.add_argument("--orthotiff", help="Path to an aerial image of the region where the flight took place", default=None)
    args = parser.parse_args()

    data = getBatvikData(args.path)

    plotData(data, args.orthotiff)

    plt.show()