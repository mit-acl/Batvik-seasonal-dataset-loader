# A script used for visualizing camera poses and point locations.
# Jouko Kinnari, jouko.kinnari@saabgroup.com, 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRot

class Visualizer:
    def __init__(self, ax=None):
        self.ax = ax

        if (self.ax is None):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.clearScene()

        self.xmin = None
        self.ymin = None
        self.zmin = None
        self.xmax = None
        self.ymax = None
        self.zmax = None

        return
    
    def setPlotRange(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax


    def set3Dplotaxes(self):
        # In matplotlib, ax.set_aspect('equal') has not been implemented for whatever reason.
        # Create cubic bounding box to simulate equal aspect ratio.
        
        if (self.xmin is None):

            xs = []
            ys = []
            zs = []

            for (idx, p) in enumerate(self.points):
                xs.append(p[0])
                ys.append(p[1])
                zs.append(p[2])

            for (idx, c) in enumerate(self.camerats):
                t = self.camerats[idx]
                xs.append(t[0])
                ys.append(t[1])
                zs.append(t[2])

            xmax = np.max(xs)
            xmin = np.min(xs)

            ymax = np.max(ys)
            ymin = np.min(ys)

            zmax = np.max(zs)
            zmin = np.min(zs)
        else:
            xmin = self.xmin
            xmax = self.xmax
            ymin = self.ymin
            ymax = self.ymax
            zmin = self.zmin
            zmax = self.zmax

        max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
        
        for xb, yb, zb in zip(Xb, Yb, Zb):
           self.ax.plot([xb], [yb], [zb], 'w')

    def addCamera(self, R, t, name, verticalFovDeg = 60, aspectRatio = 1280/1024, color='k', imagePlaneDepth=1.0, arrowLength=1.0):
        self.cameraRs.append(R)
        self.camerats.append(np.ravel(t))
        self.cameraNames.append(name)
        self.cameraFovVs.append(verticalFovDeg)
        self.cameraAspectRatios.append(aspectRatio)
        self.cameraColors.append(color)
        self.imagePlaneDepths.append(imagePlaneDepth)
        self.arrowLengths.append(arrowLength)
        return
    
    def addPoint(self,p, name, color=None):
        self.points.append(p)
        self.pointNames.append(name)
        self.pointColors.append(color)
        return

    def addPointPair(self,point1,p1name,point2,p2name,p2covariance=None):
        self.pointPairsPoint1.append(point1)
        self.pointPairsPoint1names.append(p1name)
        self.pointPairsPoint2.append(point2)
        self.pointPairsPoint2names.append(p2name)
        self.pointPairsPoint2covariances.append(p2covariance)
        return    
    
    def clearScene(self):
        self.cameraRs = []
        self.camerats = []
        self.cameraFovVs = []
        self.cameraAspectRatios = []
        self.cameraNames = []
        self.cameraColors = []
        self.imagePlaneDepths = []
        self.arrowLengths = []
        self.pointPairsPoint1 = []
        self.pointPairsPoint1names = []
        self.pointPairsPoint2 = []
        self.pointPairsPoint2names = []
        self.pointPairsPoint2covariances = []

        self.points = []
        self.pointNames = []
        self.pointColors = []

        return

    def getFigAndAx(self):
        return (self.fig, self.ax)

    def visualize(self):

        # Plot points as dots
        for (idx, p) in enumerate(self.points):
            color = self.pointColors[idx]
            if (color is None):
                color = 'c'

            self.ax.scatter(p[0],p[1],p[2],color=color)
            self.ax.text(p[0],p[1],p[2],self.pointNames[idx])
        
        # Plot point pairs as connected dots
        # todo: This branch has a lot of copypasted code. Consider refactoring.
        for (idx, _) in enumerate(self.pointPairsPoint1):
            p1 = self.pointPairsPoint1[idx]
            p1name = self.pointPairsPoint1names[idx]
            p2 = self.pointPairsPoint2[idx]
            p2name = self.pointPairsPoint2names[idx]
            p2cov = self.pointPairsPoint2covariances[idx]
        
            xs = [p1[0],p2[0]]
            ys = [p1[1],p2[1]]
            zs = [p1[2],p2[2]]
            
            p = self.ax.plot(xs,ys,zs,marker='o')
            
            color = p[-1].get_color()
            
            self.ax.text(p1[0],p1[1],p1[2],p1name)
            self.ax.text(p2[0],p2[1],p2[2],p2name)
            
            if (p2cov is not None):
                x_std = np.sqrt(p2cov[0,0])
                y_std = np.sqrt(p2cov[1,1])
                z_std = np.sqrt(p2cov[2,2])
            
                xs_covariance = [p2[0] - x_std,p2[0] + x_std]
                ys_covariance = [p2[1],p2[1]]
                zs_covariance = [p2[2],p2[2]]
                                
                self.ax.plot(xs_covariance,ys_covariance,zs_covariance,color=color)

                xs_covariance = [p2[0],p2[0]]
                ys_covariance = [p2[1]-y_std,p2[1]+y_std]
                zs_covariance = [p2[2],p2[2]]
                                
                self.ax.plot(xs_covariance,ys_covariance,zs_covariance,color=color)
                
                xs_covariance = [p2[0],p2[0]]
                ys_covariance = [p2[1],p2[1]]
                zs_covariance = [p2[2]-z_std,p2[2]+z_std]
                                
                self.ax.plot(xs_covariance,ys_covariance,zs_covariance,color=color)                
                
        
        # Plot cameras by showing axes and a pyramid
        for (idx, c) in enumerate(self.cameraNames):

            R = self.cameraRs[idx]
            t = self.camerats[idx]
            
            arrowLength = self.arrowLengths[idx]

            color = self.cameraColors[idx]
    
            verticalFovDeg = self.cameraFovVs[idx]
            aspectRatio = self.cameraAspectRatios[idx]
            horizFovDeg = aspectRatio * verticalFovDeg

            # Camera pyramid depth in visualization
            z_p = self.imagePlaneDepths[idx]

            x_l = -z_p * np.tan(horizFovDeg/2/180*np.pi)
            x_r = -x_l

            y_t = z_p * np.tan(verticalFovDeg/2/180*np.pi)
            y_b = -y_t

            cameraPyramidPoints_z = np.array([0, z_p, z_p, 0, z_p, z_p, 0, z_p, z_p, z_p, z_p])
            cameraPyramidPoints_x = np.array([0, x_l, x_l, 0, x_r, x_r, 0, x_l, x_r, x_r, x_l])
            cameraPyramidPoints_y = np.array([0, y_b, y_t, 0, y_t, y_b, 0, y_t, y_t, y_b, y_b])

            cameraPyramidPoints = np.vstack((cameraPyramidPoints_x,cameraPyramidPoints_y,cameraPyramidPoints_z))
            
            rotatedCameraPyramidPoints = R @ cameraPyramidPoints

            cameraPyramidPoints_x = np.squeeze(np.asarray(rotatedCameraPyramidPoints[0,:]))
            cameraPyramidPoints_y = np.squeeze(np.asarray(rotatedCameraPyramidPoints[1,:]))
            cameraPyramidPoints_z = np.squeeze(np.asarray(rotatedCameraPyramidPoints[2,:]))

            camera_x = t[0]
            camera_y = t[1]
            camera_z = t[2]

            # Rotation matrix unit vectors
            camera_x_vect = R[:,0]*arrowLength
            camera_y_vect = R[:,1]*arrowLength
            camera_z_vect = R[:,2]*arrowLength

            self.ax.quiver(camera_x,camera_y,camera_z,camera_x_vect[0],camera_x_vect[1],camera_x_vect[2],color='r')
            self.ax.quiver(camera_x,camera_y,camera_z,camera_y_vect[0],camera_y_vect[1],camera_y_vect[2],color='g')
            self.ax.quiver(camera_x,camera_y,camera_z,camera_z_vect[0],camera_z_vect[1],camera_z_vect[2],color='b')

            self.ax.plot(camera_x+cameraPyramidPoints_x,camera_y+cameraPyramidPoints_y,camera_z+cameraPyramidPoints_z,color=color)

            self.ax.text(camera_x,camera_y,camera_z,c)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.set3Dplotaxes()


        if (self.xmin is not None):
            self.ax.axes.set_xlim3d(left=self.xmin, right=self.xmax) 
            self.ax.axes.set_ylim3d(bottom=self.ymin, top=self.ymax) 
            self.ax.axes.set_zlim3d(bottom=self.zmin, top=self.zmax) 

        return
        
    def show(self):
        plt.show()

def runTests():
    print("Running tests")

    vis = Visualizer()

    numPoints = 10
    numCameras = 10

    for p in np.arange(numPoints):
        point_x = np.random.randn()*10
        point_y = np.random.randn()*10
        point_z = np.random.randn()*10

        vis.addPoint([point_x,point_y,point_z],"Point {}".format(p))

    for c in np.arange(numCameras):
        cam_x = np.random.randn()*10
        cam_y = np.random.randn()*10
        cam_z = np.random.randn()*10        

        angle_around_x = np.random.uniform(low=0,high=2*np.pi)
        angle_around_y = np.random.uniform(low=0,high=2*np.pi)
        angle_around_z = np.random.uniform(low=0,high=2*np.pi)

        R = ScipyRot.from_euler('zyx', [angle_around_z, angle_around_y, angle_around_x], degrees=False).as_matrix()

        vis.addCamera(R,[cam_x,cam_y,cam_z], "Camera {}".format(c),color='r',imagePlaneDepth=5)

    vis.visualize()
    vis.show()

def show():
    plt.show()

if __name__== "__main__":
    # If this script is run directly from command line, visualize a scene to allow visual checking that everything works.
    runTests()
