# Båtvik Seasonal Dataset loader

## Information

This repository contains Python scripts that can be used for parsing the data that was released as part of the ["SOS-Match: Segmentation for Open-Set Robust Correspondence Search and Robot Localization in Unstructured Environments"](https://acl.mit.edu/SOS-Match/) paper.

Website: https://acl.mit.edu/SOS-Match/

Paper: https://arxiv.org/abs/2401.04791

### Description of data

The dataset contains recordings of six flights along the same trajectory in coastal Finland. The flights take place at different times of the year. The images contain significant seasonal appearance variation.

The following measurements, and much more, are available in the dataset:
- Image frames from a camera carried by a drone
- Camera position and orientation information in a georeferenced coordinate system
- Raw IMU data
- Downward-facing LIDAR rangefinder measurements, barometer measurements
- Drone motor speed control reference signals

For a complete description, see the files `readBatvikData.py` and `plotBatvikData.py`.

### Citing
If you decide to use this dataset in your work, please cite it:

```
@article{thomas2024sosmatch,
  author    = {Thomas, Annika and Kinnari, Jouko and Lusk, Parker and Konda, Kota and How, Jonathan},
  title     = {SOS-Match: Segmentation for Open-Set Robust Correspondence Search and Robot Localization in Unstructured Environments},
  journal   = {arXiv preprint arXiv:2401.04791},
  year      = {2024},
}
```

### Further information

For questions related to this repository, feel free to contact Jouko Kinnari.

## Preparations

### Setting up a Python environment

A working Python environment is listed in `requirements.txt`.

### Downloading data
Download the data from the [url shown on the paper website](https://acl.mit.edu/SOS-Match/).

## Usage

The file `readBatvikData.py` contains a script that reads the data and image filenames from a data folder and provides a dict with Numpy arrays containing relevant data.

The file `plotBatvikData.py` contains plotting scripts for visualizing the data. The easiest way to explore the data is to run the plotting script to visualize the parameters recorded in the data:

```
python3 plotBatvikData.py --path=../batvik-seasonal-dataset/Early\ Spring/ --orthotiff=../orthophotos/K4224H.jp2
```
The `--orthotiff` parameter is optional and can be used for specifying a map image over which the drone's trajectory is plotted.


## Remarks on the quality of the data

The data was recorded with a somewhat unoptimal hardware setup (the hardware was designed for a surveillance application and not for photogrammetry). Due to the hardware configuration in the drone used for data collection, a part of the data was collected at a high image resolution, which caused some timing jitter on the timestamps of the images. The resolution was later configured to a lower one, hence some of the data have higher resolution images than others.

The ground truth position infromation is from a non-RTK GPS. The camera orientation information is computed as a combination of two transformations. First, the drone frame pose is estimated by the drone flight controller EKF and second, the orientation of the camera with respect to the drone frame is measured with rotation encoders in the camera gimbal. So do expect some error with the orientation information as well - it won't be pixel perfect if you e.g. compute the reprojection error of the same landmark observed from two different places.

The drone was equipped with a downward-facing LIDAR sensor and two barometers. The LIDAR range is about 20m, so it only shows a meaningful measurement value when the drone is ascending or descending near the start and return locations.

## Camera parameters

The following camera parameters are given for an image scaled to a resolution 960 by 540 pixels. For higher resolution images (e.g. Early Spring), first scale them to 960 by 540 pixels.

```
   distortion_parameters:
      k1: -0.10381091174304308
      k2: 0.06234361571694061
      p1: 0.00280555268843055
      p2: -0.0011575804154380536
      k3: 0
   projection_parameters:
      fx: 782.215325118697
      fy: 777.15751271533485
      cx: 498.33643785710705
      cy: 280.9999085826019
      s: 0
   image_dimensions:
      w: 960
      h: 540
```

## How to download an orhtophoto

In order to use the Båtvik seasonal dataset, you do not need to download an orthophoto, but for convenience, instructions for downloading one are provided below.

A high-quality orthophoto of the area is available via [National Land Survey of Finland](https://www.maanmittauslaitos.fi/en). The license of National Land Survey of Finland does not permit redistributing the map data, hence you have to download it yourself.

See in particular the page [MapSite "Download geospatial data" service](https://asiointi.maanmittauslaitos.fi/karttapaikka/tiedostopalvelu?lang=en)
- Click "Orthophoto"
- Find the tile K4224H (you can find it by searching for Båtvik, Kirkkonummi) and add it to the shopping cart
- Click "Go to checkout" and fill your name and email, then click "Order"
- Follow the instructions in your mailbox to download the map tile.

Note that the National Land Survey of Finland download service also offers elevation data, should you need it for your own research.