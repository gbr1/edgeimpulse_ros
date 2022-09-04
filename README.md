# edgeimpulse_ros

ROS2 wrapper for Edge Impulse on Linux.


## 1. Topics

- `/detection/input/image`, image topic to analyze
- `/detection/output/image`, image with bounding boxes
- `/detection/output/info`, VisionInfo message
- `/detection/output/results`, results as text

## 2. Parameters

- `frame_id` (**string**), _"base_link"_, frame id of output topics
- `model.filepath` (**string**), _""_, absolute filepath to .eim file
- `show.overlay` (**bool**), _true_, show bounding boxes on output image
- `show.labels` (**bool**), _true_, show labels on bounding boxes,
- `show.classification_info` (**bool**), _true_, show the attendibility (0-1) of the prediction


## 3. How to install

1. install edge_impulse_linux: <br>
    `pip3 install edge_impulse_linux`

2. on some boards and aarch64 these are required (e.g. vm on mac m1): <br>
    `sudo apt-get install libatlas-base-dev libportaudio2 libportaudiocpp0 portaudio19-dev` <br>
    `pip3 install pyaudio` <br>

3. download your .eim file as **"linux board"** and choose your architecture

4. make your eim file as executable: <br>
    `cd /path/to/your/eim/file` <br>
    `chmod +x <file>.eim` <br>

5. clone this repo in your workspace: <br>
    `cd ~/dev_ws/src`
    `git clone https://github.com/gbr1/edgeimpulse_ros`

6. check dependencies: <br>
    `cd ~/dev_ws` <br>
    `rosdep install --from-paths src --ignore-src -r -y` <br>

7. build: <br>
    `colcon build --symlink-install` <br>
    `source install/setup.bash` <br>


## 4.  How to run

Launch the node: <br>
    `ros2 run edgeimpulse_ros image_classification --ros-args -p model.filepath:="</absolute/path/to/your/eim/file.eim>" -r /detection/input/image:="/your_image_topic"`
` <br>

## 5. Models

Here you find some prebuilt models: [https://github.com/gbr1/edgeimpulse_example_models](https://github.com/gbr1/edgeimpulse_example_models)

## 6. Known issues

- this wrapper works on foxy, galactic and humble are coming soon (incompatibility on vision msgs by ros-perception)
- if you use a classification model, topic results is empty
- you cannot change color of bounding boxes (coming soon)
- other types (imu and sound based ml) are unavailable



***Copyright © 2022 Giovanni di Dio Bruno - gbr1.github.io***



    

