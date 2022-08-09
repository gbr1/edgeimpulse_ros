# edgeimpulse_ros
ROS2 wrapper for Edge Impulse 


## How to install

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


## How to run

Launch the node: <br>
    `ros2 run edgeimpulse_ros image_classification --ros-args -p model.filepath:="</absolute/path/to/your/eim/file.eim>" -r /detection/input/image:="/your_image_topic"`
` <br>



***Copyright © 2022 Giovanni di Dio Bruno - gbr1.github.io***



    

