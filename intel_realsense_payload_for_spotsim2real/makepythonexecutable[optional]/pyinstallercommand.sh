#!/bin/bash
pyinstaller --onefile --hidden-import=bosdyn.client.resources --add-data=/home/spot/miniconda3/envs/spot_ros/lib/python3.9/site-packages/bosdyn/client/resources/robot.pem:bosdyn/client/resources intelrealsense_image_service.py
pyinstaller --onefile --hidden-import=bosdyn.client.resources --add-data=/home/spot/miniconda3/envs/spot_ros/lib/python3.9/site-packages/bosdyn/client/resources/robot.pem:bosdyn/client/resources register_image_service_payload.py
pyinstaller --onefile --hidden-import=bosdyn.client.resources --add-data=/home/spot/miniconda3/envs/spot_ros/lib/python3.9/site-packages/bosdyn/client/resources/robot.pem:bosdyn/client/resources spot-client.py