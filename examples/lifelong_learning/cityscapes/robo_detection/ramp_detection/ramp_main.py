# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from cv_bridge import CvBridge
from robosdk.core import Robot
from robosdk.utils.context import Context
from robosdk.msgs.sender.ros import RosMessagePublish

from ramp_detection.ramp_interface import Estimator

class RampDetection:
    def __init__(self):
        self.robot = Robot(name="x20", config="ysc_x20")
        self.segment = Estimator()
        self.robot.connect()
        self.publish = RosMessagePublish()
        _topic = Context.get("ramp_detection", "/robovideo")
        self.publish.register("ramp_detection", topic=_topic,
                              converter=CvBridge().cv2_to_imgmsg,
                              convert_param={"encoding": "bgr8"})

    def run(self):
        if not getattr(self.robot, "camera", ""):
            return
        while 1:
            img, dep = self.robot.camera.get_rgb_depth()
            if img is None:
                continue

            location = self.segment.predict(img, depth=dep)
            if not location:
                self.publish.send("ramp_detection", img)
                continue

            if location == "no_ramp":
                self.robot.logger.info(f"Ramp location: {location}. Keep moving.")
                continue
           
            if location == "small_trapezoid": 
                self.robot.logger.info(f"Ramp location: {location}. Keep moving.")
                self.robot.navigation.go_forward()
                continue

            self.robot.logger.info(f"Ramp detected: {location}.")

            if location == "upper_left" or location == "center_left":
                self.robot.logger.info("Move to the left!")
                self.robot.navigation.turn_left()

            elif location == "bottom_left":
                self.robot.logger.info("Backward and move to the left!")
                self.robot.navigation.go_backward()
                self.robot.navigation.turn_left()

            elif location == "upper_right" or location == "center_right":
                self.robot.logger.info("Move to the right!")
                self.robot.navigation.turn_right()

            elif location == "bottom_right":
                self.robot.logger.info("Backward and move to the right!")
                self.robot.navigation.go_backward()
                self.robot.navigation.turn_right()

            self.robot.navigation.go_forward()


if __name__ == '__main__':
    project = RampDetection()
    project.run()
