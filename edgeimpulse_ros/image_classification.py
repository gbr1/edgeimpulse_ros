# Copyright 2022 Giovanni di Dio Bruno - gbr1.github.io

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from distutils.log import info
from unittest import result
from .submodules import device_patches
import cv2

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import VisionInfo

import os
import time
import sys, getopt
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner




class EI_Image_node(Node):

    def __init__(self):

        self.occupied = False
        self.img = None
        self.info_msg = VisionInfo()
        self.cv_bridge = CvBridge()

        super().__init__('ei_image_classifier_node')
        self.init_parameters()
        self.ei_classifier = self.EI_Classifier(self.modelfile, self.get_logger())

        self.info_msg.header.frame_id = self.frame_id
        self.info_msg.method = self.ei_classifier.model_info['model_parameters']['model_type']
        self.info_msg.database_location = self.ei_classifier.model_info['project']['name']+' / '+self.ei_classifier.model_info['project']['owner']
        self.info_msg.database_version = self.ei_classifier.model_info['project']['deploy_version']
        self.timer_parameter = self.create_timer(2,self.parameters_callback)

        self.image_publisher = self.create_publisher(Image,'/detection/output/image',1)
        self.results_publisher = self.create_publisher(Detection2DArray,'/detection/output/results',1)
        self.info_publisher = self.create_publisher(VisionInfo, '/detection/output/info',1)

        self.timer_classify = self.create_timer(0.01,self.classify_callback)
        self.timer_classify.cancel()
        self.subscription = self.create_subscription(Image,'/detection/input/image',self.listener_callback,1)
        self.subscription 




    def init_parameters(self):
        self.declare_parameter('model.filepath','')
        self.modelfile= self.get_parameter('model.filepath').get_parameter_value().string_value

        self.declare_parameter('frame_id','base_link')
        self.frame_id= self.get_parameter('frame_id').get_parameter_value().string_value

        self.declare_parameter('show.overlay', True)
        self.show_overlay = self.get_parameter('show.overlay').get_parameter_value().bool_value

        self.declare_parameter('show.center', False)
        self.show_center = self.get_parameter('show.center').get_parameter_value().bool_value


        self.declare_parameter('show.labels',True)
        self.show_labels_on_image = self.get_parameter('show.labels').get_parameter_value().bool_value

        self.declare_parameter('show.classification_info', False)
        self.show_extra_classification_info = self.get_parameter('show.classification_info').get_parameter_value().bool_value




    def parameters_callback(self):
        self.show_labels_on_image = self.get_parameter('show.labels').get_parameter_value().bool_value
        self.show_extra_classification_info = self.get_parameter('show.classification_info').get_parameter_value().bool_value
        self.show_overlay = self.get_parameter('show.overlay').get_parameter_value().bool_value
        self.show_center= self.get_parameter('show.center').get_parameter_value().bool_value




    def listener_callback(self, msg):
        if len(msg.data):
            current_frame = self.cv_bridge.imgmsg_to_cv2(msg)
            img_encoding = msg.encoding
            if img_encoding=="bgr8":
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            if not self.occupied:
                self.img = current_frame
                self.timer_classify.reset()
            




    def classify_callback(self):
        self.occupied = True

        # vision msgs
        results_msg = Detection2DArray()
        time_now = self.get_clock().now().to_msg()
        results_msg.header.stamp = time_now
        results_msg.header.frame_id = self.frame_id

        
        # classify
        features, cropped, res = self.ei_classifier.classify(self.img)


        #p repare output
        if "classification" in res["result"].keys():
            if self.show_extra_classification_info:
                self.get_logger().info('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
            for label in self.labels:
                score = res['result']['classification'][label]
                if self.show_extra_classification_info:
                    self.get_logger().info('%s: %.2f\t' % (label, score), end='')
                    

        elif "bounding_boxes" in res["result"].keys():
            if self.show_extra_classification_info:
                self.get_logger().info('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    
            for bb in res["result"]["bounding_boxes"]:

                centerX=bb['x']+int(bb['width']/2)
                centerY=bb['y']+int(bb['height']/2)
                center=(centerX, centerY)

                result_msg = Detection2D()
                result_msg.header.stamp = time_now
                result_msg.header.frame_id = self.frame_id

                # object with hypthothesis
                obj_hyp = ObjectHypothesisWithPose()
                obj_hyp.hypothesis.class_id = bb['label'] #str(self.ei_classifier.labels.index(bb['label']))
                obj_hyp.hypothesis.score = bb['value']
                obj_hyp.pose.pose.position.x = float(centerX)
                obj_hyp.pose.pose.position.y = float(centerY)
                result_msg.results.append(obj_hyp)

                # bounding box
                result_msg.bbox.center.position.x = float(centerX)
                result_msg.bbox.center.position.y = float(centerY)
                result_msg.bbox.center.theta = 0.0
                result_msg.bbox.size_x = float(bb['width'])
                result_msg.bbox.size_y = float(bb['height'])
                result_msg.id = bb['label']


                results_msg.detections.append(result_msg)

                # image
                if self.show_extra_classification_info:
                    self.get_logger().info('%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                if self.show_overlay:

                    print(self.ei_classifier.labels.index(bb['label']))
                    
                    box_color = (0,255,0)

                    img_res = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), box_color, 1)
                    if self.show_labels_on_image:
                        composed_label = bb['label'] +' '+str(round(bb['value'],2))
                        img_res = cv2.putText(img_res, composed_label, (bb['x'], bb['y']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color,1)
                    if self.show_center:
                        img_res = cv2.circle(img_res, center, 3, box_color, 1)
                        composed_center = '('+str(centerX)+','+str(centerY)+')'
                        img_res = cv2.putText(img_res, composed_center, (centerX, centerY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color,1)


        cropped=cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

        # publish message
        self.image_publisher.publish(self.cv_bridge.cv2_to_imgmsg(cropped,"bgr8"))
        self.results_publisher.publish(results_msg)
        self.info_msg.header.stamp = time_now
        self.info_publisher.publish(self.info_msg)

        self.occupied= False
        self.timer_classify.cancel()
        
    
    class EI_Classifier:
        def __init__(self, modelfile, logger):
            self.runner = None
            self.labels = None
            self.logger = logger
            with ImageImpulseRunner(modelfile) as self.runner:
                try:
                    self.model_info = self.runner.init()
                    #print(self.model_info)
                    self.logger.info('Model loaded successfully!')
                    self.logger.info('Model owner: '+ self.model_info['project']['owner'])
                    self.logger.info('Model name: ' + self.model_info['project']['name'])
                    self.logger.info('Model version: ' + str(self.model_info['project']['deploy_version']))
                    self.logger.info('Model type: '+ self.model_info['model_parameters']['model_type'])
                    self.labels = self.model_info['model_parameters']['labels']
                    
                except:
                    self.logger.error('Issue on opening .eim file')
                    if (self.runner):
                        self.runner.stop()


        def __del__(self):
            if (self.runner):
                self.runner.stop()  

        def classify(self, img):
            try:
                # classification
                features, cropped = self.runner.get_features_from_image(img)
                res = self.runner.classify(features)
                return features, cropped, res
            except:
                # somenthing went wrong >_<
                self.logger.error('Error on classification')




def main():
    rclpy.init()
    node = EI_Image_node()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
   main()



