# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:17:20 2021

@author: I. Sutyrina

Client to connect to the camera running in the kubernetes cluster for taking a
picture to assess the agent performance. 

"""

from proto.cameraservice import cameraservice_pb2 
from proto.cameraservice import cameraservice_pb2_grpc
import logging
from io import BytesIO 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import grpc

VIDEO_ADDRESSE = 'cameraservice.sally.svc.cluster.local:50051'

class CameraClient():
    
    def __init__(self):
        try:
            self.channel = grpc.insecure_channel(VIDEO_ADDRESSE) 
            self.stub = cameraservice_pb2_grpc.HardwareControllerStub(self.channel) 
            logging.info("Connection to camera service established")
        except:
            logging.error("Impossible to rpc connect to the camera service")

    def take_picture(self):
        request = cameraservice_pb2.PictureConfig(data_format=cameraservice_pb2.DataFormat.BMP)
        result_generator = self.stub.TakePicture(request)
        img_data = BytesIO()
        # get data in chunks  
        for res in result_generator:        
            img_data.write(res.data) 
        print('[Data is received]')  
        # transfrom and display an image 
        img_data.seek(0)    
        img_data_arr = np.array(Image.open(img_data))
        #return img_data_arr
        #plt.imshow(img_data_arr)
        #plt.axis('off')
        #plt.show()
        #return img_data


        
        