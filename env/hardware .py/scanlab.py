# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:48:06 2021

@author: I. Sutyrina

Client to connect to the scanlab container in the kubernetes cluster for moving 
mirror axis inside of the scanhead. Moving them will refract the laser beam in 
a new angle, thus it will reach a new position on a sample. 
   
"""

#import scannerservice_pb2_grpc
#import scannerservice_pb2
import grpc 
import logging

class ScannerClient ():
    
    def __init__(self, host):
        # establish the connection to the scanlab container
        try:            
            self.channel = grpc.insecure_channel(host)
            self.stub = scannerservice_pb2_grpc.HardwareControllerStub(self.channel)
            logging.info("Connection to scanner service established")
        except:
            logging.error("Impossible to rpc connect to the scanner service")
    
    def jump(self, position):
        return 0
        # move mirror axis 
        pos = scannerservice_pb2.Position(x = position[0],y = position[1])
        self.stub.JumpToPosition(pos)
        
    
    