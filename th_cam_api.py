from import_clr import *

clr.AddReference("ManagedIR16Filters")

from Lepton import CCI
from IR16Filters import IR16Capture, NewIR16FrameEvent, NewBytesFrameEvent
from System.Drawing import ImageConverter
from System import Array, Byte
#from matplotlib import pyplot as plt
import numpy
import time
from datetime import datetime

lep, = (dev.Open()
        for dev in CCI.GetDevices())

# uncomment the following if running in jupyter
#%matplotlib inline

#print(lep.sys.GetCameraUpTime())

# frame callback function
# this will be called everytime a new frame comes in from the camera
numpyArr = None
capture = None
image_ct = datetime.now()
ini = 0
def getFrameRaw(arr, width, height):
    global numpyArr,image_ct,ini
    #numpyArr = None
    ini = 0
    numpyArr = numpy.fromiter(arr, dtype="uint16").reshape(height, width)
    
    reset_delay = (datetime.now()-image_ct).seconds
    if reset_delay>0:
        #print('reset')
        #print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        ini = 1
    image_ct = datetime.now()
    
    
def start():
    global capture
    # Build an IR16 capture device
    capture = IR16Capture()
    capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(getFrameRaw))
    
    capture.RunGraph()
    
def frame():
    #print (numpyArr)
    return numpyArr,ini
def stop():
    global capture
    capture.StopGraph()
    capture.Dispose()