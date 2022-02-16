import cv2
import numpy as np
import os,sys
import time
from datetime import datetime
#import imutils
import th_cam_api as th
#import lock_api as key
from configparser import ConfigParser
#import face_recognition
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLineEdit,QFormLayout,QDesktopWidget, QDialog, QInputDialog,QMessageBox,QCheckBox, QFrame,QGroupBox
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt, QTimer
from PyQt5.QtGui import QPixmap,QColor, QPen,QPalette
import warnings
import winsound
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
#from imutils import face_utils
from box_utils import *
#import db_api as db
#import emotion_api as EM
#import run_api as FR


warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

onnx_path = 'res/UltraLight/models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name


  
model = 'IV-TS001'
ver = 'v4.4.10'
cfg = ConfigParser()

## 參數
cfg.read('config.ini')
high = float(cfg['degree']['high'])
deg_dict = float(cfg['degree']['dict'])

th_high = float(cfg['param']['th_high'])
th_low = float(cfg['param']['th_low'])
th_diff = float(cfg['param']['th_diff'])
th_rate = float(cfg['param']['th_rate'])
th_high_envir = float(cfg['param']['th_high_envir'])
th_low_envir = float(cfg['param']['th_low_envir'])
th_envir = float(cfg['param']['th_envir'])
face_high_deg = float(cfg['param']['face_high_deg'])
face_low_deg = float(cfg['param']['face_low_deg'])

face_detection_model= cfg['param']['face_detection_model']

th_face_box_min = float(cfg['param']['th_face_box_min'])
th_face_box_max = float(cfg['param']['th_face_box_max'])

degree_eq_1 = float(cfg['param']['degree_eq_1'])
degree_eq_2 = float(cfg['param']['degree_eq_2'])

port1 = int(cfg['cam']['port1'])
port2 = int(cfg['cam']['port2'])
size_rate = float(cfg['cam']['size_rate'])
deltax = int(cfg['cam']['deltax'])
deltay = int(cfg['cam']['deltay'])

rgb_thermal_rate = float(cfg['cam']['rgb_thermal_rate'])
win_rate = float(cfg['cam']['win_rate'])

colorMapType = int(cfg['param']['colorMapType'])

distance = int(cfg['param']['distance'])

edge_x1 = int(cfg['param']['edge_x1'])
edge_y1 = int(cfg['param']['edge_y1'])
edge_x2 = int(cfg['param']['edge_x2'])
edge_y2 = int(cfg['param']['edge_y2'])

en_mode = int(cfg['param']['en_mode'])

calib_mode = int(cfg['param']['calib_mode'])
image_mode = int(cfg['param']['image_mode'])
high_low_mode = int(cfg['param']['high_low_mode'])
edge_mode = int(cfg['param']['edge_mode'])
flip_mode = int(cfg['param']['flip_mode'])

left_logo_display = int(cfg['param']['left_logo_display'])
right_logo_display = int(cfg['param']['right_logo_display'])
title_display = int(cfg['param']['title_display'])

reset_ct = int(cfg['param']['reset_ct'])

sound_switch = cfg['param']['sound_switch']

photo_switch = cfg['param']['photo_switch']

d_th = float(cfg['param']['d_th'])

distance = float(cfg['calib_param']['distance'])
distance1 = float(cfg['calib_param']['distance1'])
distance2 = float(cfg['calib_param']['distance2'])
distance3 = float(cfg['calib_param']['distance3'])
distance4 = float(cfg['calib_param']['distance4'])
distance5 = float(cfg['calib_param']['distance5'])
distance6 = float(cfg['calib_param']['distance6'])
distance7 = float(cfg['calib_param']['distance7'])
distance8 = float(cfg['calib_param']['distance8'])
distance9 = float(cfg['calib_param']['distance9'])
distance10 = float(cfg['calib_param']['distance10'])
distance11 = float(cfg['calib_param']['distance11'])
distance12 = float(cfg['calib_param']['distance12'])
distance13 = float(cfg['calib_param']['distance13'])
distance14 = float(cfg['calib_param']['distance14'])

if not os.path.isdir('./data') and photo_switch.lower()=='on':
    os.makedirs('data')

logo_img = cv2.imread("res/IV_Logo.jpg", -1)
logo_img = cv2.resize( logo_img, (256,256))
facelist = [logo_img]*5
templist = ['']*5
check_templist = [0]*5
facesencodelist = [[]]*5
ct=0
ctt=0
image_ct = datetime.now()
reset_time = 0
reset_check = False
tempImage =0
rgb_temp = 0
faces_temp = []
warn_count =0
temptemp = 0
high_deg_temp = 0
low_deg_temp = 0
rate = 1
ini = 1
tt = 0
last_High_temp_check = 0
last_low_temp_check = 0
delta_temp = 0
emotion = ''
name = ''

if en_mode == 1:
    cv2.namedWindow('img',0)


rgbimg = np.zeros((480,640,3), np.uint8)
rgbimg.fill(255)
thimg = np.zeros((480,640,3), np.uint8)
thimg.fill(255)

rgbimg1 = np.zeros((480,640,3), np.uint8)
rgbimg1.fill(255)

# color map
def generate_colour_map():
    """
    Conversion of the colour map from GetThermal to a numpy LUT:
        https://github.com/groupgets/GetThermal/blob/bb467924750a686cc3930f7e3a253818b755a2c0/src/dataformatter.cpp#L6
    """

    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    #colorMaps
    colormap_grayscale = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255];

    colormap_rainbow = [1, 3, 74, 0, 3, 74, 0, 3, 75, 0, 3, 75, 0, 3, 76, 0, 3, 76, 0, 3, 77, 0, 3, 79, 0, 3, 82, 0, 5, 85, 0, 7, 88, 0, 10, 91, 0, 14, 94, 0, 19, 98, 0, 22, 100, 0, 25, 103, 0, 28, 106, 0, 32, 109, 0, 35, 112, 0, 38, 116, 0, 40, 119, 0, 42, 123, 0, 45, 128, 0, 49, 133, 0, 50, 134, 0, 51, 136, 0, 52, 137, 0, 53, 139, 0, 54, 142, 0, 55, 144, 0, 56, 145, 0, 58, 149, 0, 61, 154, 0, 63, 156, 0, 65, 159, 0, 66, 161, 0, 68, 164, 0, 69, 167, 0, 71, 170, 0, 73, 174, 0, 75, 179, 0, 76, 181, 0, 78, 184, 0, 79, 187, 0, 80, 188, 0, 81, 190, 0, 84, 194, 0, 87, 198, 0, 88, 200, 0, 90, 203, 0, 92, 205, 0, 94, 207, 0, 94, 208, 0, 95, 209, 0, 96, 210, 0, 97, 211, 0, 99, 214, 0, 102, 217, 0, 103, 218, 0, 104, 219, 0, 105, 220, 0, 107, 221, 0, 109, 223, 0, 111, 223, 0, 113, 223, 0, 115, 222, 0, 117, 221, 0, 118, 220, 1, 120, 219, 1, 122, 217, 2, 124, 216, 2, 126, 214, 3, 129, 212, 3, 131, 207, 4, 132, 205, 4, 133, 202, 4, 134, 197, 5, 136, 192, 6, 138, 185, 7, 141, 178, 8, 142, 172, 10, 144, 166, 10, 144, 162, 11, 145, 158, 12, 146, 153, 13, 147, 149, 15, 149, 140, 17, 151, 132, 22, 153, 120, 25, 154, 115, 28, 156, 109, 34, 158, 101, 40, 160, 94, 45, 162, 86, 51, 164, 79, 59, 167, 69, 67, 171, 60, 72, 173, 54, 78, 175, 48, 83, 177, 43, 89, 179, 39, 93, 181, 35, 98, 183, 31, 105, 185, 26, 109, 187, 23, 113, 188, 21, 118, 189, 19, 123, 191, 17, 128, 193, 14, 134, 195, 12, 138, 196, 10, 142, 197, 8, 146, 198, 6, 151, 200, 5, 155, 201, 4, 160, 203, 3, 164, 204, 2, 169, 205, 2, 173, 206, 1, 175, 207, 1, 178, 207, 1, 184, 208, 0, 190, 210, 0, 193, 211, 0, 196, 212, 0, 199, 212, 0, 202, 213, 1, 207, 214, 2, 212, 215, 3, 215, 214, 3, 218, 214, 3, 220, 213, 3, 222, 213, 4, 224, 212, 4, 225, 212, 5, 226, 212, 5, 229, 211, 5, 232, 211, 6, 232, 211, 6, 233, 211, 6, 234, 210, 6, 235, 210, 7, 236, 209, 7, 237, 208, 8, 239, 206, 8, 241, 204, 9, 242, 203, 9, 244, 202, 10, 244, 201, 10, 245, 200, 10, 245, 199, 11, 246, 198, 11, 247, 197, 12, 248, 194, 13, 249, 191, 14, 250, 189, 14, 251, 187, 15, 251, 185, 16, 252, 183, 17, 252, 178, 18, 253, 174, 19, 253, 171, 19, 254, 168, 20, 254, 165, 21, 254, 164, 21, 255, 163, 22, 255, 161, 22, 255, 159, 23, 255, 157, 23, 255, 155, 24, 255, 149, 25, 255, 143, 27, 255, 139, 28, 255, 135, 30, 255, 131, 31, 255, 127, 32, 255, 118, 34, 255, 110, 36, 255, 104, 37, 255, 101, 38, 255, 99, 39, 255, 93, 40, 255, 88, 42, 254, 82, 43, 254, 77, 45, 254, 69, 47, 254, 62, 49, 253, 57, 50, 253, 53, 52, 252, 49, 53, 252, 45, 55, 251, 39, 57, 251, 33, 59, 251, 32, 60, 251, 31, 60, 251, 30, 61, 251, 29, 61, 251, 28, 62, 250, 27, 63, 250, 27, 65, 249, 26, 66, 249, 26, 68, 248, 25, 70, 248, 24, 73, 247, 24, 75, 247, 25, 77, 247, 25, 79, 247, 26, 81, 247, 32, 83, 247, 35, 85, 247, 38, 86, 247, 42, 88, 247, 46, 90, 247, 50, 92, 248, 55, 94, 248, 59, 96, 248, 64, 98, 248, 72, 101, 249, 81, 104, 249, 87, 106, 250, 93, 108, 250, 95, 109, 250, 98, 110, 250, 100, 111, 251, 101, 112, 251, 102, 113, 251, 109, 117, 252, 116, 121, 252, 121, 123, 253, 126, 126, 253, 130, 128, 254, 135, 131, 254, 139, 133, 254, 144, 136, 254, 151, 140, 255, 158, 144, 255, 163, 146, 255, 168, 149, 255, 173, 152, 255, 176, 153, 255, 178, 155, 255, 184, 160, 255, 191, 165, 255, 195, 168, 255, 199, 172, 255, 203, 175, 255, 207, 179, 255, 211, 182, 255, 216, 185, 255, 218, 190, 255, 220, 196, 255, 222, 200, 255, 225, 202, 255, 227, 204, 255, 230, 206, 255, 233, 208]

    colourmap_ironblack = [
        255, 255, 255, 253, 253, 253, 251, 251, 251, 249, 249, 249, 247, 247,
        247, 245, 245, 245, 243, 243, 243, 241, 241, 241, 239, 239, 239, 237,
        237, 237, 235, 235, 235, 233, 233, 233, 231, 231, 231, 229, 229, 229,
        227, 227, 227, 225, 225, 225, 223, 223, 223, 221, 221, 221, 219, 219,
        219, 217, 217, 217, 215, 215, 215, 213, 213, 213, 211, 211, 211, 209,
        209, 209, 207, 207, 207, 205, 205, 205, 203, 203, 203, 201, 201, 201,
        199, 199, 199, 197, 197, 197, 195, 195, 195, 193, 193, 193, 191, 191,
        191, 189, 189, 189, 187, 187, 187, 185, 185, 185, 183, 183, 183, 181,
        181, 181, 179, 179, 179, 177, 177, 177, 175, 175, 175, 173, 173, 173,
        171, 171, 171, 169, 169, 169, 167, 167, 167, 165, 165, 165, 163, 163,
        163, 161, 161, 161, 159, 159, 159, 157, 157, 157, 155, 155, 155, 153,
        153, 153, 151, 151, 151, 149, 149, 149, 147, 147, 147, 145, 145, 145,
        143, 143, 143, 141, 141, 141, 139, 139, 139, 137, 137, 137, 135, 135,
        135, 133, 133, 133, 131, 131, 131, 129, 129, 129, 126, 126, 126, 124,
        124, 124, 122, 122, 122, 120, 120, 120, 118, 118, 118, 116, 116, 116,
        114, 114, 114, 112, 112, 112, 110, 110, 110, 108, 108, 108, 106, 106,
        106, 104, 104, 104, 102, 102, 102, 100, 100, 100, 98, 98, 98, 96, 96,
        96, 94, 94, 94, 92, 92, 92, 90, 90, 90, 88, 88, 88, 86, 86, 86, 84, 84,
        84, 82, 82, 82, 80, 80, 80, 78, 78, 78, 76, 76, 76, 74, 74, 74, 72, 72,
        72, 70, 70, 70, 68, 68, 68, 66, 66, 66, 64, 64, 64, 62, 62, 62, 60, 60,
        60, 58, 58, 58, 56, 56, 56, 54, 54, 54, 52, 52, 52, 50, 50, 50, 48, 48,
        48, 46, 46, 46, 44, 44, 44, 42, 42, 42, 40, 40, 40, 38, 38, 38, 36, 36,
        36, 34, 34, 34, 32, 32, 32, 30, 30, 30, 28, 28, 28, 26, 26, 26, 24, 24,
        24, 22, 22, 22, 20, 20, 20, 18, 18, 18, 16, 16, 16, 14, 14, 14, 12, 12,
        12, 10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 9,
        2, 0, 16, 4, 0, 24, 6, 0, 31, 8, 0, 38, 10, 0, 45, 12, 0, 53, 14, 0,
        60, 17, 0, 67, 19, 0, 74, 21, 0, 82, 23, 0, 89, 25, 0, 96, 27, 0, 103,
        29, 0, 111, 31, 0, 118, 36, 0, 120, 41, 0, 121, 46, 0, 122, 51, 0, 123,
        56, 0, 124, 61, 0, 125, 66, 0, 126, 71, 0, 127, 76, 1, 128, 81, 1, 129,
        86, 1, 130, 91, 1, 131, 96, 1, 132, 101, 1, 133, 106, 1, 134, 111, 1,
        135, 116, 1, 136, 121, 1, 136, 125, 2, 137, 130, 2, 137, 135, 3, 137,
        139, 3, 138, 144, 3, 138, 149, 4, 138, 153, 4, 139, 158, 5, 139, 163,
        5, 139, 167, 5, 140, 172, 6, 140, 177, 6, 140, 181, 7, 141, 186, 7,
        141, 189, 10, 137, 191, 13, 132, 194, 16, 127, 196, 19, 121, 198, 22,
        116, 200, 25, 111, 203, 28, 106, 205, 31, 101, 207, 34, 95, 209, 37,
        90, 212, 40, 85, 214, 43, 80, 216, 46, 75, 218, 49, 69, 221, 52, 64,
        223, 55, 59, 224, 57, 49, 225, 60, 47, 226, 64, 44, 227, 67, 42, 228,
        71, 39, 229, 74, 37, 230, 78, 34, 231, 81, 32, 231, 85, 29, 232, 88,
        27, 233, 92, 24, 234, 95, 22, 235, 99, 19, 236, 102, 17, 237, 106, 14,
        238, 109, 12, 239, 112, 12, 240, 116, 12, 240, 119, 12, 241, 123, 12,
        241, 127, 12, 242, 130, 12, 242, 134, 12, 243, 138, 12, 243, 141, 13,
        244, 145, 13, 244, 149, 13, 245, 152, 13, 245, 156, 13, 246, 160, 13,
        246, 163, 13, 247, 167, 13, 247, 171, 13, 248, 175, 14, 248, 178, 15,
        249, 182, 16, 249, 185, 18, 250, 189, 19, 250, 192, 20, 251, 196, 21,
        251, 199, 22, 252, 203, 23, 252, 206, 24, 253, 210, 25, 253, 213, 27,
        254, 217, 28, 254, 220, 29, 255, 224, 30, 255, 227, 39, 255, 229, 53,
        255, 231, 67, 255, 233, 81, 255, 234, 95, 255, 236, 109, 255, 238, 123,
        255, 240, 137, 255, 242, 151, 255, 244, 165, 255, 246, 179, 255, 248,
        193, 255, 249, 207, 255, 251, 221, 255, 253, 235, 255, 255, 24]

    def chunk(ulist, step):
        return map(lambda i: ulist[i: i + step], range(0, len(ulist), step))

    if (colorMapType == 1):
        chunks = chunk(colormap_rainbow, 3)
    elif (colorMapType == 2):
        chunks = chunk(colormap_grayscale, 3)
    else:
        chunks = chunk(colourmap_ironblack, 3)

    red = []
    green = []
    blue = []

    for chunk in chunks:
        red.append(chunk[0])
        green.append(chunk[1])
        blue.append(chunk[2])

    lut[:, 0, 0] = blue

    lut[:, 0, 1] = green

    lut[:, 0, 2] = red

    return lut

def ktof(val):
    return round(((1.8 * ktoc(val) + 32.0)), 2)

def ktoc(val):
    return round(((val - 27315) / 100.0), 2)

def display_temperatureF(img, val_k, loc, color):
    val = ktof(val_k)
    x, y = loc
    if en_mode == 1:
        cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.line(img, (x - 20, y), (x + 20, y), color, 3)
        cv2.line(img, (x, y - 20), (x, y + 20), color, 3)
    
    return  val 

def display_temperatureC(img, val_k, loc, color):
    val = ktoc(val_k)
    x, y = loc
    if  en_mode == 1:
        cv2.putText(img,"{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.line(img, (x - 20, y), (x + 20, y), color, 3)
        cv2.line(img, (x, y - 20), (x, y + 20), color, 3)
    
    return  val   

def display_temperatureF_face(img, val_k, loc, color):
    val = ktof(val_k)
    x, y = loc
    if en_mode == 1:
        
        cv2.line(img, (x - 10, y), (x + 10, y), color, 2)
        cv2.line(img, (x, y - 10), (x, y + 10), color, 2)
    
    return  val 

def display_temperatureC_face(img, val_k, loc, color):
    val = ktoc(val_k)
    x, y = loc
    if  en_mode == 1:
        
        cv2.line(img, (x - 10, y), (x + 10, y), color, 2)
        cv2.line(img, (x, y - 10), (x, y + 10), color, 2)
    
    return  val   

def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)
# old not use
def rgb_face(image):
    global facelist,templist,distance
    X_face_locations = face_recognition.face_locations(image , distance ,model= face_detection_model)
    
    faces = []
    
    for y1,x2,y2,x1 in X_face_locations:
        
        
        rate = ((x2-x1)+(y2-y1))/2
        if (rate<th_face_box_min or rate>th_face_box_max) or (x1 < edge_x1 or y1<edge_y1 or x2>edge_x2 or y2>edge_y2):
            continue
        faces.append((x1,y1,x2,y2))
        
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0,255), 2)
            
        
    return image,faces
# 人臉
def rgb_face_ultra_light(image):
    global facelist,templist,distance
    
    h, w, _ = image.shape
    
    img_mean = np.array([127, 127, 127])
    img = (image - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    
    confidences, boxes = ort_session.run(None, {input_name: img})
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
    
    faces = []
    
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rate = ((x2-x1)+(y2-y1))/2

        if edge_mode == 1:
            if (rate<th_face_box_min or rate>th_face_box_max) or (x1 < edge_x1 or y1<edge_y1 or x2>edge_x2 or y2>edge_y2) :
                continue
        else:
            if (rate<th_face_box_min or rate>th_face_box_max) :
                continue
        
        faces.append((x1,y1,x2,y2))
        
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0,255), 2)
            
        
    return image,faces
# thermal 溫度演算法     
def th_face(image,rgb_im,faces):
    global facelist,templist,facesencodelist,ct,warn_count,check_templist,ctt,rgbimg,rgbimg1,temptemp,ini,high_deg_temp,low_deg_temp,delta_temp,last_High_temp_check,last_low_temp_check
    
    high_deg,low_deg = min_max_temp(image)
 
    if  reset_check :
            print ('H:',high_deg,'L:',low_deg)
            print ('------------------------------')
            print ('bH:',high_deg_temp,'bL:',low_deg_temp)
            
            if abs(round(high_deg,2) - round(high_deg_temp,2))>2 :
                
                if abs(round(low_deg,2) - round(low_deg_temp,2))>2:
                    delta_temp =  0.4
                else:
                    
                    delta_temp = (round(low_deg,2) - round(low_deg_temp,2))
                
            
            else:
                delta_temp = (round(high_deg,2) - round(high_deg_temp,2))

            cv2.putText(rgbimg, "Initializing", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(delta_temp)
            last_low_temp_check = 0
            last_High_temp_check = 0
    else:
        
        if high_deg_temp==0 and low_deg_temp==0:
 
            high_deg_temp = high_deg
            low_deg_temp = low_deg
        if 0.5>abs(high_deg_temp-high_deg)>0 :

            high_deg_temp = high_deg
        elif abs(high_deg_temp-high_deg)>1.5:
            last_High_temp_check+=1
            if last_High_temp_check>2:
                high_deg_temp = high_deg
                last_High_temp_check = 0
            
        if 0.5>abs(low_deg_temp-low_deg)>0 :

            low_deg_temp = low_deg
        elif abs(low_deg_temp-low_deg)>1.5:
            last_low_temp_check+=1
            if last_low_temp_check >2:
                low_deg_temp = low_deg
                last_low_temp_check = 0
            
    if faces == []:
        rgbimg1 = rgbimg.copy()
        return image
    for face in faces:

        
        (x1,y1,x2,y2) = face
        
        rate = ((x2-x1)+(y2-y1))/2
        deg = face_max_temp(image,x1,y1,x2,y2)
        if en_mode == 1:
            print (high_deg,low_deg , deg,rate)
        
        if  reset_check :
            
            if delta_temp < 0:
                
                deg = deg+delta_temp
                
            else:
                
                deg = deg-delta_temp
          
            
        if calib_mode == 1:
        
            deg = deg+degree_eq_2
           
            if 140<=rate:
                deg=deg+distance
            elif 130<=rate<140:
                deg=deg+distance1
            elif 120<=rate<130:
                deg=deg+distance2
            elif 110<=rate<120:
                deg=deg+distance3
            elif 100<=rate<110:
                deg=deg+distance4
            elif 90<=rate<100:
                deg=deg+distance5
            elif 80<=rate<90:
                deg=deg+distance6
            elif 70<=rate<80:
                deg=deg+distance7
            elif 60<=rate<70:
                deg=deg+distance8
            elif 50<=rate<60:
                deg=deg+distance9
            elif 40<=rate<50:
                deg=deg+distance10
            elif 30<=rate<40:
                deg=deg+distance11
            elif 20<=rate<30:
                deg=deg+distance12
            elif 10<=rate<20:
                deg=deg+distance13
            elif 0<=rate<10:
                deg=deg+distance14
            else:
                pass
        else:
            deg = deg+degree_eq_2
        

        temptemp = deg
        
        if deg >=high :
            
            
            capture_photo = 0
            if sound_switch.lower() =='on'  and (time.clock()-ctt)>1 :
                                
                ctt = time.clock()
  
                
                winsound.PlaySound(None, winsound.SND_ASYNC)
                winsound.PlaySound('res/alert.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)

                if en_mode == 1:
                    print ('Sound')
                if photo_switch.lower()=='on':
                    capture_photo = 1
            
            image_save = rgbimg.copy()
            cv2.rectangle(image_save, (x1, y1), (x2, y2), (0, 0,255), 2)
            cv2.putText(image_save, "{0:.1f} C".format(deg), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX,
      1.2, (0, 0, 255), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0,255), 2)
            cv2.putText(image, "{0:.1f} degC".format(deg), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX,
      1.2, (0, 0, 255), 2)
            
            cv2.putText(rgbimg, "{0:.1f} ".format(deg), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            
            if photo_switch.lower()=='on' and capture_photo == 1:
                image_path = './data/'+time.strftime("%Y%m%d", time.localtime())
                if not os.path.isdir(image_path) :
                    os.makedirs(image_path)
                cv2.imwrite(image_path+'/warning_'+time.strftime("%H%M%S", time.localtime())+'_'+"{0:.1f}".format(deg)+".jpg",image_save)
        elif high>deg>(high-deg_dict):
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, "{0:.1f} degC".format(deg), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX,
      1.2, (0, 255, 255), 2)
            cv2.putText(rgbimg, "{0:.1f} ".format(deg), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX,  1.2, (0, 255, 255), 2)
        elif deg>40 or deg<25:
            continue
        
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255,0), 2)
            cv2.putText(image, "{0:.1f} degC".format(deg), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX,
      1.2, (0, 255, 0), 2)
            cv2.putText(rgbimg, "{0:.1f} ".format(deg), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    rgbimg1 = rgbimg.copy()
    
    return image

#人臉最高溫
def face_max_temp(image,x1,y1,x2,y2):
    
    
    
    imgt = tempImage[y1:y2,x1:x2]
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(imgt)
    
    maxLoc = (maxLoc[0] + x1, maxLoc[1] + y1)
    
    if ktoc(maxVal) < high:
        deg = display_temperatureC_face(image, maxVal, maxLoc, (0, 255, 0)) #displays max temp at max temp location on image
    else:
        deg = display_temperatureC_face(image, maxVal, maxLoc, (0, 0, 255)) #displays max temp at max temp location on image
    
    
    
    return deg 
    
def min_max_temp(image):
    
    imgt = tempImage
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(imgt)

    high = display_temperatureC(image, maxVal, maxLoc, (0, 0, 255)) #displays max temp at max temp location on image
    low = display_temperatureC(image, minVal, minLoc, (255, 0, 0)) #displays max temp at max temp location on image
    
    
    return high,low
#校正演算法
def move_img(img):
    rows, cols, channal = img.shape
    MAT = np.float32([[1, 0, -deltax], [0, 1, -deltay]])
    mov = cv2.warpAffine(img, MAT, (cols, rows),borderValue =(255, 255, 255))
    return mov

class ThermalVideo(QtCore.QObject):
    
    image_thermal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=port1, parent=None):
        super().__init__(parent)
        ## Ubuntu Get Thermal Cam
        # self.camera = cv2.VideoCapture(camera_port)
        # self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        # self.camera.set(cv2.CAP_PROP_CONVERT_RGB, False)
        ## Windows Get Thermal Cam
        th.start()
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        global tempImage,image_ct,reset_check,reset_time,ini,delta_temp
        
        
        if (event.timerId() != self.timer.timerId()):
            
            return

        ini_check =0
        data,ini_check = th.frame()
        #read, data = self.camera.read()

        if data is not None:
            if flip_mode == 1:
                data = cv2.flip(data,0)
            '''
            data = data[:-2,:]
            data = cv2.resize(data[:,:], (640, 480))
            tempImage = data.copy()
            data = cv2.LUT(raw_to_8bit(data), generate_colour_map())
            
            self.image_thermal.emit(data)
            '''    
            data = cv2.resize(data[:,:], (640, 480))
            tempImage = data.copy()
            data = cv2.LUT(raw_to_8bit(data), generate_colour_map())
            
            self.image_thermal.emit(data)
        if ini_check == 1:
            if reset_check == False:
                reset_check = True
                reset_time = datetime.now()
                if en_mode == 1:

                    print('reset')
                    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
                
        if  reset_check == True and reset_time != 0:

            if (((datetime.now()-reset_time).seconds)>reset_ct ) :

                if en_mode == 1:

                    print ('Initialize Finish')
                reset_check = False
                reset_time = 0
                delta_temp = 0
                #ini = 0
        
class RGBVideo(QtCore.QObject):
    
    image_rgb = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=port2, parent=None):
        try:
            super().__init__(parent)
            self.camera = cv2.VideoCapture(camera_port)
            self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print (self.width,self.height)
            self.timer = QtCore.QBasicTimer()
        except ValueError:
            print ('RGB Get Error')
            
    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            if flip_mode == 1:
                data = cv2.flip(data,0)
            self.image_rgb.emit(data)
#not use UI SHOW 人臉
class RecordMessage(QtCore.QObject):
    
    
    image_message = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        global facelist,templist
        
        cv2.putText(facelist[0], str(templist[0]), (100, 240), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 1)
        cv2.putText(facelist[1], str(templist[1]), (100, 240), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 1)
        cv2.putText(facelist[2], str(templist[2]), (100, 240), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 1)
        cv2.putText(facelist[3], str(templist[3]), (100, 240), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 1)
        cv2.putText(facelist[4], str(templist[4]), (100, 240), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 255), 1)
        
        faces = np.hstack((facelist[4],facelist[3],facelist[2],facelist[1],facelist[0]))
        
        if (event.timerId() != self.timer.timerId()):
            return

        
        self.image_message.emit(faces)
        
class ImageShow(QtCore.QObject):
    
    
    image_show = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        
        
        global rgbimg1,thimg,rgbimg
        
        
        
        if (event.timerId() != self.timer.timerId()):
            return
        try:
            #image = np.hstack((rgbimg,thimg))
            if en_mode == 1:
                cv2.imshow('img',thimg)
            image = rgbimg1
        except ValueError:
            print ('show image error')
        
        self.image_show.emit(image)
#RGB處理
class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self,  parent=None):
        super().__init__(parent)
       
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    

    def image_rgb_slot(self, image_rgb):
        global rgb_temp,faces_temp,rgbimg,name,emotion
        
        
        image_rgb=cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        
        image_rgb = cv2.resize( image_rgb, (0,0), fx=size_rate, fy=size_rate)
        h,w = image_rgb.shape[:-1]
        image_rgb = image_rgb[ int(h/2)-240:int(h/2)+240, int(w/2)-320:int(w/2)+320]
        image_rgb = move_img(image_rgb)

        rgb_temp = image_rgb.copy()
        image_rgb,faces = rgb_face_ultra_light(image_rgb)

        
        rgb_temp=cv2.cvtColor(rgb_temp, cv2.COLOR_RGB2BGR)
        
        faces_temp = faces
        
        

        image_rgb=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        h1,w1 = image_rgb.shape[:-1]
        

        if edge_mode == 1:
            cv2.rectangle(image_rgb, (edge_x1, edge_y1), (edge_x2, edge_y2), (0, 255, 0), 2)
        
        rgbimg = image_rgb.copy()
        self.image = self.get_qimage(image_rgb)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()
        
    

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
#Thermal處理        
class ThermalWidget(QtWidgets.QWidget):
    def __init__(self,  parent=None):
        super().__init__(parent)
        
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    

    
        
    def image_thermal_slot(self, image_thermal):
        global rgb_temp,faces_temp,thimg#,image_ct,reset_check,reset_time

        image_thermal = th_face(image_thermal,rgb_temp,faces_temp)
        

        
        h1,w1 = image_thermal.shape[:-1]
        

        if edge_mode == 1:
            cv2.rectangle(image_thermal, (edge_x1, edge_y1), (edge_x2, edge_y2), (0, 255, 0), 2)
        thimg = image_thermal.copy()
        
        self.image = self.get_qimage(image_thermal)
        
        if self.image.size() != self.size():
            
            self.setFixedSize(self.image.size())
        
        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
#Show        
class ShowImageWidget(QtWidgets.QWidget):
    def __init__(self,  parent=None):
        super().__init__(parent)
        
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    

    
        
    def image_show_slot(self,image_show):

        
        
        image_show = cv2.resize( image_show, (0,0), fx=rgb_thermal_rate*rate, fy=rgb_thermal_rate*rate)
        
        self.image = self.get_qimage(image_show)
        
        if self.image.size() != self.size():
            
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
#not use
class MessageWidget(QtWidgets.QWidget):
    def __init__(self,  parent=None):
        super().__init__(parent)
        
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    

    def message_data_slot(self, image_message):

        
        
        self.image = self.get_qimage(image_message)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.getText()
        self.initUI()
    def initUI(self):
        
        global high,rate
        
        desktop_size = QtWidgets.QDesktopWidget().screenGeometry()
        print (desktop_size.height(),desktop_size.width(),desktop_size.width()/2560)
        rate = desktop_size.width()/2560
        self.img = np.ndarray(())
        self.logo_img = QPixmap("res/NHRI_Logo.jpg") 
        self.logo_img = self.logo_img.scaled(300*rate, 300*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.logo1_img = QPixmap("res/IV_Logo_with_chinese.jpg")
        self.logo1_img = self.logo1_img.scaled(300*rate, 300*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.logo2_img = QPixmap("res/Logo.png")
        self.logo2_img = self.logo2_img.scaled(350*rate*1.5, 350*rate*1.5,Qt.KeepAspectRatio, Qt.FastTransformation)
        
        self.label_text = QtWidgets.QLabel('')
        self.label_text.setFont(QtGui.QFont("Roman times",50*rate,QtGui.QFont.Bold))
        self.title_img = QPixmap("res/title.png") 
        self.title_img = self.title_img.scaled(1000*rate, 1000*rate,Qt.KeepAspectRatio, Qt.FastTransformation)
        
        self.icon_member_name = QtWidgets.QLabel()
        self.icon_weight = QtWidgets.QLabel()
        self.icon_height = QtWidgets.QLabel()
        self.icon_blood = QtWidgets.QLabel()
        self.icon_emotion = QtWidgets.QLabel()
        self.icon_temp = QtWidgets.QLabel()
        self.icon_cal = QtWidgets.QLabel()
        
        
        self.label_member_name_data = QtWidgets.QLabel()
        self.label_weight_data = QtWidgets.QLabel('Developing...')
        self.label_height_data = QtWidgets.QLabel('Developing...')
        self.label_blood_data = QtWidgets.QLabel('Developing...')
        self.label_emotion_data = QtWidgets.QLabel()
        self.label_temp_data = QtWidgets.QLabel()
        self.label_cal_data = QtWidgets.QLabel('Developing...')
        
        self.label_calibrate = QtWidgets.QLabel("{0:.1f}".format(degree_eq_2))
        
        self.icon_member_name.setPixmap(QtGui.QPixmap('res/p.png').scaled(100*rate, 100*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.icon_weight.setPixmap(QtGui.QPixmap('res/w.png').scaled(100*rate, 100*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.icon_blood.setPixmap(QtGui.QPixmap('res/h.png').scaled(100*rate, 100*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.icon_emotion.setPixmap(QtGui.QPixmap('res/e.jpg').scaled(100*rate, 100*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.icon_temp.setPixmap(QtGui.QPixmap('res/d.png').scaled(100*rate, 100*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.icon_height.setPixmap(QtGui.QPixmap('res/t.png').scaled(100*rate, 100*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.icon_cal.setPixmap(QtGui.QPixmap('res/burn.png').scaled(100*rate, 100*rate,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        
        self.label_member_name_data.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
       
        self.label_weight_data.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        self.label_height_data.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        self.label_blood_data.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        self.label_emotion_data.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        self.label_temp_data.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        self.label_cal_data.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        self.label_calibrate.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        
        
        self.timer_id = 0
        self.label_date = QtWidgets.QLabel()
        self.label_time = QtWidgets.QLabel()
        self.icon_date = QtWidgets.QLabel()
        self.icon_time = QtWidgets.QLabel()
        
        self.icon_date.setPixmap(QtGui.QPixmap('res/calendar.png').scaled(100*rate, 100*rate,Qt.KeepAspectRatio, Qt.FastTransformation))
        self.icon_time.setPixmap(QtGui.QPixmap('res/alarm.png').scaled(100*rate, 100*rate,Qt.KeepAspectRatio, Qt.FastTransformation))
        #self.icon_date.scaledToWidth( 400 )
        #self.icon_time.scaledToWidth( 400 )
        self.label_degree = QtWidgets.QLabel()
        self.label_degree.setPixmap(QtGui.QPixmap('res/thermometer.png').scaled(100*rate, 100*rate,Qt.KeepAspectRatio, Qt.FastTransformation))
        self.label_start = QtWidgets.QLabel("Start")
        self.label_change = QtWidgets.QLabel("Change")
        self.label_setting = QtWidgets.QLabel("Setting")
        self.label_cancel = QtWidgets.QLabel("Cancel")
        
        
        self.photo_checkbox = QCheckBox()
        self.sound_checkbox = QCheckBox()
        self.photo_checkbox.setIcon(QtGui.QIcon('res/camera.png'))
        self.photo_checkbox.setIconSize(QtCore.QSize(50*rate, 50*rate))
        self.sound_checkbox.setIcon(QtGui.QIcon('res/notification.png'))
        self.sound_checkbox.setIconSize(QtCore.QSize(50*rate, 50*rate))
        #self.photo_checkbox.setStyleSheet("QCheckBox {spacing: 50px;font-size:25px;} ") #QCheckBox::indicator { width: 500px; height: 500px;}
        #self.sound_checkbox.setStyleSheet("QCheckBox {spacing: 50px;font-size:25px;} ")
        
        if photo_switch.lower() =='on':
            self.photo_checkbox.setChecked(True)
        if sound_switch.lower() =='on':
            self.sound_checkbox.setChecked(True)
        
        self.label_logo = QtWidgets.QLabel("")
        self.label_logo.setPixmap(self.logo_img) 
        self.label_logo1 = QtWidgets.QLabel("")
        self.label_logo1.setPixmap(self.logo1_img)
        
        self.label_logo2 = QtWidgets.QLabel("")
        self.label_logo2.setPixmap(self.logo2_img) 
        
        self.label_title = QtWidgets.QLabel("")
        self.label_title.setPixmap(self.title_img) 
        
        self.label_crop = QtWidgets.QLabel()
        self.label_crop.setPixmap(QtGui.QPixmap("res/co-work.jpg")) 
        
        self.label_date.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        self.label_time.setFont(QtGui.QFont("Roman times",30*rate,QtGui.QFont.Bold))
        #self.label_crop.setFont(QtGui.QFont("Roman times",50,QtGui.QFont.Bold))
        
        #self.label_date.setAlignment(Qt.AlignCenter|QtCore.Qt.AlignLeft)
        #self.label_time.setAlignment(Qt.AlignCenter|QtCore.Qt.AlignLeft)
        self.label_crop.setAlignment(Qt.AlignCenter)
        self.label_logo.setAlignment(Qt.AlignCenter)
        self.label_logo1.setAlignment(Qt.AlignCenter)
        self.label_logo2.setAlignment(Qt.AlignCenter)
        self.textbox = QLineEdit(self)
        font = self.textbox.font()      # lineedit current font
        font.setPointSize(32*rate)               # change it's size
        self.textbox.setFont(font)      # set font 
        self.textbox.setFixedWidth(100*rate)
        self.textbox.setAlignment(Qt.AlignCenter)
        self.textbox.setText(str(high))
        self.label_degree.setAlignment(Qt.AlignCenter)
        
        font1 = self.label_degree.font()      # lineedit current font
        font1.setPointSize(32*rate) 
        self.label_degree.setFont(font1)
        
        
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
        
        pe = QPalette()
        pe.setColor(QPalette.WindowText,Qt.darkRed)
        #self.label_crop.setPalette(pe)
        
        self.face_detection_widget = FaceDetectionWidget()
        self.thermal_widget = ThermalWidget()
        self.imageshow_widget = ShowImageWidget()
        #self.message_widget = MessageWidget()
        
        
        self.openSlot
        
        # TODO: set video port
        self.thermal_video = ThermalVideo()
        
        self.rgb_video = RGBVideo()
        
        self.image_show = ImageShow()
        
        #self.record_message = RecordMessage()

        
        
        image_rgb_slot = self.face_detection_widget.image_rgb_slot
        image_thermal_slot = self.thermal_widget.image_thermal_slot
        image_show_slot = self.imageshow_widget.image_show_slot
        #image_message_slot = self.message_widget.message_data_slot
        
        self.thermal_video.image_thermal.connect(image_thermal_slot)
        self.rgb_video.image_rgb.connect(image_rgb_slot)
        self.image_show.image_show.connect(image_show_slot)
        #self.record_message.image_message.connect(image_message_slot)
        
        self.confirm_button = QtWidgets.QPushButton("Change",self)
        self.confirm_button.setIcon(QtGui.QIcon('res/thermometer1.png'))
        self.confirm_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        self.about_button = QtWidgets.QPushButton("About",self)
        self.about_button.setIcon(QtGui.QIcon('res/about.png'))
        self.about_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        #self.setting_button = QtWidgets.QPushButton("Setting",self)
        self.up_button = QtWidgets.QPushButton(">",self)
        self.up_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        self.down_button = QtWidgets.QPushButton("<",self)
        self.down_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        self.upup_button = QtWidgets.QPushButton(">>",self)
        self.upup_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        self.downdown_button = QtWidgets.QPushButton("<<",self)
        self.downdown_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        self.cancel_button = QtWidgets.QPushButton("Cancel",self)
        self.cancel_button.setIcon(QtGui.QIcon('res/cancel.png'))
        self.cancel_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        
        self.sound_switch_button = QtWidgets.QPushButton("Alarm",self)
        self.photo_switch_button = QtWidgets.QPushButton("Report",self)
        if sound_switch == 'on': 
            
            self.sound_switch_button.setIcon(QtGui.QIcon('res/notification.png'))
       
        else:
            
            self.sound_switch_button.setIcon(QtGui.QIcon('res/disable-alarm.png'))
            
        if photo_switch == 'on':
            
            self.photo_switch_button.setIcon(QtGui.QIcon('res/camera.png'))
            
        else:
            
            self.photo_switch_button.setIcon(QtGui.QIcon('res/camera-off.png'))
        
        self.sound_switch_button.setIconSize(QtCore.QSize(50*rate,50*rate))
        self.photo_switch_button.setIconSize(QtCore.QSize(50*rate,50*rate))
            
        font2 = self.textbox.font()      # lineedit current font
        font2.setPointSize(32*rate) 
        
        self.confirm_button.setFont(font2)
        self.about_button.setFont(font2)
        #self.setting_button.setFont(font2)
        self.cancel_button.setFont(font2)
        self.upup_button.setFont(font2)
        self.downdown_button.setFont(font2)
        self.up_button.setFont(font2)
        self.down_button.setFont(font2)
        self.sound_switch_button.setFont(font2)
        self.photo_switch_button.setFont(font2)
        
        
        #self.rightFrame = QFrame(self)
        #self.rightFrame.setFrameShape(QFrame.StyledPanel)
        #self.rightFrame.setFrameShadow(QFrame.Raised)
        #self.rightFrame.setLineWidth(0.6)
        
        self.label_setting = QtWidgets.QLabel()
        self.label_setting.setPixmap(QtGui.QPixmap('res/settings.png').scaled(100*rate, 100*rate,Qt.KeepAspectRatio, Qt.FastTransformation))
        
        self.groupbox_1 = QGroupBox('Info', self)
        self.groupbox_2 = QGroupBox('Setting', self)
        #self.groupbox_2.setCheckable(True)
        self.groupbox_1.setStyleSheet("QGroupBox{    font-size: 25px;    font-weight: bold;} ")
        self.groupbox_2.setStyleSheet("QGroupBox{    font-size: 25px;    font-weight: bold;} ")#QGroupBox::indicator {  width: 50px;  height: 50px;} QGroupBox::indicator:checked {image: url(:/res/settings.png);}")
        #self.groupbox_1.setIcon(QtGui.QIcon('res/google-forms.png'))
        #self.groupbox_2.setIcon(QtGui.QIcon('res/settings.png'))
        
        mainlayout = QtWidgets.QHBoxLayout()
        #layout2 = QtWidgets.QVBoxLayout()
        #messagelayout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QVBoxLayout()
        layout_degree = QtWidgets.QHBoxLayout()
        layout_logo = QtWidgets.QHBoxLayout()
        layout_date = QtWidgets.QHBoxLayout()
        layout_time = QtWidgets.QHBoxLayout()
        layout_name = QtWidgets.QHBoxLayout()
        layout_weight = QtWidgets.QHBoxLayout()
        layout_height = QtWidgets.QHBoxLayout()
        layout_ppg = QtWidgets.QHBoxLayout()
        layout_emotion = QtWidgets.QHBoxLayout()
        layout_cal = QtWidgets.QHBoxLayout()
        layout_temp = QtWidgets.QHBoxLayout()
        layout_datetime = QtWidgets.QVBoxLayout()
        layout_checkbox = QtWidgets.QHBoxLayout()
        layout_info = QtWidgets.QVBoxLayout()
        layout_setting = QtWidgets.QVBoxLayout()
        layout_function = QtWidgets.QVBoxLayout()
        layout_calibrate = QtWidgets.QHBoxLayout()
        layout3 = QtWidgets.QVBoxLayout()
        #flo=QFormLayout()
        #flo.addRow(self.label_degree,self.textbox)
        layout_degree.addWidget(self.label_degree,0,Qt.AlignCenter |Qt.AlignRight)
        layout_degree.addWidget(self.textbox,0,Qt.AlignCenter |Qt.AlignLeft)
        layout_degree.addWidget(self.confirm_button)
        
        
        #layout.addWidget(self.face_detection_widget)
        #layout.addWidget(self.thermal_widget)
        if title_display == 1:
            layout.addStretch()
            layout.addWidget(self.label_title,0,Qt.AlignTop | Qt.AlignCenter)
            layout.addWidget(self.imageshow_widget,3,Qt.AlignCenter)
            layout.addStretch()
        else:
            layout.addWidget(self.imageshow_widget,Qt.AlignCenter)
        
        
        layout_logo.addStretch()
        #layout_logo.addWidget(self.label_logo2,Qt.AlignCenter)
        layout_logo.addWidget(self.label_logo,Qt.AlignCenter)
        #layout_logo.addWidget(self.label_crop,Qt.AlignCenter)
        #layout2.addStretch()
        #self.label_logo1.setAlignment(QtCore.Qt.AlignCenter)
        layout_logo.addWidget(self.label_logo1,Qt.AlignCenter)
        layout_logo.addStretch()
        #layout2.addLayout(layout_logo)
        #layout2.addStretch()
        #self.label_date.setAlignment(QtCore.Qt.AlignLeft)
        #self.label_time.setAlignment(QtCore.Qt.AlignLeft)
        #self.icon_date.setAlignment(QtCore.Qt.AlignRight)
        #self.icon_time.setAlignment(QtCore.Qt.AlignRight)
        #layout_date.addStretch()

        layout_date.addWidget(self.icon_date,1,Qt.AlignCenter|Qt.AlignRight)
        
        layout_date.addWidget(self.label_date,2,Qt.AlignCenter|Qt.AlignLeft)
        
        layout_time.addWidget(self.icon_time,1,Qt.AlignCenter|Qt.AlignRight)
        
        layout_time.addWidget(self.label_time,2,Qt.AlignCenter|Qt.AlignLeft)
        #layout_time.addStretch()
        layout_datetime.addLayout(layout_date,0)
        layout_datetime.addLayout(layout_time,1)
        
        layout_checkbox.addWidget(self.photo_checkbox,1,Qt.AlignCenter)
        layout_checkbox.addWidget(self.sound_checkbox,1,Qt.AlignCenter)
        #layout_function.addStretch()
        #layout_function.addLayout(layout_datetime)
        #layout_function.addStretch()
        layout_info.addLayout(layout_datetime)
        layout_info.addLayout(layout_name)
        layout_info.addLayout(layout_emotion)
        layout_info.addLayout(layout_temp)
        
        layout_info.addLayout(layout_weight)
        layout_info.addLayout(layout_height)
        layout_info.addLayout(layout_ppg)
        layout_info.addLayout(layout_cal)
        
        #layout_setting.addLayout(layout_checkbox)
        #layout_function.addStretch()
        layout_setting.addLayout(layout_degree)
        #layout2.addLayout(layout_function)
        
        #layout2.addLayout(flo,1)
        #layout3.addLayout(flo)
        
        
        #layout3.addWidget(self.confirm_button)
        layout_calibrate.addWidget(self.downdown_button)
        layout_calibrate.addWidget(self.down_button)
        
        layout_calibrate.addWidget(self.label_calibrate)
        
        layout_calibrate.addWidget(self.up_button)
        layout_calibrate.addWidget(self.upup_button)
        
        #layout3.addWidget(self.up_button)
        #layout3.addWidget(self.down_button)
        layout3.addLayout(layout_calibrate)
        layout3.addWidget(self.photo_switch_button)
        layout3.addWidget(self.sound_switch_button)
        #layout3.addWidget(self.setting_button)
        layout3.addWidget(self.about_button)
        layout3.addWidget(self.cancel_button)
        #layout2.addWidget(self.confirm_button,0,Qt.AlignRight)
        
        #layout2.addStretch()
        #layout2.addLayout(layout3)
        #layout2.addStretch()
        
        layout_setting.addLayout(layout3)
        layout_function.addStretch()
        if right_logo_display == 1:
            #layout_function.addWidget(self.label_logo2,0,Qt.AlignCenter)
            layout_function.addLayout(layout_logo)
        layout_function.addStretch()    
        layout_function.addWidget(self.groupbox_1)
        layout_function.addWidget(self.groupbox_2)
        
        layout_function.addStretch()
        #layout1.addLayout(layout)
        
        self.groupbox_1.setLayout(layout_info)
        self.groupbox_2.setLayout(layout_setting)
        
        #layout3.addLayout(layout1)
        #layout3.addWidget(self.message_widget)
        #mainlayout.addLayout(layout3)
        #mainlayout.addStretch()

        if left_logo_display == 1:
            mainlayout.addLayout(layout_logo)
        mainlayout.addStretch()
        mainlayout.addLayout(layout)
        #mainlayout.addStretch()
        mainlayout.addStretch()
        #mainlayout.addLayout(layout2)
        mainlayout.addLayout(layout_function)
        
        mainlayout.addStretch()
        self.confirm_button.clicked.connect(self.on_click)
        self.about_button.clicked.connect(self.About_Box)
        #self.setting_button.clicked.connect(self.Setting_Box)
        self.cancel_button.clicked.connect(self.Close_App)
        self.sound_switch_button.clicked.connect(self.sound_check_button)
        self.photo_switch_button.clicked.connect(self.photo_check_button)
        self.up_button.clicked.connect(self.up_temp_button)
        self.down_button.clicked.connect(self.down_temp_button)
        self.upup_button.clicked.connect(self.upup_temp_button)
        self.downdown_button.clicked.connect(self.downdown_temp_button)
        self.thermal_video.start_recording()
        self.rgb_video.start_recording()
        self.image_show.start_recording()
        #self.photo_checkbox.stateChanged.connect(self.photo_check)
        #self.sound_checkbox.stateChanged.connect(self.sound_check)
        #self.record_message.start_recording()
        self.timer()
        #self.openSlot()
        
        
        self.setLayout(mainlayout)
        
    '''    
    def showdialog(self):
       self.d = QDialog()
       self.logo1_img = QPixmap("IV_Logo.jpg")
       self.label_logo1 = QtWidgets.QLabel("")
       self.label_logo1.setPixmap(self.logo1_img)
       self.company = QtWidgets.QLabel('Company: Insight Vision Ltd.')
       self.Designer = QtWidgets.QLabel('Designer: ???')
       self.Contact = QtWidgets.QLabel('Tel:')
       
       
       self.b1 = QtWidgets.QPushButton("ok",self.d)
       
       self.grid = QtWidgets.QGridLayout()
       
       self.grid.addWidget(self.label_logo1,0,0)
       self.grid.addWidget(self.company,1,0)
       self.grid.addWidget(self.Designer,2,0)
       self.grid.addWidget(self.Contact,3,0)
       self.grid.addWidget(self.b1,4,0)
       
       self.setLayout(self.grid)   
       
       self.d.setWindowTitle("About")
       self.d.setWindowModality(Qt.ApplicationModal)
       
       self.d.exec_()
        
    '''
    def Close_App(self):
        sys.exit()
        
    def About_Box(self):
        self.logo_img = QPixmap("res/IV_Logo_with_chinese.jpg")
          
        
        messagebox = QtWidgets.QMessageBox()
        messagebox.setText("Company: \n Insight Vision Co., Ltd.")
        #messagebox.setInformativeText("Designer: ??? \nTel: (xx)xxxx-xxx\nEmail: service@insghtvision.com.tw")
        messagebox.setInformativeText("Contact: \nFor Service \nservice@insghtvision.com.tw \nFor Sales \nSales@insghtvision.com.tw")
        messagebox.setWindowTitle("About")
        messagebox.setDetailedText('Model: '+model+' \nVersion: '+ver)
        messagebox.setIconPixmap(self.logo_img)
        messagebox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        exe = messagebox.exec_()
        #box.warning(self,"提示","這是一個按鈕事件")
    '''
    def Setting_Box(self):
        
        photo_checkbox = QCheckBox('拍照模式')
        sound_checkbox = QCheckBox('音效模式')
        
        messagebox = QtWidgets.QMessageBox()
        messagebox.setText("勾選您要開啟的功能")
        
        messagebox.setWindowTitle("Setting")
        Save = messagebox.addButton('保存', QMessageBox.AcceptRole)
        NoSave = messagebox.addButton('取消', QMessageBox.RejectRole)
        Cancel = messagebox.addButton('不保存', QMessageBox.DestructiveRole)
        
        messagebox.setCheckBox(photo_checkbox)
        messagebox.setCheckBox(sound_checkbox)
        photo_checkbox.stateChanged.connect(self.photo_check)
        sound_checkbox.stateChanged.connect(self.sound_check)
        messagebox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        reply = messagebox.exec_()
    '''
    def photo_check(self):
        global photo_switch
        if self.sender().isChecked():
            photo_switch = 'on'
            cfg['param']['photo_switch'] = 'on'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
        else:
            photo_switch = 'off'
            cfg['param']['photo_switch'] = 'off'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
            
    def sound_check(self):
        global sound_switch
        if self.sender().isChecked():
            sound_switch = 'on'
            cfg['param']['sound_switch'] = 'on'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
        else:
            sound_switch = 'off'
            cfg['param']['sound_switch'] = 'off'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
    
        
    def photo_check_button(self):
        global photo_switch
        if photo_switch == 'off':
            photo_switch = 'on'
            self.photo_switch_button.setIcon(QtGui.QIcon('res/camera.png'))
            #self.photo_switch_button.setIconSize(QtCore.QSize(50*rate,50*rate))
            cfg['param']['photo_switch'] = 'on'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
        else:
            self.photo_switch_button.setIcon(QtGui.QIcon('res/camera-off.png'))
            #self.photo_switch_button.setIconSize(QtCore.QSize(50*rate,50*rate))
            photo_switch = 'off'
            cfg['param']['photo_switch'] = 'off'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
            
    def sound_check_button(self):
        global sound_switch
        if sound_switch == 'off':
            sound_switch = 'on'
            self.sound_switch_button.setIcon(QtGui.QIcon('res/notification.png'))
            #self.sound_switch_button.setIconSize(QtCore.QSize(50*rate,50*rate))
            cfg['param']['sound_switch'] = 'on'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
        else:
            sound_switch = 'off'
            self.sound_switch_button.setIcon(QtGui.QIcon('res/disable-alarm.png'))
            #self.sound_switch_button.setIconSize(QtCore.QSize(50*rate,50*rate))
            cfg['param']['sound_switch'] = 'off'
            with open('config.ini', 'w', encoding='utf-8') as f:
                cfg.write(f)
    
    def timer(self):
        self.timer_id = self.startTimer(1000, timerType = QtCore.Qt.VeryCoarseTimer)

    
    def up_temp_button(self):
        global degree_eq_2
        degree_eq_2 +=0.1
        self.label_calibrate.setText("{0:.1f}".format(degree_eq_2))
        cfg['param']['degree_eq_2'] = str(degree_eq_2)
        with open('config.ini', 'w', encoding='utf-8') as f:
            cfg.write(f)
    def down_temp_button(self):
        global degree_eq_2
        degree_eq_2 -=0.1
        self.label_calibrate.setText("{0:.1f}".format(degree_eq_2))
        cfg['param']['degree_eq_2'] = str(degree_eq_2)
        with open('config.ini', 'w', encoding='utf-8') as f:
            cfg.write(f)
    def upup_temp_button(self):
        global degree_eq_2
        degree_eq_2 +=0.5
        self.label_calibrate.setText("{0:.1f}".format(degree_eq_2))
        cfg['param']['degree_eq_2'] = str(degree_eq_2)
        with open('config.ini', 'w', encoding='utf-8') as f:
            cfg.write(f)
    def downdown_temp_button(self):
        global degree_eq_2
        degree_eq_2 -=0.5
        self.label_calibrate.setText("{0:.1f}".format(degree_eq_2))
        cfg['param']['degree_eq_2'] = str(degree_eq_2)
        with open('config.ini', 'w', encoding='utf-8') as f:
            cfg.write(f)
    
    def timerEvent(self, event):
        global name,emotion,temptemp
        self.label_date.setText(time.strftime("%Y/%m/%d"))
        #self.label_date.setPixmap(QtGui.QPixmap('date.jpg'))
        
        self.label_time.setText(time.strftime("%H:%M:%S"))
        #self.label_emotion_data.setText(emotion)
        #self.label_time.setPixmap(QtGui.QPixmap('time.jpg'))
        #self.label_date.setAlignment(Qt.AlignCenter|QtCore.Qt.AlignLeft)
        #self.label_time.setAlignment(Qt.AlignCenter|QtCore.Qt.AlignLeft)
        userid = ''
        if name == 'Ben':
            userid ='0476166658'
        if  name!='Unknown':
            #db.bodyweight(userid)
            #db.hbrtbl(userid)
            #print (str(db.bodyweight(userid)))
            #self.label_weight_data.setText(str(db.bodyweight(userid)))
            #self.label_blood_data.setText(str(db.hbrtbl(userid)))
            self.label_member_name_data.setText(name)
            self.label_emotion_data.setText(emotion)
            self.label_temp_data.setText("{0:.1f} degC".format(temptemp))
        elif name=='Unknown':
            #self.label_weight_data.setText('No data')
            #self.label_blood_data.setText('No data')
            self.label_member_name_data.setText("非會員")
            self.label_emotion_data.setText(emotion)
            self.label_temp_data.setText("{0:.1f} degC".format(temptemp))
        else:
            #self.label_weight_data.setText('')
            #self.label_blood_data.setText('')
            self.label_member_name_data.setText('')
            self.label_emotion_data.setText('')
            self.label_temp_data.setText('')
        
    def on_click(self):
        global high
        tempera = self.textbox.text()
        #print (tempera)
        if not tempera :
            high =38.0
        else:
            high = float(tempera)        
        
        self.textbox.setText(str(high))
    
    
        
    def openSlot(self):
        # 調用打開文件diglog
        #fileName, tmp = QFileDialog.getOpenFileName(
            #self, 'Open Image', './__data', '*.png *.jpg *.bmp')

        #if fileName is '':
            #return

        # 採用opencv函數讀取數據
        self.img = cv2.imread("res/IV_Logo_with_chinese.jpg", -1)
        self.img = cv2.resize( self.img, (0,0), fx=1, fy=1)
        if self.img.size == 1:
            return

        self.refreshShow1()

    def refreshShow1(self):
        # 提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, colors = self.img.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage
        QPixmap = QtGui.QPixmap

        self.qImg = QImage(self.img,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888).rgbSwapped()

        
        

        # 將Qimage顯示出來
        self.label_logo1.setPixmap(QPixmap.fromImage(self.qImg))
        
    # SN     
    def getText(self):
        sn = cfg['serial_number']['SN']
        if sn !='':
            check = key.AES_Decrypt(sn)
            if check is not True:
                self.Question_Box()
                #sys.exit()
        #global filename,number
        else:
            self.inputSN()
        
    def inputSN(self):
            text, okPressed = QInputDialog.getText(self, "Serial Number","SN:", QLineEdit.Normal, "Input SN")
            if okPressed and text != '' and text !='Input SN':
                #print(text)
                
                check = key.AES_Decrypt(text)
                if check is not True:
                    sys.exit()
                else:
                    cfg['serial_number']['SN'] = str(text)
                    with open('config.ini', 'w', encoding='utf-8') as f:
                        cfg.write(f)
            else:
                sys.exit()
                
    def Error_Message(self, message_str):
        messagebox = QtWidgets.QMessageBox()
        messagebox.setWindowTitle("Error")
        messagebox.setIcon(QMessageBox.Warning)
        messagebox.setText(message_str)
        messagebox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        exe = messagebox.exec_()
        
            
    def Question_Box(self):
        reply = QtWidgets.QMessageBox.question(self,'詢問','Serial Number 驗證錯誤 是否重新輸入?', QMessageBox.Yes | QMessageBox.No , QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.inputSN()
        elif reply == QMessageBox.No:
            self.Error_Message('Serial Number 驗證失敗 程式關閉!')
            sys.exit()
        else:
            self.Error_Message('Serial Number 驗證失敗 程式關閉!')
            sys.exit()
def main():

    
    app = QtWidgets.QApplication(sys.argv)
    screen = QDesktopWidget().screenGeometry()
    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget()
    main_window.setCentralWidget(main_widget)
    main_window.setGeometry(0, 0, 400, 400)
    main_window.setWindowIcon(QtGui.QIcon("res/IV_Logo.jpg"))
    main_window.setWindowTitle('IV-TS001')
    
    main_window.resize(screen.width(),screen.height())
    if en_mode != 1:
        main_window.showFullScreen()
    
    main_window.show()
    sys.exit(app.exec_())
    
    
        
    




if __name__ == '__main__':

    #try:
        
    main()
    #print (os.path.dirname(__file__))
    #except:
    #    print ("Can't get Cam")
    th.stop()
    cv2.destroyAllWindows()        

