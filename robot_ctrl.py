import time
import os

import cv2
import imageio
import numpy as np
from collections import deque
import math

from base_camera import BaseCamera
from base_ctrl import BaseController
import datetime
from fractions import Fraction

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Encoder
from picamera2.outputs import FfmpegOutput

import threading
import imutils
import yaml
import json
import textwrap

import mediapipe as mp

import audio_ctrl


curpath = os.path.realpath(__file__)
thisPath = os.path.dirname(curpath)
with open(thisPath + '/config.yaml', 'r') as yaml_file:
    f = yaml.safe_load(yaml_file)

usb_camera = False

def is_raspberry_pi5():
    with open('/proc/cpuinfo', 'r') as file:
        for line in file:
            if 'Model' in line:
                if 'Raspberry Pi 5' in line:
                    return True
                else:
                    return False

if is_raspberry_pi5():
    base = BaseController('/dev/ttyAMA0', 115200)
    # usb_camera = True
else:
    base = BaseController('/dev/serial0', 115200)

base.lights_ctrl(0, 0)

faceCascade = cv2.CascadeClassifier(thisPath + '/haarcascade_frontalface_default.xml')

default_resolutions = {
    "1944P": (2592, 1944),
    "1080P": (1920, 1080),
     "960P": (1280, 960),
     "480P": (640, 480),
     "240P": (320, 240)
}

resolution_to_set = default_resolutions["480P"]

set_new_resolution_flag = False
video_path = thisPath + '/videos/'
photo_path = thisPath + '/static/'
photo_filename = '/static/'

video_record_fps = 30
video_filename = '/videos/video.mp4'
set_video_record_flag = False
get_frame_flag = False
frame_scale = 1.0

cv_mode = f['code']['cv_none']
feedback_interval = f['sbc_config']['feedback_interval']

led_mode_fb = 0 # 0:off 1:auto 2:on
base_light_fb = 0
detection_reaction_flag = f['code']['re_none']
frame_rate = 0
base_light_pwm = 0
head_light_pwm = 0
cv_movtion_lock_flag = True
gimbal_x = 0
gimbal_y = 0
mission_flag = True

base_info_feedback_json = {'T':0,'pan':0,'tilt':0,'v':0}

color_list = {
    'red':  [np.array([  0,200, 70]), np.array([ 10, 255, 190])],
    'green':[np.array([ 50, 90, 80]), np.array([ 72, 255, 240])],
    'blue': [np.array([ 90,120, 90]), np.array([120, 255, 220])]
}
if f['cv']['default_color'] in color_list:
    color_lower = color_list[f['cv']['default_color']][0]
    color_upper = color_list[f['cv']['default_color']][1]
else:
    color_lower = np.array(f['cv']['color_lower'])
    color_upper = np.array(f['cv']['color_upper'])

line_lower = np.array([25, 150, 70])
line_upper = np.array([42, 255, 255])

# caffe model
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(thisPath + '/deploy.prototxt', thisPath + '/mobilenet_iter_73000.caffemodel')

track_color_iterate = f['cv']['track_color_iterate']
track_faces_iterate = f['cv']['track_faces_iterate']
track_spd_rate = f['cv']['track_spd_rate']
track_acc_rate = f['cv']['track_acc_rate']


info_deque = deque(maxlen=10)
info_scale_x = 320 / 480
info_scale_y = 370 / 480
info_scale = 270 / 480
info_bg_color = (0, 0, 0)
show_info_flag = False
info_update_time = time.time()


recv_deque = deque(maxlen=20)
recv_line_max = 26
show_recv_flag = False


# MediaPipe
mpDraw = mp.solutions.drawing_utils


# MediaPipe Hand GS
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
max_distance = 1
gs_pic_interval = 6
gs_pic_last_time = time.time()


# MediaPipe Faces
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=1, 
                    smooth_landmarks=True, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)


# findline autodrive
sampling_line_1 = 0.6
sampling_line_2 = 0.9
slope_impact = 1.5
base_impact = 0.005
speed_impact = 0.5
line_track_speed = 0.3
slope_on_speed = 0.1


def format_json_numbers(obj):
    if isinstance(obj, dict):
        return {k: format_json_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_json_numbers(elem) for elem in obj]
    elif isinstance(obj, float):
        return round(obj, 2)
    return obj


def get_feedback():
    global base_info_feedback_json, info_update_time, show_info_flag
    while True:
        try:
            # data_recv_buffer = base.on_data_received()
            data_recv_buffer = json.loads(base.rl.readline().decode('utf-8'))
            if 'T' in data_recv_buffer:
                if data_recv_buffer['T'] == 1001:
                    base_info_feedback_json = data_recv_buffer
                elif data_recv_buffer['T'] == 1003:
                    info_update_time = time.time()
                    info_deque.appendleft({'text':json.dumps(data_recv_buffer['mac']),'color':(16,64,255),'size':0.5}) # mac
                    wrapped_lines = textwrap.wrap(json.dumps(data_recv_buffer['megs']), recv_line_max) # megs
                    for line in wrapped_lines:
                        info_deque.appendleft({'text':line,'color':(255,255,255),'size':0.5})
                    show_info_flag = True

            if show_recv_flag:
                if data_recv_buffer['T'] == 1001:
                    recv_deque.appendleft(json.dumps(format_json_numbers(data_recv_buffer)))
                else:
                    recv_deque.appendleft(json.dumps(data_recv_buffer))
                print(data_recv_buffer)
        except Exception as e:
            print(f"An exception occurred: {e}")
            pass
        # time.sleep(feedback_interval)


class RobotCtrlMiddleWare:
    def __init__(self):
        self.base_ctrl_speed = 25

    def json_command_handler(self, input_json):
        base.base_json_ctrl(input_json)

    def base_oled(self, line_input, text_input):
        base.base_oled(line_input, text_input)

    def set_led_mode_off(self):
        global led_mode_fb, head_light_pwm
        led_mode_fb = 0
        head_light_pwm = 0
        base.lights_ctrl(base_light_pwm, head_light_pwm)

    def set_led_mode_auto(self):
        global led_mode_fb, head_light_pwm
        led_mode_fb = 1
        head_light_pwm = 0
        base.lights_ctrl(base_light_pwm, head_light_pwm)

    def set_led_mode_on(self):
        global led_mode_fb, head_light_pwm
        led_mode_fb = 2
        head_light_pwm = 255
        base.lights_ctrl(base_light_pwm, head_light_pwm)

    def set_base_led_on(self):
        global base_light_pwm, base_light_fb
        base_light_pwm = 255
        base_light_fb = 1
        base.lights_ctrl(base_light_pwm, head_light_pwm)

    def set_base_led_off(self):
        global base_light_pwm, base_light_fb
        base_light_pwm = 0
        base_light_fb = 0
        base.lights_ctrl(base_light_pwm, head_light_pwm)

    def base_led_ctrl(self):
        global base_light_pwm, base_light_fb
        if base_light_fb:
            base_light_fb = 0
            base_light_pwm = 0
        else:
            base_light_fb = 1
            base_light_pwm = 255
        base.lights_ctrl(base_light_pwm, head_light_pwm)

    def head_led_ctrl(self):
        global head_light_pwm, led_mode_fb
        if head_light_pwm != 0:
            head_light_pwm = 0
            led_mode_fb = 0
        else:
            head_light_pwm = 255
            led_mode_fb = 2
        base.lights_ctrl(base_light_pwm, head_light_pwm)

    def play_random_audio(self, input_dirname, force_flag):
        audio_ctrl.play_random_audio(input_dirname, force_flag)

    def audio_play(self, audio_file):
        audio_ctrl.play_audio_thread(audio_file)

    def audio_stop(self):
        audio_ctrl.stop()


class Camera(BaseCamera):
    video_source = 0
    # frame_scale = 1
    video_record_status_flag = False
    force_record_stop_flag = False

    # opencv values
    overlay = None
    last_frame_capture_time = datetime.datetime.now()
    led_status = False
    avg = None
    last_movtion_captured = datetime.datetime.now()

    points = deque(maxlen=32)
    min_radius = f['cv']['min_radius']
    sampling_rad = f['cv']['sampling_rad']
    CMD_GIMBAL = f['cmd_config']['cmd_gimbal_ctrl']
    aimed_error = f['cv']['aimed_error']

    cv_event = threading.Event()
    cv_event.clear()

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        global photo_filename, video_filename, frame_rate, get_frame_flag, show_info_flag

        if not usb_camera:
            encoder = H264Encoder(1000000)
            picam2 = Picamera2()
            picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (resolution_to_set[0], resolution_to_set[1])}))
            picam2.start()
        else:
            camera = cv2.VideoCapture(-1)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_to_set[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_to_set[1])
            if not camera.isOpened():
                raise RuntimeError('Could not start camera.')

        feedback_thread = threading.Thread(target=get_feedback, daemon=True)
        feedback_thread.start()

        base.gimbal_base_ctrl(2, 2, 0)
        fps_start_time = time.time()
        fps_frame_count = 0

        while True:
            if usb_camera:
                _, img_cap = camera.read()
                img_cap = cv2.cvtColor(img_cap, cv2.COLOR_BGR2BGRA)
            else:
                img_cap = picam2.capture_array()

            # photo corp
            if frame_scale == 1:
                if show_recv_flag:
                    for i in range(0, len(recv_deque)):
                        cv2.putText(img_cap, recv_deque[i], 
                                (round(0.05*resolution_to_set[0]), round(0.1*resolution_to_set[0] + i * 13 * resolution_to_set[1] / 480)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.369 * resolution_to_set[1] / 480, (255, 255, 255), 1)
            elif frame_scale > 1.0:
                img_height, img_width = img_cap.shape[:2]
                img_width_d2  = img_width/2
                img_height_d2 = img_height/2
                x_start = int(img_width_d2 - (img_width_d2//frame_scale))
                x_end   = int(img_width_d2 + (img_width_d2//frame_scale))
                y_start = int(img_height_d2 - (img_height_d2//frame_scale))
                y_end   = int(img_height_d2 + (img_height_d2//frame_scale))
                img_cap = img_cap[y_start:y_end, x_start:x_end]

            # opencv & render
            if cv_mode != f['code']['cv_none']:
                if not Camera.cv_event.is_set():
                    Camera.cv_event.set()
                    Camera.opencv_threading(Camera, img_cap)
                try:
                    mask = Camera.overlay.astype(bool)
                    img_cap[mask] = Camera.overlay[mask]
                    cv2.addWeighted(Camera.overlay, 1, img_cap, 1, 0, img_cap)
                except Exception as e:
                    print("An error occurred:", e)
            elif show_info_flag:
                # render info
                if time.time() - info_update_time > 10:
                    show_info_flag = False
                overlay = img_cap.copy()
                cv2.rectangle(overlay,  (round((info_scale-0.005)*resolution_to_set[0]), round((0.33)*resolution_to_set[1])), 
                                        (round(0.98*resolution_to_set[0]), round((0.78)*resolution_to_set[1])), 
                                        info_bg_color, -1)
                cv2.addWeighted(overlay, 0.5, img_cap, 0.5, 0, img_cap)

                # info_deque.appendleft(time.time())
                for i in range(0, len(info_deque)):
                    cv2.putText(img_cap, info_deque[i]['text'], 
                                (round(info_scale*resolution_to_set[0]), round(info_scale*resolution_to_set[0] - i * 20 * resolution_to_set[1] / 480)), 
                                cv2.FONT_HERSHEY_SIMPLEX, info_deque[i]['size'] * resolution_to_set[1] / 480, info_deque[i]['color'], 1)
            
            # photo capture
            if get_frame_flag:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                photo_filename = f'{photo_path}photo_{current_time}.jpg'
                try:
                    cv2.imwrite(photo_filename, img_cap)
                    get_frame_flag = False
                    print(photo_filename)
                except:
                    pass
            
            # video capture
            if not usb_camera:
                if not set_video_record_flag and not Camera.video_record_status_flag:
                    pass
                elif set_video_record_flag and not Camera.video_record_status_flag:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    video_filename = f'{video_path}video_{current_time}.mp4'
                    picam2.start_recording(encoder, FfmpegOutput(video_filename))
                    Camera.video_record_status_flag = True
                elif not set_video_record_flag and Camera.video_record_status_flag:
                    picam2.stop_recording()
                    picam2.start()
                    Camera.video_record_status_flag = False
            elif usb_camera:
                if not set_video_record_flag and not Camera.video_record_status_flag:
                    pass
                elif set_video_record_flag and not Camera.video_record_status_flag:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    video_filename = f'{video_path}video_{current_time}.mp4'
                    writer = imageio.get_writer(video_filename, fps=30)
                    Camera.video_record_status_flag = True
                elif set_video_record_flag and Camera.video_record_status_flag:
                    writer.append_data(np.array(cv2.cvtColor(img_cap, cv2.COLOR_BGRA2RGB)))
                elif not set_video_record_flag and Camera.video_record_status_flag:
                    Camera.video_record_status_flag = False
                    writer.close()

            # render record red dot
            if Camera.video_record_status_flag:
                cv2.circle(img_cap, (15, 15), 5, (64, 64, 255), -1)

            yield cv2.imencode('.jpg', img_cap)[1].tobytes()
            
            # get fps
            fps_frame_count += 1
            fps_current_time = time.time()
            fps_elapsed_time = fps_current_time - fps_start_time
            if fps_elapsed_time >= 2.0:
                frame_rate = fps_frame_count / fps_elapsed_time
                fps_frame_count = 0
                fps_start_time = fps_current_time


    def set_video_resolution(self, input_resolution):
        global set_new_resolution_flag, resolution_to_set
        if self.video_record_status_flag:
            return
        if input_resolution in ["1944P", "1080P", "960P", "480P", "240P"]:
            resolution_to_set = default_resolutions[input_resolution]
            set_new_resolution_flag = True
            return True
        return False


    def capture_frame(self, input_path):
        global photo_path, get_frame_flag
        photo_path = input_path
        get_frame_flag = True


    def record_video(self, input_cmd, input_path):
        global video_path, set_video_record_flag
        video_path = input_path
        if input_cmd == 1:
            set_video_record_flag = True
        else:
            set_video_record_flag = False


    def scale_frame(self, input_scale_rate):
        global frame_scale
        if input_scale_rate >= 1.0:
            frame_scale = input_scale_rate


    def set_cv_mode(self, input_mode):
        global cv_mode, set_video_record_flag
        cv_mode = input_mode
        if cv_mode == f['code']['cv_none']:
            set_video_record_flag = False
            base.lights_ctrl(base_light_pwm, 0)


    def set_detection_reaction(self, input_reaction):
        global detection_reaction_flag, set_video_record_flag
        detection_reaction_flag = input_reaction
        if detection_reaction_flag == f['code']['re_none']:
            set_video_record_flag = False


    def get_status(self):
        try:
            feedback_json = {
            f['fb']['led_mode']:    led_mode_fb,
            f['fb']['detect_type']: cv_mode,
            f['fb']['detect_react']:detection_reaction_flag,
            f['fb']['pan_angle']:   base_info_feedback_json['pan'],
            f['fb']['tilt_angle']:  base_info_feedback_json['tilt'],
            f['fb']['base_voltage']:base_info_feedback_json['v'],
            f['fb']['video_fps']:   frame_rate,
            f['fb']['cv_movtion_mode']: cv_movtion_lock_flag,
            f['fb']['base_light']:  base_light_fb
            }
            return feedback_json
        except:
            pass


    def set_pan_id(self):
        base.bus_servo_id_set(254, 2)


    def release_torque(self):
        base.bus_servo_torque_lock(254, 0)


    def middle_set(self):
        base.bus_servo_mid_set(254)


    def set_tilt_id(self):
        base.bus_servo_id_set(254, 1)


    def set_movtion_lock(self, cmd):
        global cv_movtion_lock_flag, gimbal_x, gimbal_y
        if cmd == f['code']['mc_lock']:
            cv_movtion_lock_flag = True
            gimbal_x = 0
            gimbal_y = 0
        elif cmd == f['code']['mc_unlo']:
            cv_movtion_lock_flag = False


    def gimbal_track(self, fx, fy, gx, gy, iterate):
        global gimbal_x, gimbal_y
        distance = math.sqrt((fx - gx) ** 2 + (gy - fy) ** 2)
        gimbal_x += (gx - fx) * iterate
        gimbal_y += (fy - gy) * iterate
        if gimbal_x > 180:
            gimbal_x = 180
        elif gimbal_x < -180:
            gimbal_x = -180
        if gimbal_y > 90:
            gimbal_y = 90
        elif gimbal_y < -30:
            gimbal_y = -30
        gimbal_spd = int(distance * track_spd_rate)
        gimbal_acc = int(distance * track_acc_rate)
        if gimbal_acc < 1:
            gimbal_acc = 1
        if gimbal_spd < 1:
            gimbal_spd = 1
        base.base_json_ctrl({"T":self.CMD_GIMBAL,"X":gimbal_x,"Y":gimbal_y,"SPD":gimbal_spd,"ACC":gimbal_acc})
        return distance


    def cv_detect_movition(self, img):
        global set_video_record_flag, get_frame_flag
        timestamp = datetime.datetime.now()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg is None:
            self.avg = gray.copy().astype("float")
            return
        try:
            cv2.accumulateWeighted(gray, self.avg, 0.5)
        except:
            return
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        overlay_buffer = np.zeros_like(img)
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 2000:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (mov_x, mov_y, mov_w, mov_h) = cv2.boundingRect(c)
            cv2.rectangle(overlay_buffer, (mov_x, mov_y), (mov_x + mov_w, mov_y + mov_h), (128, 255, 0), 1)
            self.last_movtion_captured = timestamp

            if(timestamp - self.last_frame_capture_time).seconds >= 1:
                if detection_reaction_flag == f['code']['re_none']:
                    pass
                elif detection_reaction_flag == f['code']['re_capt']: 
                    get_frame_flag = True
                elif detection_reaction_flag == f['code']['re_reco']:
                    set_video_record_flag = True
                self.last_frame_capture_time = datetime.datetime.now()
            
        if (timestamp - self.last_movtion_captured).seconds >= 1.5:
            if detection_reaction_flag == f['code']['re_reco']:
                if(timestamp - self.last_frame_capture_time).seconds >= 5:
                    set_video_record_flag = False

        self.overlay = np.zeros_like(img)
        self.overlay = overlay_buffer


    def cv_detect_faces(self, img):
        global set_video_record_flag, get_frame_flag
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
                gray_img,     
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(20, 20)
            )
        overlay_buffer = np.zeros_like(img)

        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2

        max_area = 0
        max_face_center = (0, 0)

        if len(faces):
            if led_mode_fb == 1:
                if not self.led_status:
                    base.lights_ctrl(base_light_pwm, 255)
                    self.led_status = True

            for (x,y,w,h) in faces:
                cv2.rectangle(overlay_buffer,(x,y),(x+w,y+h),(64,128,255),1)
                face_area = w * h
                if face_area > max_area:
                    max_area = face_area
                    max_face_center = (x + w // 2, y + h // 2)

            if not cv_movtion_lock_flag:
                self.gimbal_track(self, center_x, center_y, max_face_center[0], max_face_center[1], track_faces_iterate)

            if(datetime.datetime.now() - self.last_frame_capture_time).seconds >= 3:
                if detection_reaction_flag == f['code']['re_none']:
                    pass
                elif detection_reaction_flag == f['code']['re_capt']:
                    get_frame_flag = True
                elif detection_reaction_flag == f['code']['re_reco']:
                    set_video_record_flag = True
                self.last_frame_capture_time = datetime.datetime.now()
        else:
            if led_mode_fb == 1:
                if self.led_status:
                    base.lights_ctrl(base_light_pwm, 0)
                    self.led_status = False

            if detection_reaction_flag == f['code']['re_reco']:
                if(datetime.datetime.now() - self.last_frame_capture_time).seconds >= 5:
                    set_video_record_flag = False

        cv2.putText(overlay_buffer, 'NUMBER: {}'.format(len(faces)), (center_x+50, center_y+40), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, 'ITERATE: {}'.format(track_faces_iterate), (center_x+50, center_y+60), 
                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' SPD_R: {}'.format(track_spd_rate), (center_x+50, center_y+80), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' ACC_R: {}'.format(track_acc_rate), (center_x+50, center_y+100), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.overlay = np.zeros_like(img)
        self.overlay = overlay_buffer


    def cv_detect_objects(self, img):
        overlay_buffer = np.zeros_like(img)
        cv2.putText(overlay_buffer, 'CV_OBJS', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(class_names[idx], confidence * 100)
                cv2.rectangle(overlay_buffer, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(overlay_buffer, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.overlay = np.zeros_like(img)
        self.overlay = overlay_buffer


    def cv_detect_color(self, img):
        global head_light_pwm
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, None, iterations=5)
        mask = cv2.dilate(mask, None, iterations=5)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        overlay_buffer = np.zeros_like(img)

        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), self.sampling_rad, (255), thickness=-1)

        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        masked_hsv_pixels = masked_hsv[mask == 255]
        lower_hsv = np.min(masked_hsv_pixels, axis=0)
        upper_hsv = np.max(masked_hsv_pixels, axis=0)

        cv2.putText(overlay_buffer, ' UPPER: {}'.format(upper_hsv), (center_x+50, center_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' LOWER: {}'.format(lower_hsv), (center_x+50, center_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(overlay_buffer, ' UPPER: {}'.format(color_upper), (center_x+50, center_y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(overlay_buffer, ' LOWER: {}'.format(color_lower), (center_x+50, center_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(overlay_buffer, 'ITERATE: {}'.format(track_color_iterate), (center_x+50, center_y+140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' SPD_R: {}'.format(track_spd_rate), (center_x+50, center_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' ACC_R: {}'.format(track_acc_rate), (center_x+50, center_y+180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(overlay_buffer, (center_x, center_y), self.sampling_rad, (64, 255, 64), 1)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > self.min_radius:
                if not cv_movtion_lock_flag:
                    distance = self.gimbal_track(self, center_x, center_y, center[0], center[1], track_color_iterate)
                    if distance < self.aimed_error:
                        head_light_pwm = 3
                        base.lights_ctrl(base_light_pwm, head_light_pwm)
                    else:
                        head_light_pwm = 0
                        base.lights_ctrl(base_light_pwm, head_light_pwm)
                    cv2.putText(overlay_buffer, 'DIF: {}'.format(distance), (center_x+50, center_y+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(overlay_buffer, (int(x), int(y)), int(radius),
                    (128, 255, 255), 1)
                cv2.circle(overlay_buffer, center, 3, (128, 255, 255), -1)
                cv2.line(overlay_buffer, center, (center_x, center_y), (0, 0, 255), 1)
                cv2.putText(overlay_buffer, 'RAD: {}'.format(radius), (center_x+50, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                self.points.appendleft(center)
            else:
                head_light_pwm = 0
                base.lights_ctrl(base_light_pwm, head_light_pwm)
                self.points.appendleft(None)

            for i in range(1, len(self.points)):
                if self.points[i-1] is None or self.points[i] is None:
                    continue
                cv2.line(overlay_buffer, self.points[i - 1], self.points[i], (255, 255, 128), 1)

        self.overlay = np.zeros_like(img)
        self.overlay = overlay_buffer


    def calculate_angle(self, A1, A2, B1, B2):
        vector_A = (A2.x - A1.x, A2.y - A1.y)
        vector_B = (B2.x - B1.x, B2.y - B1.y)

        dot_product = vector_A[0] * vector_B[0] + vector_A[1] * vector_B[1]

        magnitude_A = math.sqrt(vector_A[0]**2 + vector_A[1]**2)
        magnitude_B = math.sqrt(vector_B[0]**2 + vector_B[1]**2)

        angle = math.acos(dot_product / (magnitude_A * magnitude_B))

        angle_deg = math.degrees(angle)

        return angle_deg


    def calculate_distance(self, lm1, lm2):
        return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5


    def map_value(self, value, original_min, original_max, new_min, new_max):
        if original_max == 0:
            return 0
        return (value - original_min) / (original_max - original_min) * (new_max - new_min) + new_min


    def cv_detect_hand(self, img):
        global max_distance, gs_pic_last_time

        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        overlay_buffer = np.zeros_like(imgRGB)
        get_pwm = 0

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # draw joints
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = imgRGB.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(overlay_buffer, (cx, cy), 5, (255, 0, 0), -1)

                # draw lines
                mpDraw.draw_landmarks(overlay_buffer, handLms, mpHands.HAND_CONNECTIONS)

                target_pos = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                # print(f"x:{target_pos.x} y:{target_pos.y}")
                if not cv_movtion_lock_flag:
                    distance = self.gimbal_track(self, center_x, center_y, width*target_pos.x, height*target_pos.y, track_faces_iterate)

                # check hand gs
                pinky_finger_gs = self.calculate_angle(self,
                                            handLms.landmark[mpHands.HandLandmark.WRIST],
                                            handLms.landmark[mpHands.HandLandmark.PINKY_MCP],
                                            handLms.landmark[mpHands.HandLandmark.PINKY_MCP],
                                            handLms.landmark[mpHands.HandLandmark.PINKY_TIP])

                index_finger_gs = self.calculate_angle(self,
                                            handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP],
                                            handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP],
                                            handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP],
                                            handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP])

                middle_finger_gs = self.calculate_angle(self,
                                            handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP],
                                            handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP],
                                            handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP],
                                            handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP])

                # LED Ctrl
                if middle_finger_gs > 20 and pinky_finger_gs > 90:
                    cv2.putText(overlay_buffer, ' GS: LED Ctrl', (center_x+50, center_y+100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
                    tips_distance = self.calculate_distance(self, handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP],
                        handLms.landmark[mpHands.HandLandmark.THUMB_TIP])

                    if index_finger_gs < 3:
                        max_distance = tips_distance
                    # print(index_finger_gs)

                    get_pwm = int(self.map_value(self, tips_distance, 0.01, max_distance, 0, 128))
                    base.lights_ctrl(get_pwm, get_pwm)

                    try:
                        print(f"dis:{tips_distance} max:{max_distance} pwm:{get_pwm}")
                    except Exception as e:
                        print(e)

                # Take Pic
                elif middle_finger_gs < 10 and pinky_finger_gs > 90 and index_finger_gs < 10:
                    cv2.putText(overlay_buffer, ' GS: Take Pic', (center_x+50, center_y+100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
                    if time.time() - gs_pic_last_time > gs_pic_interval:
                        base.lights_ctrl(255, 255)
                        time.sleep(0.01)
                        self.capture_frame(self, thisPath + '/static/')
                        base.lights_ctrl(0, 0)
                        gs_pic_last_time = time.time()

                # Not Found
                else:
                    cv2.putText(overlay_buffer, ' GS: Not Defined', (center_x+50, center_y+100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
                    base.lights_ctrl(0, 0)

        cv2.putText(overlay_buffer, 'ITERATE: {}'.format(track_faces_iterate), (center_x+50, center_y+140), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' SPD_R: {}'.format(track_spd_rate), (center_x+50, center_y+160), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' ACC_R: {}'.format(track_acc_rate), (center_x+50, center_y+180), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        overlay_buffer = cv2.cvtColor(overlay_buffer, cv2.COLOR_RGB2BGR)
        self.overlay = cv2.cvtColor(overlay_buffer, cv2.COLOR_BGR2BGRA)


    def cv_auto_drive(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # get a sampling
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        mask_sampling = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask_sampling, (center_x, center_y), int(self.sampling_rad/4), (255), thickness=-1)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_sampling)
        masked_hsv_pixels = masked_hsv[mask_sampling == 255]
        lower_hsv = np.min(masked_hsv_pixels, axis=0)
        upper_hsv = np.max(masked_hsv_pixels, axis=0)

        # select the line color & get the mask
        # img = cv2.GaussianBlur(img, (11, 11), 0)
        line_mask = cv2.inRange(hsv, line_lower, line_upper)
        line_mask = cv2.erode(line_mask, None, iterations=2)
        line_mask = cv2.dilate(line_mask, None, iterations=2)

        sampling_h1 = int(height * sampling_line_1)
        sampling_h2 = int(height * sampling_line_2)

        get_sampling_1 = line_mask[sampling_h1]
        get_sampling_2 = line_mask[sampling_h2]

        sampling_width_1 = np.sum(get_sampling_1 == 255)
        sampling_width_2 = np.sum(get_sampling_2 == 255)

        if sampling_width_1:
            sam_1 = True
        else:
            sam_1 = False
        if sampling_width_2:
            sam_2 = True
        else:
            sam_2 = False

        line_index_1 = np.where(get_sampling_1 == 255)
        line_index_2 = np.where(get_sampling_2 == 255)

        if sam_1:
            sampling_1_left  = line_index_1[0][0]
            sampling_1_right = line_index_1[0][sampling_width_1 - 1]
            sampling_1_center= int((sampling_1_left + sampling_1_right) / 2)
        if sam_2:
            sampling_2_left  = line_index_2[0][0]
            sampling_2_right = line_index_2[0][sampling_width_2 - 1]
            sampling_2_center= int((sampling_2_left + sampling_2_right) / 2)

        line_slope = 0
        input_speed = 0
        input_turning = 0
        if sam_1 and sam_2:
            line_slope = (sampling_1_center - sampling_2_center) / abs(sampling_h1 - sampling_h2)
            impact_by_slope = slope_on_speed * abs(line_slope)
            # if impact_by_slope > input_speed:
            #     impact_by_slope = input_speed
            input_speed = line_track_speed - impact_by_slope
            # print(f'im_by_slope:{impact_by_slope}   input_speed:{input_speed}')
            input_turning = -(line_slope * slope_impact + (sampling_2_center - center_x) * base_impact) #+ (speed_impact * input_speed)
        elif not sam_1 and sam_2:
            input_speed = 0
            input_turning = (sampling_2_center - center_x) * base_impact
        elif sam_1 and not sam_2:
            input_speed = (line_track_speed / 3)
            input_turning = 0
        else:
            input_speed = - (line_track_speed / 3)
            input_turning = 0

        # input_turning = - line_slope * slope_impact
        # try:
        #     input_turning = -(sampling_2_center - center_x) * base_impact
        # except:
        #     pass
        if not cv_movtion_lock_flag:
            base.base_json_ctrl({"T":13,"X":input_speed,"Z":input_turning})

        overlay_buffer = np.zeros_like(img)
        overlay_buffer = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGRA)

        cv2.putText(overlay_buffer, 'Line Following', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.circle(overlay_buffer, (center_x, center_y), int(self.sampling_rad/4), (64, 255, 64), 1)

        cv2.putText(overlay_buffer, ' SAM_H1: {}'.format(sampling_line_1), (center_x-150, sampling_h1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(overlay_buffer, ' SAM_H2: {}'.format(sampling_line_2), (center_x-150, sampling_h2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)

        cv2.putText(overlay_buffer, f'X: {input_speed:.2f}, Z: {input_turning:.2f}', (center_x+50, center_y+0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(overlay_buffer, ' UPPER: {}'.format(upper_hsv), (center_x+50, center_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_buffer, ' LOWER: {}'.format(lower_hsv), (center_x+50, center_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(overlay_buffer, ' UPPER: {}'.format(line_upper), (center_x+50, center_y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(overlay_buffer, ' LOWER: {}'.format(line_lower), (center_x+50, center_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(overlay_buffer, f' SLOPE: {line_slope:.2f}', (center_x+50, center_y+140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(overlay_buffer, f' SAM_1 SAM_2 SLOPE_IM BASE_IM SPD_IM LT_SPD SLOPE_SPD', (center_x-250, center_y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(overlay_buffer, f' {sampling_line_1:.2f}   {sampling_line_2:.2f}   {slope_impact:.2f}      {base_impact:.4f}  {speed_impact:.2f}    {line_track_speed:.2f}    {slope_on_speed:.2f}', (center_x-250, center_y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)

        cv2.line(overlay_buffer, (0, sampling_h1), (width, sampling_h1), (255, 0, 0), 2)
        cv2.line(overlay_buffer, (0, sampling_h2), (width, sampling_h2), (255, 0, 0), 2)

        if sam_1:
            cv2.line(overlay_buffer, (sampling_1_left, sampling_h1+20), (sampling_1_left, sampling_h1-20), (0, 255, 0), 2)
            cv2.line(overlay_buffer, (sampling_1_right, sampling_h1+20), (sampling_1_right, sampling_h1-20), (0, 255, 0), 2)
        if sam_2:
            cv2.line(overlay_buffer, (sampling_2_left, sampling_h2+20), (sampling_2_left, sampling_h2-20), (0, 255, 0), 2)
            cv2.line(overlay_buffer, (sampling_2_right, sampling_h2+20), (sampling_2_right, sampling_h2-20), (0, 255, 0), 2)
        if sam_1 and sam_2:
            cv2.line(overlay_buffer, (sampling_1_center, sampling_h1), (sampling_2_center, sampling_h2), (255, 0, 0), 2)

        self.overlay = np.zeros_like(img)
        self.overlay = overlay_buffer


    def mediaPipe_faces(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        overlay_buffer = np.zeros_like(image)
        cv2.putText(overlay_buffer, 'MediaPipe Faces', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if results.detections:
            for detection in results.detections:
                mpDraw.draw_detection(overlay_buffer, detection)
        overlay_buffer = cv2.cvtColor(overlay_buffer, cv2.COLOR_RGB2BGR)
        self.overlay = np.zeros_like(img)
        self.overlay = cv2.cvtColor(overlay_buffer, cv2.COLOR_BGR2BGRA)


    def mediaPipe_pose(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        overlay_buffer = np.zeros_like(image)
        cv2.putText(overlay_buffer, 'MediaPipe Pose', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(overlay_buffer, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        overlay_buffer = cv2.cvtColor(overlay_buffer, cv2.COLOR_RGB2BGR)
        self.overlay = np.zeros_like(img)
        self.overlay = cv2.cvtColor(overlay_buffer, cv2.COLOR_BGR2BGRA)


    def cv_process(self, frame):
        cv_mode_list = {
        f['code']['cv_moti']: self.cv_detect_movition,
        f['code']['cv_face']: self.cv_detect_faces,
        f['code']['cv_objs']: self.cv_detect_objects,
        f['code']['cv_clor']: self.cv_detect_color,
        f['code']['cv_hand']: self.cv_detect_hand,
        f['code']['cv_auto']: self.cv_auto_drive,
        f['code']['mp_face']: self.mediaPipe_faces,
        f['code']['mp_pose']: self.mediaPipe_pose
        }
        cv_mode_list[cv_mode](self, frame)
        self.cv_event.clear()


    def opencv_threading(self, input_img):
        cv_thread = threading.Thread(target=self.cv_process, args=(self, input_img,), daemon=True)
        cv_thread.start()


    def info_update(self, megs, color, size):
        global show_info_flag, info_update_time
        wrapped_lines = textwrap.wrap(megs, recv_line_max)
        for line in wrapped_lines:
            info_deque.appendleft({'text':line,'color':color,'size':size})
        info_update_time = time.time()
        show_info_flag = True


    def change_target_color(self, lc, uc):
        global color_lower, color_upper
        color_lower = np.array([lc[0], lc[1], lc[2]])
        color_upper = np.array([uc[0], uc[1], uc[2]])


    def change_line_color(self, lc, uc):
        global line_lower, line_upper
        line_lower = np.array([lc[0], lc[1], lc[2]])
        line_upper = np.array([uc[0], uc[1], uc[2]])


    def set_line_track_args(self, sam_pos_1, sam_pos_2, slope_im, base_im, spd_im, lt_spd, slope_spd):
        global sampling_line_1, sampling_line_2, slope_impact, base_impact, speed_impact, line_track_speed, slope_on_speed
        # findline autodrive
        sampling_line_1 = sam_pos_1
        if sam_pos_2 < sam_pos_1:
            sam_pos_2 = sam_pos_1 + 0.1
        sampling_line_2 = sam_pos_2
        slope_impact = slope_im
        base_impact = base_im
        speed_impact = spd_im
        line_track_speed = lt_spd
        slope_on_speed = slope_spd


    def selet_target_color(self, color_name):
        global color_lower, color_upper
        # print(color_name)
        if color_name in color_list:
            color_lower = color_list[color_name][0]
            color_upper = color_list[color_name][1]


    def timelapse(self, input_speed, input_time, input_interval, input_loop_times):
        global mission_flag
        mission_flag = True
        for i in range(0, input_loop_times):
            if not mission_flag:
                mission_flag = True
                break
            base.base_json_ctrl({"T":1,"L":input_speed,"R":input_speed})
            time.sleep(input_time)
            base.base_json_ctrl({"T":1,"L":0,"R":0})
            time.sleep(input_interval/2)
            base.lights_ctrl(255, 255)
            time.sleep(0.01)
            self.capture_frame(thisPath + '/static/')
            base.lights_ctrl(0, 0)
            time.sleep(input_interval/2)
            if not mission_flag:
                mission_flag = True
                break


    def cmd_process(self, args_str):
        global show_recv_flag, show_info_flag, info_update_time, mission_flag
        global track_color_iterate, track_faces_iterate, track_spd_rate, track_acc_rate
        args = args_str.split()
        if args[0] == 'base':
            self.info_update("CMD:" + args_str, (0,255,255), 0.36)
            if args[1] == '-c' or args[1] == '--cmd':
                base.base_json_ctrl(json.loads(args[2]))
            elif args[1] == '-r' or args[1] == '--recv':
                if args[2] == 'on':
                    show_recv_flag = True
                elif args[2] == 'off':
                    show_recv_flag = False

        elif args[0] == 'info':
            info_update_time = time.time()
            show_info_flag = True

        elif args[0] == 'audio':
            self.info_update("CMD:" + args_str, (0,255,255), 0.36)
            if args[1] == '-s' or args[1] == '--say':
                audio_ctrl.play_speech_thread(' '.join(args[2:]))
            elif args[1] == '-v' or args[1] == '--volume':
                audio_ctrl.set_audio_volume(args[2])
            elif args[1] == '-p' or args[1] == '--play_file':
                audio_ctrl.play_file(args[2])

        elif args[0] == 'test':
            info_update_time = time.time()
            info_deque.appendleft({'text':json.dumps(args[1]),'color':(0,0,255),'size':0.5}) # mac
            wrapped_lines = textwrap.wrap(' '.join(args[2:]), recv_line_max) # megs
            for line in wrapped_lines:
                info_deque.appendleft({'text':line,'color':(255,255,255),'size':0.5})
            show_info_flag = True

        elif args[0] == 'send':
            self.info_update("CMD:" + args_str, (0,255,255), 0.36)
            if args[1] == '-a' or args[1] == '--add':
                if args[2] == '-b' or args[2] == '--broadcast':
                    base.base_json_ctrl({"T":303,"mac":"FF:FF:FF:FF:FF:FF"})
                else:
                    base.base_json_ctrl({"T":303,"mac":args[2]})
            elif args[1] == '-rm' or args[1] == '--remove':
                if args[2] == '-b' or args[2] == '--broadcast':
                    base.base_json_ctrl({"T":304,"mac":"FF:FF:FF:FF:FF:FF"})
                else:
                    base.base_json_ctrl({"T":304,"mac":args[2]})
            elif args[1] == '-b' or args[1] == '--broadcast':
                base.base_json_ctrl({"T":306,"mac":"FF:FF:FF:FF:FF:FF","dev":0,"b":0,"s":0,"e":0,"h":0,"cmd":3,"megs":' '.join(args[2:])})
            elif args[1] == '-g' or args[1] == '--group':
                base.base_json_ctrl({"T":305,"dev":0,"b":0,"s":0,"e":0,"h":0,"cmd":3,"megs":' '.join(args[2:])})
            else:
                base.base_json_ctrl({"T":306,"mac":args[1],"dev":0,"b":0,"s":0,"e":0,"h":0,"cmd":3,"megs":' '.join(args[2:])})

        elif args[0] == 'cv':
            self.info_update("CMD:" + args_str, (0,255,255), 0.36)
            if args[1] == '-r' or args[1] == '--range':
                try:
                    lower_trimmed = args[2].strip("[]")
                    lower_nums = [int(lower_num) for lower_num in lower_trimmed.split(",")]
                    if all(0 <= num <= 255 for num in lower_nums):
                        pass
                    else:
                        return
                except:
                    return
                try:
                    upper_trimmed = args[3].strip("[]")
                    upper_nums = [int(upper_num) for upper_num in upper_trimmed.split(",")]
                    if all(0 <= num <= 255 for num in upper_nums):
                        pass
                    else:
                        return
                except:
                    return
                self.change_target_color(lower_nums, upper_nums)

            elif args[1] == '-s' or args[1] == '--select':
                self.selet_target_color(args[2])

        elif args[0] == 'line':
            self.info_update("CMD:" + args_str, (0,255,255), 0.36)
            if args[1] == '-r' or args[1] == '--range':
                try:
                    lower_trimmed = args[2].strip("[]")
                    lower_nums = [int(lower_num) for lower_num in lower_trimmed.split(",")]
                    if all(0 <= num <= 255 for num in lower_nums):
                        pass
                    else:
                        return
                except:
                    return
                try:
                    upper_trimmed = args[3].strip("[]")
                    upper_nums = [int(upper_num) for upper_num in upper_trimmed.split(",")]
                    if all(0 <= num <= 255 for num in upper_nums):
                        pass
                    else:
                        return
                except:
                    return
                self.change_line_color(lower_nums, upper_nums)
            elif args[1] == '-s' or args[1] == '--set':
                if len(args) != 9:
                    return
                try:
                    for i in range(2,9):
                        float(args[i])
                except:
                    return
                self.set_line_track_args(float(args[2]), float(args[3]), float(args[4]), float(args[5]), float(args[6]), float(args[7]), float(args[8]))

        elif args[0] == 'track':
            self.info_update("CMD:" + args_str, (0,255,255), 0.36)
            if args[1] == '-c' or args[1] == '--color_iterate':
                track_color_iterate = float(args[2])
            elif args[1] == '-f' or args[1] == '--faces_iterate':
                track_faces_iterate = float(args[2])
            elif args[1] == '-s' or args[1] == '--speed':
                track_spd_rate = float(args[2])
            elif args[1] == '-a' or args[1] == '--acc':
                track_acc_rate = float(args[2])

        elif args[0] == 'timelapse':
            if args[1] == '-s' or args[1] == '--start':
                if len(args) != 6:
                    return
                try:
                    move_speed = float(args[2])
                    move_time  = float(args[3])
                    t_interval = float(args[4])
                    loop_times = int(args[5])
                except:
                    return
                self.timelapse(move_speed, move_time, t_interval, loop_times)
            elif args[1] == '-e' or args[1] == '--end' or args[1] == '--stop':
                mission_flag = False
