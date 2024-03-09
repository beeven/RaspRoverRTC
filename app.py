#!/usr/bin/env python3
from importlib import import_module
import os, socket, psutil
import subprocess, re, netifaces
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, send_file
from werkzeug.utils import secure_filename
import json

from flask_socketio import SocketIO, emit

from robot_ctrl import Camera
from robot_ctrl import RobotCtrlMiddleWare

import time
import logging
import threading

import yaml

curpath = os.path.realpath(__file__)
thisPath = os.path.dirname(curpath)
with open(thisPath + '/config.yaml', 'r') as yaml_file:
    f = yaml.safe_load(yaml_file)

robot_name  = f['base_config']['robot_name']
sbc_version = f['base_config']['sbc_version']

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True

socketio = SocketIO(app)
camera = Camera()
robot = RobotCtrlMiddleWare()

net_interface = "wlan0"
wifi_mode = "None"
eth0_ip = None
wlan_ip = None

UPLOAD_FOLDER = thisPath + '/sounds/others'

pic_size = 0;
vid_size = 0;
cpu_read = 0;
cpu_temp = 0;
ram_read = 0;
rssi_read= 0;

cmd_actions = {
    f['code']['min_res']: lambda: camera.set_video_resolution("240P"),
    f['code']['mid_res']: lambda: camera.set_video_resolution("480P"),
    f['code']['max_res']: lambda: camera.set_video_resolution("960P"),
    f['code']['zoom_x1']: lambda: camera.scale_frame(1),
    f['code']['zoom_x2']: lambda: camera.scale_frame(2),
    f['code']['zoom_x4']: lambda: camera.scale_frame(4),
    f['code']['pic_cap']: lambda: camera.capture_frame(thisPath + '/static/'),
    f['code']['vid_sta']: lambda: camera.record_video(1, thisPath + '/videos/'),
    f['code']['vid_end']: lambda: camera.record_video(0, thisPath + '/videos/'),
    f['code']['cv_none']: lambda: camera.set_cv_mode(f['code']['cv_none']),
    f['code']['cv_moti']: lambda: camera.set_cv_mode(f['code']['cv_moti']),
    f['code']['cv_face']: lambda: camera.set_cv_mode(f['code']['cv_face']),
    f['code']['cv_objs']: lambda: camera.set_cv_mode(f['code']['cv_objs']),
    f['code']['cv_clor']: lambda: camera.set_cv_mode(f['code']['cv_clor']),
    f['code']['cv_hand']: lambda: camera.set_cv_mode(f['code']['cv_hand']),
    f['code']['cv_auto']: lambda: camera.set_cv_mode(f['code']['cv_auto']),
    f['code']['mp_face']: lambda: camera.set_cv_mode(f['code']['mp_face']),
    f['code']['mp_pose']: lambda: camera.set_cv_mode(f['code']['mp_pose']),
    f['code']['re_none']: lambda: camera.set_detection_reaction(f['code']['re_none']),
    f['code']['re_capt']: lambda: camera.set_detection_reaction(f['code']['re_capt']),
    f['code']['re_reco']: lambda: camera.set_detection_reaction(f['code']['re_reco']),
    f['code']['mc_lock']: lambda: camera.set_movtion_lock(f['code']['mc_lock']),
    f['code']['mc_unlo']: lambda: camera.set_movtion_lock(f['code']['mc_unlo']),
    f['code']['led_off']: robot.set_led_mode_off,
    f['code']['led_aut']: robot.set_led_mode_auto,
    f['code']['led_ton']: robot.set_led_mode_on,
    f['code']['base_of']: robot.set_base_led_off,
    f['code']['base_on']: robot.set_base_led_on,
    f['code']['head_ct']: robot.head_led_ctrl,
    f['code']['base_ct']: robot.base_led_ctrl,
    f['code']['s_panid']: camera.set_pan_id,
    f['code']['release']: camera.release_torque,
    f['code']['set_mid']: camera.middle_set,
    f['code']['s_tilid']: camera.set_tilt_id
}


@app.route('/config')
def get_config():
    with open(thisPath + '/config.yaml', 'r') as file:
        yaml_content = file.read()
    return yaml_content


def get_signal_strength(interface):
    try:
        output = subprocess.check_output(["/sbin/iwconfig", interface]).decode("utf-8")
        signal_strength = re.search(r"Signal level=(-\d+)", output)
        if signal_strength:
            return int(signal_strength.group(1))
        return 0
    except FileNotFoundError:
        print("iwconfig command not found. Please ensure it's installed and in your PATH.")
        return -1
    except subprocess.CalledProcessError as e:
        print(f"Error executing iwconfig: {e}")
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


def get_wifi_mode():
    global wifi_mode
    try:
        result = subprocess.check_output(['/sbin/iwconfig', 'wlan0'], encoding='utf-8')

        if "Mode:Master" in result or "Mode:AP" in result:
            wifi_mode = "AP"
            return "AP"

        if "Mode:Managed" in result:
            wifi_mode = "STA"
            return "STA"

    except subprocess.CalledProcessError as e:
        print(f"Error checking Wi-Fi mode: {e}")
        return None

    return None


def get_ip_address(interface):
    try:
        interface_info = netifaces.ifaddresses(interface)

        ipv4_info = interface_info.get(netifaces.AF_INET, [{}])
        return ipv4_info[0].get('addr')
    except ValueError:
        print(f"Interface {interface} not found.")
        return None
    except IndexError:
        print(f"No IPv4 address assigned to {interface}.")
        return None


def get_cpu_usage():
    return psutil.cpu_percent(interval=2)


def get_cpu_temperature():
    try:
        temperature_str = os.popen('vcgencmd measure_temp').readline()
        temperature = float(temperature_str.replace("temp=", "").replace("'C\n", ""))
        return temperature
    except Exception as e:
        print("Error reading CPU temperature:", str(e))
        return None


def get_memory_usage():
    return psutil.virtual_memory().percent


def update_device_info():
    global pic_size, vid_size, cpu_read, ram_read, rssi_read, cpu_temp
    cpu_read = get_cpu_usage()
    cpu_temp = get_cpu_temperature()
    ram_read = get_memory_usage()
    rssi_read= get_signal_strength(net_interface)


def update_data_websocket():
    while 1:
        try:
            fb_json = camera.get_status()
        except:
            continue
        socket_data = {
                    f['fb']['picture_size']:pic_size,
                    f['fb']['video_size']:  vid_size,
                    f['fb']['cpu_load']:    cpu_read,
                    f['fb']['cpu_temp']:    cpu_temp,
                    f['fb']['ram_usage']:   ram_read,
                    f['fb']['wifi_rssi']:   rssi_read
                    }
        try:
            socket_data.update(fb_json)
            socketio.emit('update', socket_data, namespace='/ctrl')
        except:
            pass
        time.sleep(0.1)


@app.route('/')
def index():
    """Video streaming home page."""
    robot.play_random_audio("connected", False)
    return render_template('index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('templates', filename)


@app.route('/photo/<path:filename>')
def serve_static_photo(filename):
    return send_from_directory('templates', filename)


@app.route('/video/<path:filename>')
def serve_static_video(filename):
    return send_from_directory('templates', filename)
    

@app.route('/settings/<path:filename>')
def serve_static_settings(filename):
    return send_from_directory('templates', filename)


@app.route('/index')
def serve_static_home(filename):
    return redirect(url_for('index'))
    

def gen(cameraInput):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = cameraInput.get_frame()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_photo_names')
def get_photo_names():
    photo_files = sorted(os.listdir(thisPath + '/static'), key=lambda x: os.path.getmtime(os.path.join(thisPath + '/static', x)), reverse=True)
    return jsonify(photo_files)


@app.route('/get_photo/<filename>')
def get_photo(filename):
    return send_from_directory(thisPath + '/static', filename)


@app.route('/delete_photo', methods=['POST'])
def delete_photo():
    filename = request.form.get('filename')
    try:
        os.remove(os.path.join(thisPath + '/static', filename))
        return jsonify(success=True)
    except Exception as e:
        print(e)
        return jsonify(success=False)


@app.route('/delete_video', methods=['POST'])
def delete_video():
    filename = request.form.get('filename')
    try:
        os.remove(os.path.join(thisPath + '/videos', filename))
        return jsonify(success=True)
    except Exception as e:
        print(e)
        return jsonify(success=False)


@app.route('/get_video_names')
def get_video_names():
    video_files = sorted(
        [filename for filename in os.listdir(thisPath + '/videos/') if filename.endswith('.mp4')],
        key=lambda filename: os.path.getctime(os.path.join(thisPath + '/videos/', filename)),
        reverse=True
    )
    return jsonify(video_files)


@app.route('/videos/<path:filename>')
def videos(filename):
    return send_from_directory(thisPath + '/videos', filename)


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    # Convert total_size to MB
    size_in_mb = total_size / (1024 * 1024)
    return round(size_in_mb,2)


@socketio.on('message', namespace='/ctrl')
def handle_socket_cmd(message):
    try:
        json_data = json.loads(message)
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return
    cmd_a = float(json_data.get("A", 0))
    # cmd_b = float(json_data.get("B", 0))
    # cmd_c = float(json_data.get("C", 0))
    if cmd_a in cmd_actions:
        cmd_actions[cmd_a]()
    else:
        pass


@socketio.on('json', namespace='/json')
def handle_socket_json(json):
    try:
        robot.json_command_handler(json)
    except Exception as e:
        print("Error handling JSON data:", e)
        return


def oled_update():
    global eth0_ip, wlan_ip
    robot.base_oled(0, f"E: No Ethernet")
    robot.base_oled(1, f"W: NO {net_interface}")
    robot.base_oled(2, "F/J:5000/8888")
    get_wifi_mode()
    start_time = time.time()
    last_folder_check_time = 0

    while True:
        current_time = time.time()

        if current_time - last_folder_check_time > 600:
            pic_size = get_folder_size(thisPath + '/static')
            vid_size = get_folder_size(thisPath + '/videos')
            last_folder_check_time = current_time
        
        update_device_info() # the interval of this loop is set in here
        get_wifi_mode()

        if get_ip_address('eth0') != eth0_ip:
            eth0_ip = get_ip_address('eth0');
            if eth0_ip:
                robot.base_oled(0, f"E:{eth0_ip}")
            else:
                robot.base_oled(0, f"E: No Ethernet")

        if get_ip_address(net_interface) != wlan_ip:
            wlan_ip = get_ip_address(net_interface)
            if wlan_ip:
                robot.base_oled(1, f"W:{wlan_ip}")
            else:
                robot.base_oled(1, f"W: NO {net_interface}")

        elapsed_time = current_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        robot.base_oled(3, f"{wifi_mode} {hours:02d}:{minutes:02d}:{seconds:02d} {rssi_read}dBm")


@app.route('/send_command', methods=['POST'])
def handle_command():
    command = request.form['command']
    print("Received command:", command)
    # camera.info_update("CMD:" + command, (0,255,255), 0.36)
    camera.cmd_process(command)
    return jsonify({"status": "success", "message": "Command received"})


@app.route('/getAudioFiles', methods=['GET'])
def get_audio_files():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    return jsonify(files)


@app.route('/uploadAudio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return jsonify({'success': 'File uploaded successfully'})


@app.route('/playAudio', methods=['POST'])
def play_audio():
    audio_file = request.form['audio_file']
    print(thisPath + '/sounds/others/' + audio_file)
    robot.audio_play(thisPath + '/sounds/others/' + audio_file)
    return jsonify({'success': 'Audio is playing'})


@app.route('/stop_audio', methods=['POST'])
def audio_stop():
    robot.audio_stop()
    return jsonify({'success': 'Audio stop'})


def cmd_on_boot():
    cmd_list = [
        'base -c {"T":142,"cmd":50}',   # set feedback interval
        'base -c {"T":131,"cmd":1}',    # serial feedback flow on
        'base -c {"T":143,"cmd":0}',    # serial echo off
        'base -c {"T":4,"cmd":2}',      # select the module - 0:None 1:RoArm-M2-S 2:Gimbal
        'base -c {"T":300,"mode":0,"mac":"EF:EF:EF:EF:EF:EF"}',  # the base won't be ctrl by esp-now broadcast cmd, but it can still recv broadcast megs.
        'send -a -b'    # add broadcast mac addr to peer
    ]
    for i in range(0, len(cmd_list)):
        camera.cmd_process(cmd_list[i])



if __name__ == '__main__':
    robot.play_random_audio("robot_started", False)
    robot.set_led_mode_on()
    date_update_thread = threading.Thread(target=update_data_websocket, daemon=True)
    date_update_thread.start()
    oled_update_thread = threading.Thread(target=oled_update, daemon=True)
    oled_update_thread.start()
    pic_size = get_folder_size(thisPath + '/static')
    vid_size = get_folder_size(thisPath + '/videos')
    robot.set_led_mode_off()
    cmd_on_boot()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)