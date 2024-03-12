import argparse
import asyncio
import json
import logging
import os
import platform
import ssl
import weakref
from typing import List, Set

import socketio
from aiohttp import web
from picam_video_track import PicamVideoSttreamTrack

import base_ctrl
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay

relay: MediaRelay = None
video_tracks = []
audio_tracks = []
speaker: MediaRecorder = None
pcs: Set[RTCPeerConnection] = set()
channels = weakref.WeakSet()
args: argparse.Namespace = None

gimbal = base_ctrl.BaseController("/dev/ttyAMA0", 115200)

# sio = socketio.AsyncServer(async_mode="aiohttp")


# @sio.event
# def connect(sid, environ, auth):
#     print("connect ", sid)


# @sio.event
# def disconnect(sid):
#     print("disconnect ", sid)


# @sio.event
# async def message(sid, data):
#     print(f"message: {sid} {data}")
#     await sio.emit("message", "you sent: " + str(data), sid)
#     # try:
#     #     robot.json_command_handler(data)
#     # except Exception as e:
#     #     print("Error handling JSON data:", e)
#     #     return


async def index(request: web.Request):
    raise web.HTTPFound("/index.html")
    # return web.Response(text="Hello, world")


def create_local_tracks() -> tuple[List[MediaStreamTrack], List[MediaStreamTrack]]:
    global relay, video_tracks, audio_tracks
    if relay is None:
        relay = MediaRelay()
        options = {"framerate": "30", "video_size": "640x480"}
        webcams = []
        if platform.system() == "Darwin":
            options["pixel_format"] = "uyvy422"
            webcams.append(
                MediaPlayer("0:0", format="avfoundation", options=options).video
            )
            webcams.append(
                MediaPlayer("1:none", format="avfoundation", options=options)
            ).video
            video_tracks = [w.video for w in webcams]
            audio_tracks.append(webcams[0].audio)
        elif platform.system() == "Windows":
            webcams.append(
                MediaPlayer("video=Integrated Camera", format="dshow", options=options)
            )
            video_tracks = [w.video for w in webcams]
            audio_tracks.append(webcams[0].audio)
        else:
            # webcams.append(MediaPlayer("/dev/video8", format="v4l2", options=options))
            video_tracks = [w.video for w in webcams]
            with open("/sys/firmware/devicetree/base/model", "r") as f:
                if "raspberry pi" in f.read().lower():
                    track = PicamVideoSttreamTrack()
                    video_tracks.append(track)
            audio_tracks.append(
                MediaPlayer("USB Camera Analog Stereo", format="openal").audio
            )

    return [relay.subscribe(audio) for audio in audio_tracks], [
        relay.subscribe(video) for video in video_tracks
    ]


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


async def broadcast_feedback():
    data = gimbal.on_data_received()
    await asyncio.sleep(1)

    pass


async def offer(request: web.Request):
    params = await request.json()
    offer = RTCSessionDescription(type="offer", sdp=params["sdp"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state changed: {}".format(pc.connectionState))
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        global speaker
        print(f"on track: {track.id}")
        if speaker is None:
            speaker = MediaRecorder("", format="alsa")
            speaker.addTrack(track)
            await speaker.start()

    @pc.on("datachannel")
    async def on_datachannel(channel: RTCDataChannel):
        global channels
        print(f"on datachannel: {channel.label}")

        @channel.on("message")
        def handle_command(msg):
            print(f"Message received: {msg}")
            # print(f"# channels: {len(channels)}")
            gimbal.base_json_ctrl(json.loads(msg))
            # print(type(msg))

            # channel.send(f"You sent: {msg}")

        channels.add(channel)

    audios, videos = create_local_tracks()
    if audios:
        for audio in audios:
            audio_sender = pc.addTrack(audio)
            if args.audio_codec:
                force_codec(pc, audio_sender, args.audio_codec)
            elif args.play_without_decoding:
                raise Exception("You must specify the audio codec using --audio-codec")

    if videos:
        for video in videos:
            video_sender = pc.addTrack(video)
            if args.video_codec:
                force_codec(pc, video_sender, args.video_codec)
            elif args.play_without_decoding:
                raise Exception("You must specify the video codec using --video-codec")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    # print(answer.sdp)
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    global webcams, relay, mic, channels
    print(f"# channels: {len(channels)}")
    coros = [pc.close() for pc in pcs]
    print("closing pcs:{0}".format(len(coros)))
    await asyncio.gather(*coros)
    print("pcs closed")
    pcs.clear()
    if video_tracks:
        print("stopping webcam")
        for track in video_tracks:
            track.stop()
        for track in audio_tracks:
            track.stop()
    print("webcam stopped")

    if speaker is not None:
        print("stopping speaker")
        await speaker.stop()
    print("speaker stopped")
    del relay
    print("relay stopped")
    print(f"channels: {len(channels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument(
        "--audio-codec", help="Force a specific audio codec (e.g. audio/opus)"
    )
    parser.add_argument(
        "--video-codec", help="Force a specific video codec (e.g. video/H264)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    path = os.path.dirname(__file__)
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(
        os.path.join(path, "ca.crt"), os.path.join(path, "ca.key")
    )

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    # sio.attach(app)
    app.add_routes(
        [
            web.get("/", index),
            web.post("/offer", offer),
            web.static("/", os.path.join(path, "web/dist/ptzweb/browser")),
        ]
    )
    # t = asyncio.create_task(broadcast_feedback())
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    # t.cancel()
