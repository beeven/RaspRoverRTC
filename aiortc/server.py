import argparse
import asyncio
import json
import logging
import os
import platform
import ssl
from typing import Set

import socketio
from aiohttp import web

import aiortc
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay

relay: MediaRelay = None
webcam: MediaPlayer = None
mic: MediaPlayer = None
speaker: MediaRecorder = None
pcs: Set[RTCPeerConnection] = set()
commandChannel: RTCDataChannel = None

sio = socketio.AsyncServer(async_mode="aiohttp")


@sio.event
def connect(sid, environ, auth):
    print("connect ", sid)


@sio.event
def disconnect(sid):
    print("disconnect ", sid)


@sio.event
async def message(sid, data):
    print(f"message: {sid} {data}")
    await sio.emit("message", "you sent: " + str(data), sid)
    # try:
    #     robot.json_command_handler(data)
    # except Exception as e:
    #     print("Error handling JSON data:", e)
    #     return


async def index(request: web.Request):
    raise web.HTTPFound("/index.html")
    # return web.Response(text="Hello, world")


def create_local_tracks():
    global relay, webcam, mic
    if relay is None:
        options = {"framerate": "30", "video_size": "640x480"}
        if platform.system() == "Darwin":
            webcam = MediaPlayer("0:0", format="avfoundation", options=options)
        elif platform.system() == "Windows":
            webcam = MediaPlayer(
                "video=Integrated Camera", format="dshow", options=options
            )
        else:
            webcam = MediaPlayer("/dev/video8", format="v4l2", options=options)
            mic = MediaPlayer("USB Camera Analog Stereo", format="openal")

        relay = MediaRelay()
    return relay.subscribe(mic.audio) if mic is not None else None, relay.subscribe(
        webcam.video
    )


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


async def offer(request: web.Request):
    params = await request.json()
    offer = RTCSessionDescription(type="offer", sdp=params["sdp"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state changed: {}".format(pc.connectionState))
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        print(f"on track: {track.id}")

    @pc.on("datachannel")
    async def on_datachannel(channel: RTCDataChannel):
        global commandChannel
        print(f"on datachannel: {channel.label}")
        if commandChannel is None:
            commandChannel = channel

            @commandChannel.on("message")
            def on_message(msg):
                print(f"on message: {msg}")
                commandChannel.send(f"You sent: {str(msg)}")

    audio, video = create_local_tracks()
    if audio:
        audio_sender = pc.addTrack(audio)
        if args.audio_codec:
            force_codec(pc, audio_sender, args.audio_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the audio codec using --audio-codec")

    if video:
        video_sender = pc.addTrack(video)
        if args.video_codec:
            force_codec(pc, video_sender, args.video_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the video codec using --video-codec")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    print("closing pcs:{0}".format(len(coros)))
    await asyncio.gather(*coros)
    print("pcs closed")
    pcs.clear()
    if webcam is not None:
        print("stopping webcam")
        if webcam.video:
            webcam.video.stop()
        if webcam.audio:
            webcam.audio.stop()
    print("webcam stopped")
    if mic is not None:
        print("stopping mic")
        if mic.audio is not None:
            mic.audio.stop()
        if mic.video is not None:
            mic.video.stop()
    print("mic stopped")
    if speaker is not None:
        print("stopping speaker")
        await speaker.stop()
    print("speaker stopped")


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

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("ca.crt", "ca.key")

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.add_routes(
        [web.static("/", "../web/dist"), web.get("/", index), web.post("/offer", offer)]
    )

    sio.attach(app)

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
