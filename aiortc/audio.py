import av
import time


def record():
    duration = 5
    container = av.open("USB Camera Analog Stereo",format="openal")
    audio_stream = container.streams.get(audio=0)[0]
    out_container = av.open("output.wav","w")
    out_stream = out_container.add_stream(codec_name='pcm_s16le',rate=44100)

    started = time.time()
    for frame in container.decode(audio_stream):
        
        # for frame in packet.decode():
        out_packet = out_stream.encode(frame)
        out_container.mux(out_packet)
        if time.time() - started > duration:
            break

    for packet in out_stream.encode(None):
        out_container.mux(packet)

    container.close()
    out_container.close()

def play():
    container = av.open("output.wav")
    stream = container.streams.audio[0]

    out_container = av.open("default","w",format="alsa")
    out_stream = out_container.add_stream(codec_name="pcm_s16le",rate=44100)

    for frame in container.decode(stream):
        for packet in out_stream.encode(frame):
            out_container.mux(packet)

    out_container.close()
    container.close()


if __name__ == "__main__":
    play()