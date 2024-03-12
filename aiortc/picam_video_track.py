import av
import picamera2

from aiortc import VideoStreamTrack


class PicamVideoSttreamTrack(VideoStreamTrack):
    """
    Video track from picamera
    """

    def __init__(self, camera_num=0, video_size=(640, 480)):
        super().__init__()
        self.camera = picamera2.Picamera2(camera_num)
        config = self.camera.create_video_configuration(
            main={"size": video_size, "format": "BGR888"}
        )
        self.camera.configure(config)
        self.camera.start()

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        img = self.camera.capture_array("main")
        frame = av.VideoFrame.from_ndarray(img, "rgb24")
        frame.pts = pts
        frame.time_base = time_base

        return frame

    def stop(self):
        self.camera.stop()
        super().stop()

    def __del__(self):
        if self.camera.is_open:
            self.camera.stop()
