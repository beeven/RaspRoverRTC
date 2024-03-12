import av
import numpy as np
from picamera2 import Picamera2
from PIL import Image

cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
)
cam.start()
array = cam.capture_array("main")
cam.stop()
img = Image.fromarray(array, mode="RGB")
img.save("output.jpg")

print(np.shape(array))
frame = av.VideoFrame.from_ndarray(array, "rgb24")
img2 = frame.to_image()
img2.save("output2.jpg")
