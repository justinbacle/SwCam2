from vidgear.gears import WriteGear
import cv2
import numpy as np

# Define writer with default output parameters

output_params_prores = {
    "-vcodec": "prores_ks",
    "-profile:v": "standard",
}

output_params_cnhf = {
    "-vcodec": "cfhd",
    "-framerate": 30
}


writer = WriteGear(
    output='Output.mov',
    compression_mode=True,
    logging=True,
    **output_params_cnhf
)

Xres, Yres = (800, 480)

for i in range(100):

    r = i*2
    g = 255-i
    b = 255-i*2

    # Generate Bayer image with random values
    arr = np.array([[b, g], [g, r]], dtype=np.uint8)
    frame = np.tile(arr, Xres // 2)
    frame = np.tile(frame, (Yres // 2, 1))

    # read frames from stream

    # {do something with frame here}
    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)

    # write a modified frame to writer
    writer.write(frame)

    # Show output window
    # cv2.imshow("Output Frame", frame)

cv2.destroyAllWindows()
# close output window

writer.close()
# safely close writer
