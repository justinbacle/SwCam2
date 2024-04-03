import cv2
import numpy as np
from pidng.core import RAW2DNG
import colour

import lib
import const

# Define writer with default output parameters

output_params_prores = {
    "-vcodec": "prores_ks",
    "-profile:v": "standard",
}

output_params_cnhf = {
    "-vcodec": "cfhd",
    "-framerate": 30
}

NFRAMES = 100

BLACK_LEVEL = 100
WHITE_LEVEL = np.iinfo(const.DTYPE).max - BLACK_LEVEL

colour.CCS_COLOURCHECKERS

# should be -> Color temp: 6500K, Tint : 10

# writer = WriteGear(
#     output='output/Output.mov',
#     compression_mode=True,
#     logging=True,
#     **output_params_cnhf
# )

# Full patches + noise
for i in range(NFRAMES):

    if i < NFRAMES / 6:
        r, g, b = (BLACK_LEVEL, BLACK_LEVEL, BLACK_LEVEL)
    elif i < 2 * NFRAMES / 6:
        r, g, b = (WHITE_LEVEL / 4, WHITE_LEVEL / 4, WHITE_LEVEL / 4)
    elif i < 3 * NFRAMES / 6:
        r, g, b = (WHITE_LEVEL, WHITE_LEVEL, WHITE_LEVEL)
    elif i < 4 * NFRAMES / 6:
        r, g, b = (WHITE_LEVEL, BLACK_LEVEL, BLACK_LEVEL)
    elif i < 5 * NFRAMES / 6:
        r, g, b = (BLACK_LEVEL, WHITE_LEVEL, BLACK_LEVEL)
    else:
        r, g, b = (BLACK_LEVEL, BLACK_LEVEL, WHITE_LEVEL)

    # Generate Bayer image with random values
    arr = np.array([[r, g], [g, b]], dtype=const.DTYPE)
    frame = np.tile(arr, const.Xres // 2)
    frame = np.tile(frame, (const.Yres // 2, 1))

    frame = frame + np.random.randint(
        0, BLACK_LEVEL * 2, size=(const.Yres, const.Xres), dtype=const.DTYPE) - BLACK_LEVEL

    r = RAW2DNG()
    r.options(lib.dngTag(
        const.Xres, const.Yres, const.bpp, const.dngCCM_xyz2srgb, const.COMPRESS,
        const.FRAMERATE), path="", compress=const.COMPRESS)
    r.convert(frame, f"output/frame_{i}.dng")

greyRamp = np.zeros(shape=(const.Yres, const.Xres), dtype=const.DTYPE)
for i in range(const.Xres):
    for j in range(const.Yres):
        greyRamp[j, i] = BLACK_LEVEL + i / const.Xres * (WHITE_LEVEL - BLACK_LEVEL)


def isRedPixel(i, j):
    return i % 2 == 0 and j % 2 == 0


def isGreenPixel(i, j):
    return i % 2 == 0 and j % 2 == 1 or i % 2 == 1 and j % 2 == 0


def isBluePixel(i, j):
    return i % 2 == 1 and j % 2 == 1


redRamp = np.zeros(shape=(const.Yres, const.Xres), dtype=const.DTYPE)
for i in range(const.Xres):
    for j in range(const.Yres):
        if isRedPixel(i, j):
            redRamp[j, i] = greyRamp[j, i]
        else:
            redRamp[j, i] = BLACK_LEVEL

greenRamp = np.zeros(shape=(const.Yres, const.Xres), dtype=const.DTYPE)
for i in range(const.Xres):
    for j in range(const.Yres):
        if isGreenPixel(i, j):
            greenRamp[j, i] = greyRamp[j, i]
        else:
            greenRamp[j, i] = BLACK_LEVEL

blueRamp = np.zeros(shape=(const.Yres, const.Xres), dtype=const.DTYPE)
for i in range(const.Xres):
    for j in range(const.Yres):
        if isBluePixel(i, j):
            blueRamp[j, i] = greyRamp[j, i]
        else:
            blueRamp[j, i] = BLACK_LEVEL

# not working ?
# colorRamp = np.zeros(shape=(Yres, Xres), dtype=DTYPE)
# for i in range(Xres):
#     for j in range(Yres):
#         redOrigin = (0, 0)
#         greenOrigin = (Yres, 0)
#         blueOrigin = (Yres, Xres)
#         color = (
#             np.sqrt(
#                 (j - redOrigin[0]) ** 2 + (i - redOrigin[1]) ** 2) / np.sqrt(Xres ** 2 + Yres ** 2) * WHITE_LEVEL,
#             np.sqrt(
#                 (j - greenOrigin[0]) ** 2 + (i - greenOrigin[1]) ** 2) / np.sqrt(Xres ** 2 + Yres ** 2) * WHITE_LEVEL,
#             np.sqrt(
#                 (j - blueOrigin[0]) ** 2 + (i - blueOrigin[1]) ** 2) / np.sqrt(Xres ** 2 + Yres ** 2) * WHITE_LEVEL
#         )

#     if isRedPixel(i, j):
#         colorRamp[j, i] = color[0]
#     elif isGreenPixel(i, j):
#         colorRamp[j, i] = color[1]
#     elif isBluePixel(i, j):
#         colorRamp[j, i] = color[2]

colorRampImg = cv2.imread("color_ramp.png")
colorRampImg = cv2.cvtColor(colorRampImg, cv2.COLOR_BGR2RGB)

colorRamp = np.zeros(shape=(const.Yres, const.Xres), dtype=const.DTYPE)
for i in range(const.Xres):
    for j in range(const.Yres):
        if isRedPixel(i, j):
            colorRamp[j, i] = colorRampImg[
                int(j / const.Yres * colorRamp.shape[0]),
                int(i / const.Xres * colorRamp.shape[1]), 0] / 255 * WHITE_LEVEL
        elif isGreenPixel(i, j):
            colorRamp[j, i] = colorRampImg[
                int(j / const.Yres * colorRamp.shape[0]),
                int(i / const.Xres * colorRamp.shape[1]), 1] / 255 * WHITE_LEVEL
        elif isBluePixel(i, j):
            colorRamp[j, i] = colorRampImg[
                int(j / const.Yres * colorRamp.shape[0]),
                int(i / const.Xres * colorRamp.shape[1]), 2] / 255 * WHITE_LEVEL


for i in range(NFRAMES):

    if i < NFRAMES / 6:
        frame = greyRamp
    elif i < 2 * NFRAMES / 6:
        frame = redRamp
    elif i < 3 * NFRAMES / 6:
        frame = greenRamp
    elif i < 4 * NFRAMES / 6:
        frame = blueRamp
    elif i < 5 * NFRAMES / 6:
        frame = colorRamp
    else:
        ...

    r = RAW2DNG()
    r.options(lib.dngTag(
        const.Xres, const.Yres, const.bpp, const.dngCCM_xyz2srgb, const.COMPRESS, const.FRAMERATE),
        path="", compress=const.COMPRESS)
    r.convert(frame, f"output/frame_{NFRAMES + i}.dng")


# cv2.destroyAllWindows()
# close output window

# writer.close()
# safely close writer
