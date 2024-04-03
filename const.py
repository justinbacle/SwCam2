import numpy as np

Xres, Yres = (1920, 1200)
bpp = 16
DTYPE = np.uint16
COMPRESS = False  # Not working ?
FRAMERATE = 10
PROCESS_FLOAT = np.float32
COLOR_CORRECTION_METHOD = "jb"  # "colour" or "jb"
DEMOSAICING_METHOD = {  # "colour" or "opencv" or "torch" or "menon"
    "display": "opencv",  # also does compressed
    "raw": "colour"
}

dngCCM_unit = [
    [1000, 1000], [0, 1000], [0, 1000],
    [0, 1000], [1000, 1000], [0, 1000],
    [0, 1000], [0, 1000], [1000, 1000]
]

# sRGB to XYZ : http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
dngCCM_srgb2xyz = [
    [4124564, 10000000], [3575761, 10000000], [1804375, 10000000],  # noqa
    [2126729, 10000000], [7151522, 10000000], [ 721750, 10000000],  # noqa
    [ 193339, 10000000], [1191920, 10000000], [9503041, 10000000],  # noqa
]

# XYZ to sRGB
dngCCM_xyz2srgb = [
    [ 32404542, 10000000], [-15371385, 10000000], [- 4985314, 10000000],  # noqa
    [- 9692660, 10000000], [ 18760108, 10000000], [   415560, 10000000],  # noqa
    [   556434, 10000000], [- 2040259, 10000000], [ 10572252, 10000000],  # noqa
]

dngCCM_IMX249_2_sRGB = [
    [1543012, 1000000], [-263652, 1000000],  [-279517, 1000000],
    [-261021, 1000000], [1578747, 1000000],  [-317627, 1000000],
    [139825, 1000000], [-705110, 1000000],  [1565359, 1000000],
]

CCM_BFLY_U3_23S6C = [
    [1.8, -0.25, -0.5],
    [-0.35, 1.2, -0.2],
    [-0.15, -0.15, 2.3],
]

CCM_TEST = [
    [-0.01,  0.11, 0.87],
    [-0.00,  0.69, -0.00],
    [1.03, -0.20, 0.19],
]

CCM_Leica_M9 = [
    [0.856, -0.2034, -0.0066],
    [-0.424, 1.36, 2.92],
    [-0.074, 0.247, 0.898],
]

WB_Leica_M9 = [
    [418953, 1000000], [1000000, 1000000], [818381, 1000000]
]

WB_Unit = [
    [1, 1], [1, 1], [1, 1]
]

WB = [
   [1816, 1000], [1000, 1000], [1232, 1000]
]
_WB = [  # 1/gain
   [600, 1000], [500, 1000], [1000, 1000]
]

WB_Manual = [
    [18, 10], [5, 10], [25, 10]
]

OUTPUT_PARAMS = {
    "prores": {
        "-vcodec": "prores_ks",
        "-profile:v": "standard",
    },
    "mp4": {
        "-vcodec": "libx264",
        "-crf": 0,
        "-preset": "fast"
    },
    "cineform": {
        "-vcodec": "cfhd",
    }
}
