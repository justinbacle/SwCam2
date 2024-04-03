import pidng  # pip install git+https://github.com/schoolpost/PiDNG.git
import pidng.defs
from pidng.core import DNGTags, Tag
import numpy as np
import OpenEXR
import scipy
import Imath
import colour_demosaicing
import cv2
import numba  # noqa F401

import const


def dngTag(Xres, Yres, bpp, CCM, COMPRESS, FRAMERATE, WB):

    # set DNG tags.
    t = DNGTags()
    t.set(Tag.ImageWidth, Xres)
    t.set(Tag.ImageLength, Yres)
    t.set(Tag.TileWidth, Xres)
    t.set(Tag.TileLength, Yres)
    t.set(Tag.Orientation, pidng.defs.Orientation.Horizontal)
    t.set(Tag.PhotometricInterpretation, pidng.defs.PhotometricInterpretation.Color_Filter_Array)  # to change ?
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, bpp)
    t.set(Tag.CFARepeatPatternDim, [2, 2])
    t.set(Tag.SampleFormat, pidng.defs.SampleFormat.Uint)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.CFAPattern, pidng.defs.CFAPattern.RGGB)
    # t.set(Tag.BlackLevel, (4096 >> (16 - bpp)))
    t.set(Tag.BlackLevel, (0))
    t.set(Tag.WhiteLevel, ((1 << bpp) - 1))  # set as max level ?
    t.set(Tag.ColorMatrix1, CCM)
    t.set(Tag.CalibrationIlluminant1, pidng.defs.CalibrationIlluminant.D55)
    t.set(Tag.AsShotNeutral, WB)
    t.set(Tag.BaselineExposure, [[0, 100]])
    t.set(Tag.Make, "SW")
    t.set(Tag.Model, "Cam_0")
    t.set(Tag.DNGVersion, pidng.defs.DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, pidng.defs.DNGVersion.V1_2)
    t.set(Tag.PreviewColorSpace, pidng.defs.PreviewColorSpace.sRGB)
    if COMPRESS:
        t.set(Tag.Compression, pidng.defs.Compression.LJ92)
    t.set(Tag.FrameRate, [[FRAMERATE, 1]])
    return t


def color_correction(img, ccm, avoidClipping):
    '''
    Input:
        img: H*W*3 numpy array, input image
        ccm: 3*3 numpy array, color correction matrix
    Output:
        output: H*W*3 numpy array, output image after color correction
    '''
    img2 = img.reshape((img.shape[0] * img.shape[1], 3))
    output = np.matmul(img2, ccm)
    output = output.reshape(img.shape).astype(img.dtype)
    if avoidClipping:
        return np.clip(output, np.iinfo(output.dtype).min, np.iinfo(output.dtype).max)
    else:
        return output


def dngCCM2CCM(dngCCM):
    _CCM = [
        [dngCCM[0][0]/dngCCM[0][1], dngCCM[1][0]/dngCCM[1][1], dngCCM[2][0]/dngCCM[2][1]],
        [dngCCM[3][0]/dngCCM[3][1], dngCCM[4][0]/dngCCM[4][1], dngCCM[5][0]/dngCCM[5][1]],
        [dngCCM[6][0]/dngCCM[6][1], dngCCM[7][0]/dngCCM[7][1], dngCCM[8][0]/dngCCM[8][1]],
    ]
    return _CCM


def CCM2dngCCM(CCM):
    E = 1000000
    dngCCM = [
        [int(CCM[0][0] * E), E], [int(CCM[0][1] * E), E], [int(CCM[0][2] * E), E],
        [int(CCM[1][0] * E), E], [int(CCM[1][1] * E), E], [int(CCM[1][2] * E), E],
        [int(CCM[2][0] * E), E], [int(CCM[2][1] * E), E], [int(CCM[2][2] * E), E],
    ]
    return dngCCM


@numba.njit(fastmath=True, nogil=True)
def applyCCM(inputImg: np.ndarray, CCM: np.ndarray):
    shape = inputImg.shape
    img = np.reshape(inputImg, (-1, 3))
    img = np.reshape(np.transpose(np.dot(CCM, np.transpose(img))), shape)
    return img


@numba.njit(fastmath=True, nogil=True)
def applyGamma(hlsImg):
    hlsImg[:, :, 2] = np.multiply(
        np.sqrt(np.sqrt(
            np.multiply(hlsImg[:, :, 2], 1 / np.iinfo(const.DTYPE).max)
        )), np.iinfo(const.DTYPE).max / 1.2)
    return hlsImg


def writeEXR(filepath, imgData):
    # imgData = np.squeeze(imgData)
    sz = imgData.shape
    if const.DEMOSAICING_METHOD["raw"] == "opencv":
        imgData = cv2.cvtColor(imgData, cv2.COLOR_BayerRG2RGB)
    elif const.DEMOSAICING_METHOD["raw"] == "colour":
        imgData = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(imgData, pattern="RGGB")
        imgData = color_correction(imgData, dngCCM2CCM(const.dngCCM_IMX249_2_sRGB), False)
        RGB_Gain = [
            const.WB[0][0]/const.WB[0][1],
            const.WB[1][0]/const.WB[1][1],
            const.WB[2][0]/const.WB[2][1],
        ]
        for i in range(3):
            imgData[:, :, i] = imgData[:, :, i] * RGB_Gain[i]

    header = OpenEXR.Header(sz[1], sz[0])
    imgData = np.divide(imgData.astype(np.float32), np.iinfo(const.DTYPE).max)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(filepath, header)
    R = (imgData[:, :, 0]).astype(np.float16).tostring()
    G = (imgData[:, :, 1]).astype(np.float16).tostring()
    B = (imgData[:, :, 2]).astype(np.float16).tostring()
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()


@numba.njit(fastmath=True, nogil=True, cache=True)
def saturationVsLumaCurve(v: float, DESAT: float) -> float:
    # get saturation coeff from value [0,1]
    LOW = 0.005
    HIGH = 0.98
    if v < 0.0 or v > 1.0:
        sat = 0.0
    elif v < LOW:
        sat = 1.0 / LOW * v
    elif v > HIGH and v < 1:
        sat = - (1 / (1 - HIGH)) * v + 1 / (1 - HIGH)
    else:
        sat = 1.0
    return sat * DESAT


def realsat(hsvImgArray):
    METHOD = "vectorize"
    if METHOD == "vectorize":
        _f = np.vectorize(lambda x: saturationVsLumaCurve(x), otypes=[const.PROCESS_FLOAT])
        satCoeff = _f(np.divide(hsvImgArray[:, :, 2], np.iinfo(const.DTYPE).max))
    elif METHOD == "applyalongaxis":
        satCoeff = np.apply_along_axis(
            saturationVsLumaCurve, 0, np.divide(hsvImgArray[:, :, 2], np.iinfo(const.DTYPE).max))
    else:
        raise NotImplementedError
    hsvImgArray[:, :, 1] = np.multiply(
        hsvImgArray[:, :, 1],
        satCoeff
    )
    return hsvImgArray


@numba.njit(fastmath=True, parallel=True, cache=True)
def fastRealSat(hsvImgArray, DESAT: np.float32):

    shape = hsvImgArray[:, :, 1].shape
    satCoeff = np.zeros(shape, dtype=const.PROCESS_FLOAT)
    for i in range(shape[0]):
        for j in range(shape[1]):
            satCoeff[i, j] = saturationVsLumaCurve(
                np.divide(hsvImgArray[i, j, 2], np.iinfo(const.DTYPE).max), DESAT=DESAT)
    hsvImgArray[:, :, 1] = np.multiply(hsvImgArray[:, :, 1], satCoeff)
    return hsvImgArray


def logProfile(v):
    return np.sqrt(v)


def _as_float_array(a, dtype=None):
    return np.asarray(a, dtype)


def _masks_CFA_Bayer(shape, pattern="RGGB"):
    channels = {channel: np.zeros(shape, dtype="bool") for channel in "RGB"}
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1
    return tuple(channels.values())


def _cnv_h(x, y):
    return scipy.ndimage.filters.convolve1d(x, y, mode="mirror")


def _cnv_v(x, y):
    return scipy.ndimage.filters.convolve1d(x, y, mode="mirror", axis=0)


def _tstack(a, dtype=None):
    a = np.asarray(a, dtype)
    if a.ndim <= 2:
        return np.transpose(a)
    return np.concatenate([x[..., None] for x in a], axis=-1)


def _tsplit(a, dtype=None):
    a = np.asarray(a, dtype)
    if a.ndim <= 2:
        return np.transpose(a)
    return np.transpose(
        a, np.concatenate([[a.ndim - 1], np.arange(0, a.ndim - 1)]))


def _refining_step_Menon2007(RGB, RGB_m, M):
    R, G, B = _tsplit(RGB)
    R_m, G_m, B_m = _tsplit(RGB_m)
    M = _as_float_array(M)
    R_G = R - G
    B_G = B - G
    FIR = np.ones(3) / 3
    B_G_m = np.where(B_m == 1, np.where(M == 1, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)), 0)
    R_G_m = np.where(R_m == 1, np.where(M == 1, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)), 0)
    G = np.where(R_m == 1, R - R_G_m, G)
    G = np.where(B_m == 1, B - B_G_m, G)
    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    R_c = np.any(R_m == 1, axis=0)[None] * np.ones(R.shape)
    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)
    B_c = np.any(B_m == 1, axis=0)[None] * np.ones(B.shape)
    R_G = R - G
    B_G = B - G
    k_b = _as_float_array([0.5, 0.0, 0.5])
    R_G_m = np.where(np.logical_and(G_m == 1, B_r == 1), _cnv_v(R_G, k_b), R_G_m)
    R = np.where(np.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
    R_G_m = np.where(np.logical_and(G_m == 1, B_c == 1), _cnv_h(R_G, k_b), R_G_m)
    R = np.where(np.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)
    B_G_m = np.where(np.logical_and(G_m == 1, R_r == 1), _cnv_v(B_G, k_b), B_G_m)
    B = np.where(np.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
    B_G_m = np.where(np.logical_and(G_m == 1, R_c == 1), _cnv_h(B_G, k_b), B_G_m)
    B = np.where(np.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)
    R_B = R - B
    R_B_m = np.where(B_m == 1, np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0)
    R = np.where(B_m == 1, B + R_B_m, R)
    R_B_m = np.where(R_m == 1, np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0)
    B = np.where(R_m == 1, R - R_B_m, B)
    return _tstack([R, G, B])


# not compatible with numba
def _colour_demosaicing_CFA_Bayer_Menon2007(CFA, pattern="RGGB", refining_step: bool = True):
    def _cnv_h(x, y):
        # no numpy equivalent ?
        return scipy.ndimage.filters.convolve1d(x, y, mode="mirror")

    def _cnv_v(x, y):
        # no numpy equivalent ?
        return scipy.ndimage.filters.convolve1d(x, y, mode="mirror", axis=0)

    def _tstack(a, dtype=None):
        a = np.asarray(a, dtype)
        if a.ndim <= 2:
            return np.transpose(a)
        return np.concatenate([x[..., None] for x in a], axis=-1)

    def _tsplit(a, dtype=None):
        a = np.asarray(a, dtype)
        if a.ndim <= 2:
            return np.transpose(a)
        return np.transpose(
            a, np.concatenate([[a.ndim - 1], np.arange(0, a.ndim - 1)]))

    def _refining_step_Menon2007(RGB, RGB_m, M):
        R, G, B = _tsplit(RGB)
        R_m, G_m, B_m = _tsplit(RGB_m)
        M = np.array(M, dtype=const.PROCESS_FLOAT)
        R_G = R - G
        B_G = B - G
        FIR = np.ones(3) / 3
        B_G_m = np.where(B_m == 1, np.where(M == 1, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)), 0)
        R_G_m = np.where(R_m == 1, np.where(M == 1, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)), 0)
        G = np.where(R_m == 1, R - R_G_m, G)
        G = np.where(B_m == 1, B - B_G_m, G)
        R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
        R_c = np.any(R_m == 1, axis=0)[None] * np.ones(R.shape)
        B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)
        B_c = np.any(B_m == 1, axis=0)[None] * np.ones(B.shape)
        R_G = R - G
        B_G = B - G
        k_b = np.array([0.5, 0.0, 0.5], dtype=const.PROCESS_FLOAT)
        R_G_m = np.where(np.logical_and(G_m == 1, B_r == 1), _cnv_v(R_G, k_b), R_G_m)
        R = np.where(np.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
        R_G_m = np.where(np.logical_and(G_m == 1, B_c == 1), _cnv_h(R_G, k_b), R_G_m)
        R = np.where(np.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)
        B_G_m = np.where(np.logical_and(G_m == 1, R_r == 1), _cnv_v(B_G, k_b), B_G_m)
        B = np.where(np.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
        B_G_m = np.where(np.logical_and(G_m == 1, R_c == 1), _cnv_h(B_G, k_b), B_G_m)
        B = np.where(np.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)
        R_B = R - B
        R_B_m = np.where(B_m == 1, np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0)
        R = np.where(B_m == 1, B + R_B_m, R)
        R_B_m = np.where(R_m == 1, np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0)
        B = np.where(R_m == 1, R - R_B_m, B)
        return _tstack([R, G, B])

    # CFA = np.squeeze(CFA)
    channels = {channel: np.zeros(CFA.shape, dtype="bool") for channel in "RGB"}
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1
    R_m, G_m, B_m = channels.values()
    h_0 = np.array([0.0, 0.5, 0.0, 0.5, 0.0], dtype=const.PROCESS_FLOAT)
    h_1 = np.array([-0.25, 0.0, 0.5, 0.0, -0.25], dtype=const.PROCESS_FLOAT)
    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m
    G_H = np.where(G_m == 0, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = np.where(G_m == 0, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)
    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)
    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)
    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])
    k = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ], dtype=const.PROCESS_FLOAT
    )
    d_H = scipy.ndimage.filters.convolve(D_H, k, mode="constant")
    d_V = scipy.ndimage.filters.convolve(D_V, np.transpose(k), mode="constant")
    # d_H = np.convolve(D_H, k, mode="same")
    # d_V = np.convolve(D_V, k, mode="same")
    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)
    M = np.where(mask, 1, 0)
    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)
    k_b = np.array([0.5, 0, 0.5], dtype=const.PROCESS_FLOAT)
    R = np.where(np.logical_and(G_m == 1, R_r == 1), G + _cnv_h(R, k_b) - _cnv_h(G, k_b), R)
    R = np.where(np.logical_and(G_m == 1, B_r == 1) == 1, G + _cnv_v(R, k_b) - _cnv_v(G, k_b), R)
    B = np.where(np.logical_and(G_m == 1, B_r == 1), G + _cnv_h(B, k_b) - _cnv_h(G, k_b), B)
    B = np.where(np.logical_and(G_m == 1, R_r == 1) == 1, G + _cnv_v(B, k_b) - _cnv_v(G, k_b), B)
    R = np.where(
        np.logical_and(B_r == 1, B_m == 1),
        np.where(M == 1, B + _cnv_h(R, k_b) - _cnv_h(B, k_b), B + _cnv_v(R, k_b) - _cnv_v(B, k_b)), R)
    B = np.where(
        np.logical_and(R_r == 1, R_m == 1),
        np.where(M == 1, R + _cnv_h(B, k_b) - _cnv_h(R, k_b), R + _cnv_v(B, k_b) - _cnv_v(R, k_b)), B)
    RGB = _tstack([R, G, B])
    if refining_step:
        RGB = _refining_step_Menon2007(RGB, _tstack([R_m, G_m, B_m]), M)
    return RGB


# Color filtering: `rggb`
@numba.njit()
def bayer(im):
    r = np.zeros(im.shape[:2], dtype=const.PROCESS_FLOAT)
    g = np.zeros(im.shape[:2], dtype=const.PROCESS_FLOAT)
    b = np.zeros(im.shape[:2], dtype=const.PROCESS_FLOAT)
    r[0::2, 0::2] += im[0::2, 0::2]
    g[0::2, 1::2] += im[0::2, 1::2]
    g[1::2, 0::2] += im[1::2, 0::2]
    b[1::2, 1::2] += im[1::2, 1::2]
    return r, g, b
