import cv2
import colour
import numpy as np
import numpy.core.umath
import numba  # noqa F401
import matplotlib.pyplot as plt  # noqa F401

import const
import lib


if const.DEMOSAICING_METHOD["display"] == "torch":
    import torch  # noqa F401
    import debayer  # pip install git+https://github.com/cheind/pytorch-debayer
    pytorchDebayer = debayer.Debayer5x5(layout=debayer.Layout.RGGB)
elif const.DEMOSAICING_METHOD["display"] == "colour":
    import colour_demosaicing  # noqa F401


def _opencvDebayer(bayeredImg: np.ndarray) -> np.ndarray:
    _img = cv2.cvtColor(bayeredImg, cv2.COLOR_BAYER_RGGB2RGB_EA)
    _img = _img.astype(const.PROCESS_FLOAT)
    return _img


# from : colour_demosaicing.demosaicing_CFA_Bayer_Menon2007
def _colourDebayer(bayeredImg: np.ndarray) -> np.ndarray:
    imgArray = bayeredImg.astype(const.PROCESS_FLOAT)
    # _img = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(imgArray, pattern="RGGB")
    _img = lib._colour_demosaicing_CFA_Bayer_Menon2007(imgArray, pattern="RGGB")
    return _img.astype(const.PROCESS_FLOAT)


def demosaicing(bayeredImg: np.ndarray) -> np.ndarray:
    if const.DEMOSAICING_METHOD["display"] == "opencv":
        _img = _opencvDebayer(bayeredImg)
    elif const.DEMOSAICING_METHOD["display"] == "colour":
        _img = _colourDebayer(bayeredImg)
    elif const.DEMOSAICING_METHOD["display"] == "torch":
        _img = pytorchDebayer(bayeredImg)
    elif const.DEMOSAICING_METHOD["display"] == "fastHalfRes":
        _img = lib.fastHalfRes(bayeredImg)
    else:
        raise NotImplementedError
    return _img


def colorCorrect(rgbImg: np.ndarray, CCM: np.ndarray) -> np.ndarray:
    if const.COLOR_CORRECTION_METHOD == "jb":
        CCM = np.array(lib.dngCCM2CCM(const.dngCCM_IMX249_2_sRGB)).astype(const.PROCESS_FLOAT)
        rgbImg = lib.applyCCM(rgbImg, CCM)
        np.clip(rgbImg, np.iinfo(const.DTYPE).min, np.iinfo(const.DTYPE).max, out=rgbImg)
    elif const.COLOR_CORRECTION_METHOD == "colour":
        rgbImg = colour.characterisation.apply_matrix_colour_correction_Finlayson2015(
            rgbImg, lib.dngCCM2CCM(const.dngCCM_IMX249_2_sRGB)).astype(const.PROCESS_FLOAT)
        np.clip(
            rgbImg, np.iinfo(const.DTYPE).min, np.iinfo(const.DTYPE).max, out=rgbImg)
        rgbImg = rgbImg.astype(const.PROCESS_FLOAT)
    else:
        raise NotImplementedError
    return rgbImg


@numba.njit(fastmath=True, parallel=True, nogil=True)
def wbCorrect(rgbImg: np.ndarray, RGB_Gain: np.ndarray) -> np.ndarray:
    rgbImg = rgbImg * RGB_Gain
    return rgbImg


@numba.njit(fastmath=True, nogil=True)
def lumaCorrect(hsvImg: np.ndarray) -> np.ndarray:
    # Black level compensation
    # hsvImg[:, :, 2] = np.add(hsvImg[:, :, 2], -50)
    # Apply gamma
    hsvImg = lib.applyGamma(hsvImg)
    # apply log profile
    # hsvImg = lib.applyLogValue(hsvImg)
    # apply srgb profile
    return hsvImg


@numba.njit(fastmath=True, nogil=True)
def saturationCorrect(hsvImg: np.ndarray) -> np.ndarray:
    # Saturation curve
    hsvImg = lib.fastRealSat(hsvImg, DESAT=0.5)
    return hsvImg


@numba.njit(fastmath=True)
def clippingCorrection(img: np.ndarray) -> np.ndarray:
    img = np.divide(img, 1.05)
    np.clip(img, 0, np.iinfo(const.DTYPE).max, out=img)
    return img


@numba.vectorize()  # not faster :s
def fastClippingCorrection(img: np.ndarray) -> np.ndarray:
    return max(min(0, np.iinfo(const.DTYPE).max), img)


def imgProcess(bayeredImg: np.ndarray, CCM: np.ndarray, RGB_Gain: np.ndarray) -> np.ndarray:
    # demosaic
    img = demosaicing(bayeredImg)  # rgb
    # color
    img = colorCorrect(img, CCM=CCM)  # rgb
    img = wbCorrect(img, RGB_Gain=np.array(RGB_Gain, dtype=const.PROCESS_FLOAT))  # rgb
    cv2.cvtColor(img.astype(const.PROCESS_FLOAT), cv2.COLOR_RGB2HSV, img)  # hsv

    # plt.ion()
    # plt.subplot(3, 1, 1)
    # plt.cla()
    # c, b = np.histogram(np.log10(np.add(img[:, :, 0].flatten(), 1)), bins=256)
    # plt.stairs(c, b)
    # plt.subplot(3, 1, 2)
    # plt.cla()
    # c, b = np.histogram(np.log10(np.add(img[:, :, 1].flatten(), 1)), bins=256)
    # plt.stairs(c, b)
    # plt.subplot(3, 1, 3)
    # plt.cla()
    # c, b = np.histogram(np.log10(np.add(img[:, :, 2].flatten(), 1)), bins=256)
    # plt.stairs(c, b)
    # plt.draw()
    # plt.pause(0.001)

    img = saturationCorrect(img)  # hsv
    # luma
    img = lumaCorrect(img)  # hsv
    cv2.cvtColor(img, cv2.COLOR_HSV2BGR, img)  # bgr
    img = clippingCorrection(img).astype(const.DTYPE)  # bgr
    return img
