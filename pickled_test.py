import pickle
import cv2
import numpy as np

import const
import workerFunctions
import lib


CCM = lib.dngCCM2CCM(const.dngCCM_IMX249_2_sRGB)

RGB_Gain = [
    const.WB_Manual[0][0]/const.WB_Manual[0][1],
    const.WB_Manual[1][0]/const.WB_Manual[1][1],
    const.WB_Manual[2][0]/const.WB_Manual[2][1],
]


bayeredImg = pickle.load(open("bayeredImg.pkl", "rb"))
cv2.imshow("img", bayeredImg)
cv2.waitKey(0)

img = workerFunctions.demosaicing(bayeredImg)  # rgb
_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).clip(0, np.iinfo(const.DTYPE).max).astype(const.DTYPE)
cv2.imshow("img", _)
cv2.waitKey(0)

img = workerFunctions.wbCorrect(img, RGB_Gain=np.array(RGB_Gain, dtype=const.PROCESS_FLOAT))  # rgb
_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).clip(0, np.iinfo(const.DTYPE).max).astype(const.DTYPE)
cv2.imshow("img", _)
cv2.waitKey(0)

img = cv2.cvtColor(img.astype(const.PROCESS_FLOAT), cv2.COLOR_RGB2HSV)  # hsv
img = workerFunctions.saturationCorrect(img)  # hsv
_ = cv2.cvtColor(img, cv2.COLOR_HSV2BGR).clip(0, np.iinfo(const.DTYPE).max).astype(const.DTYPE)
cv2.imshow("img", _)
cv2.waitKey(0)

img = workerFunctions.lumaCorrect(img)  # hsv
_ = cv2.cvtColor(img, cv2.COLOR_HSV2BGR).clip(0, np.iinfo(const.DTYPE).max).astype(const.DTYPE)
cv2.imshow("img", _)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  # bgr
workerFunctions.clippingCorrection(img).astype(const.DTYPE, copy=False)  # bgr
cv2.imshow("img", img)
cv2.waitKey(0)
