from datetime import datetime
import cv2
import numpy as np

import workerFunctions
import const
import lib  # noqa F401

NUM_RUNS = 100


RGB = np.array([1, 1, 1], dtype=const.PROCESS_FLOAT)

_ = np.random.randint(
    0, np.iinfo(const.DTYPE).max, (const.Yres, const.Xres), dtype=const.DTYPE)

# all
workerFunctions.imgProcess(_, CCM=const.CCM_TEST, RGB_Gain=RGB)
_t = datetime.now()
for i in range(NUM_RUNS):
    workerFunctions.imgProcess(_, CCM=const.CCM_TEST, RGB_Gain=RGB)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"total: {_s}s")

# demosaicing
workerFunctions.demosaicing(_)
_t = datetime.now()
for i in range(NUM_RUNS):
    workerFunctions.demosaicing(_)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"demosaic: {_s}s")

_ = (np.random.rand(
    const.Yres, const.Xres, 3) * np.iinfo(const.DTYPE).max
).astype(const.PROCESS_FLOAT)

# Color correct
workerFunctions.colorCorrect(_, CCM=const.CCM_TEST)
_t = datetime.now()
for i in range(NUM_RUNS):
    workerFunctions.colorCorrect(_, CCM=const.CCM_TEST)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"color correct: {_s}s")

# WB correct
workerFunctions.wbCorrect(_, RGB_Gain=RGB)
_t = datetime.now()
for i in range(NUM_RUNS):
    workerFunctions.wbCorrect(_, RGB_Gain=RGB)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"wb correct: {_s}s")

# Color conv
cv2.cvtColor(_, cv2.COLOR_RGB2HSV)
_t = datetime.now()
for i in range(NUM_RUNS):
    cv2.cvtColor(_, cv2.COLOR_RGB2HSV)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"color conv: {_s}s")

# sat correct
workerFunctions.saturationCorrect(_)
_t = datetime.now()
for i in range(NUM_RUNS):
    workerFunctions.saturationCorrect(_)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"saturation correct: {_s}s")

# luma correct
workerFunctions.lumaCorrect(_)
_t = datetime.now()
for i in range(NUM_RUNS):
    workerFunctions.lumaCorrect(_)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"luma correct: {_s}s")

# Color conv
cv2.cvtColor(_, cv2.COLOR_HSV2RGB)
_t = datetime.now()
for i in range(NUM_RUNS):
    cv2.cvtColor(_, cv2.COLOR_HSV2RGB)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"color conv: {_s}s")

# clipping
workerFunctions.clippingCorrection(_).astype(const.DTYPE)
_t = datetime.now()
for i in range(NUM_RUNS):
    # workerFunctions.clippingCorrection(_).astype(const.DTYPE, copy=False)
    workerFunctions.clippingCorrection(_)
t = datetime.now()
_s = np.round((t - _t).total_seconds() / NUM_RUNS, 3)
print(f"clipping: {_s}s")
