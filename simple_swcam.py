import PySpin
from pidng.core import RAW2DNG
import cv2
import numpy as np
import csv
import multiprocessing
import time
from datetime import datetime

import workerFunctions
import const
import lib


# helper functions

def readCCM_from_csv():
    CCM = []
    with open('CCM.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            CCM.append([float(i) for i in row])
    return CCM


# Const

SAVE = False
SAVE_TYPE = "compressed"  # "raw" or "compressed"
THREADED = False
RESIZE_FACTOR = 2
NFRAMES = 1000

OUTPUT_PARAMS = const.OUTPUT_PARAMS["mp4"]

# CCM = readCCM_from_csv()
CCM = lib.dngCCM2CCM(const.dngCCM_IMX249_2_sRGB)

RGB_Gain = [
    const.WB_Manual[0][0]/const.WB_Manual[0][1],
    const.WB_Manual[1][0]/const.WB_Manual[1][1],
    const.WB_Manual[2][0]/const.WB_Manual[2][1],
]


if THREADED:

    class imgDebayerWorker(multiprocessing.Process):
        def __init__(self, task_queue, result_queue=None):
            multiprocessing.Process.__init__(self)
            self.task_queue = task_queue
            self.result_queue = result_queue

        def run(self):
            while True:
                bayeredImg = self.task_queue.get()
                if not isinstance(bayeredImg, np.ndarray) and bayeredImg == -1:
                    self.task_queue.task_done()
                    break
                elif bayeredImg is None:
                    ...
                else:
                    # _t = datetime.now()
                    rgbImg = workerFunctions.demosaicing(bayeredImg)
                    # t = datetime.now()
                    # _s = np.round((t - _t).total_seconds(), 3)
                    # print(f"debayer: {_s}s")
                    self.task_queue.task_done()
                    self.result_queue.put(rgbImg)
            print('Done debayering')

    class imgColorCorrectWorker(multiprocessing.Process):
        def __init__(self, task_queue, result_queue=None):
            multiprocessing.Process.__init__(self)
            self.task_queue = task_queue
            self.result_queue = result_queue

        def run(self):
            while True:
                img = self.task_queue.get()
                if not isinstance(img, np.ndarray) and img == -1:
                    self.task_queue.task_done()
                    break
                elif img is None:
                    ...
                else:
                    # _t = datetime.now()
                    img = workerFunctions.colorCorrect(img, CCM=CCM)
                    img = workerFunctions.wbCorrect(img, RGB_Gain=np.array(RGB_Gain, dtype=const.PROCESS_FLOAT))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    img = workerFunctions.saturationCorrect(img)
                    # t = datetime.now()
                    # _s = np.round((t - _t).total_seconds(), 3)
                    # print(f"color: {_s}s")
                    self.task_queue.task_done()
                    self.result_queue.put(img)
            print('Done color correcting')

    class imgLumaCorrectWorker(multiprocessing.Process):
        def __init__(self, task_queue, result_queue=None):
            multiprocessing.Process.__init__(self)
            self.task_queue = task_queue
            self.result_queue = result_queue

        def run(self):
            while True:
                img = self.task_queue.get()
                if not isinstance(img, np.ndarray) and img == -1:
                    self.task_queue.task_done()
                    break
                elif img is None:
                    ...
                else:
                    # _t = datetime.now()
                    img = workerFunctions.lumaCorrect(img)  # hsv
                    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  # bgr
                    img = workerFunctions.clippingCorrection(img).astype(const.DTYPE)
                    # t = datetime.now()
                    # _s = np.round((t - _t).total_seconds(), 3)
                    # print(f"luma: {_s}s")
                    self.task_queue.task_done()
                    self.result_queue[0].put(img)
                    self.result_queue[1].put(img)
            print('Done luma correcting')

    class imgSaveWorker(multiprocessing.Process):
        def __init__(self, task_queue, result_queue=None):
            multiprocessing.Process.__init__(self)
            self.task_queue = task_queue
            self.result_queue = result_queue
            if SAVE and SAVE_TYPE == "compressed":
                from vidgear.gears import WriteGear
                self.writer = WriteGear(
                    output='output/Output.mov',
                    compression_mode=True,
                    logging=True,
                    **OUTPUT_PARAMS
                )

        def run(self):
            while True:
                bgrImg = self.task_queue.get()
                if not isinstance(bgrImg, np.ndarray) and bgrImg == -1:
                    self.task_queue.task_done()
                    break
                elif bgrImg is None:
                    ...
                else:
                    # _t = datetime.now()
                    if SAVE and SAVE_TYPE == "compressed":
                        self.writer.write(bgrImg)
                    self.task_queue.task_done()
                    # t = datetime.now()
                    # _s = np.round((t - _t).total_seconds(), 3)
                    # print(f"saving: {_s}s")
            self.writer.close()
            print('Done saving')

    class imgDisplayWorker(multiprocessing.Process):
        def __init__(self, task_queue, result_queue=None):
            multiprocessing.Process.__init__(self)
            self.task_queue = task_queue
            self.result_queue = result_queue

        def run(self):
            while True:
                bgrImg = self.task_queue.get()
                if not isinstance(bgrImg, np.ndarray) and bgrImg == -1:
                    self.task_queue.task_done()
                    break
                elif bgrImg is None:
                    ...
                else:
                    previewImg = cv2.resize(
                        bgrImg, (int(const.Xres/RESIZE_FACTOR), int(const.Yres/RESIZE_FACTOR)), cv2.INTER_LINEAR)
                    self.task_queue.task_done()
                    cv2.imshow("img", previewImg)
                    key = cv2.waitKey(1) & 0xFF
                    # check for 'q' key-press
                    if key == ord("q"):
                        #  if 'q' key-pressed break out
                        # TODO signal to close all processes
                        ...
            print("Done displaying")


def main():

    # WARM UP (for numba)
    _ = np.random.randint(
        0, np.iinfo(const.DTYPE).max,
        (const.Yres, const.Xres), dtype=const.DTYPE)
    workerFunctions._colourDebayer(_)
    _ = (np.random.rand(
        const.Yres, const.Xres, 3) * np.iinfo(const.DTYPE).max).astype(const.PROCESS_FLOAT)
    workerFunctions.colorCorrect(_, CCM=const.CCM_TEST)
    workerFunctions.wbCorrect(_, np.array([1, 1, 1], dtype=const.PROCESS_FLOAT))
    workerFunctions.lumaCorrect(_)
    workerFunctions.saturationCorrect(_)
    workerFunctions.clippingCorrection(_)
    lib.applyCCM(_, np.array(const.CCM_TEST, dtype=const.PROCESS_FLOAT))
    lib.applyGamma(_)
    # lib.saturationVsLumaCurve(0.0)
    lib.fastRealSat(_, 0.5)

    if THREADED:

        debayerQueue = multiprocessing.JoinableQueue()
        colorCorrectQueue = multiprocessing.JoinableQueue()
        lumaCorrectQueue = multiprocessing.JoinableQueue()
        saveQueue = multiprocessing.JoinableQueue()
        displayQueue = multiprocessing.JoinableQueue()

        demosaicingProcess = imgDebayerWorker(debayerQueue, colorCorrectQueue)
        colorCorrectProcess = imgColorCorrectWorker(colorCorrectQueue, lumaCorrectQueue)
        lumaCorrectProcess = imgLumaCorrectWorker(lumaCorrectQueue, [saveQueue, displayQueue])
        saveProcess = imgSaveWorker(saveQueue,)
        displayProcess = imgDisplayWorker(displayQueue,)

        demosaicingProcess.start()
        colorCorrectProcess.start()
        lumaCorrectProcess.start()
        displayProcess.start()
        saveProcess.start()

        # Numba warmup on threads
        debayerQueue.put(np.ones((const.Yres, const.Xres), dtype=const.DTYPE))
        time.sleep(10)

    else:
        from vidgear.gears import WriteGear
        writer = WriteGear(
            output='output/Output.mov',
            compression_mode=True,
            logging=True,
            **OUTPUT_PARAMS
        )

    system = PySpin.System.GetInstance()

    cam_list = system.GetCameras()

    if cam_list.GetSize() > 0:
        cam = cam_list[0]
    else:
        print('No camera connected')
        cam_list.Clear()
        system.ReleaseInstance()
        exit()

    cam.Init()

    # Doc : https://www.flir.ca/support-center/iis/machine-vision/application-note/spinnaker-nodes/

    FPS = const.FRAMERATE
    RAW_FORMAT = "dng"  # "exr" or "dng"

    nodemap_tldevice = cam.GetTLDeviceNodeMap()  # noqa F841
    nodemap = cam.GetNodeMap()
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

    nodePixelFormatEnum = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    nodePixelFormatEntry = PySpin.CEnumEntryPtr(nodePixelFormatEnum.GetCurrentEntry())  # noqa F841
    pixel_format = nodePixelFormatEnum.GetEntryByName('BayerRG16').GetValue()
    # pixel_format = nodePixelFormatEnum.GetEntryByName('BayerRG12p').GetValue()  # Not supported
    nodePixelFormatEnum.SetIntValue(pixel_format)

    # PySpin.CFloatPtr(nodemap.GetNode('ExposureTime')).SetValue(1/(2 * FPS) * 1e6)
    PySpin.CFloatPtr(nodemap.GetNode('ExposureTime')).SetValue(1/(2 * 30) * 1e6)
    PySpin.CFloatPtr(nodemap.GetNode('BlackLevel')).SetValue(0)
    PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionFrameRateAuto')).SetIntValue(0)
    PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate')).SetValue(float(FPS))
    PySpin.CEnumerationPtr(nodemap.GetNode('VideoMode')).SetIntValue(7)

    cam.BeginAcquisition()

    _t = datetime.now()

    for frame in range(NFRAMES):
        image_result = cam.GetNextImage()
        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
        else:
            ...
        bayeredImg = image_result.GetNDArray()

        # TODO Move to save process
        if SAVE and SAVE_TYPE == "raw":
            if RAW_FORMAT == "dng":
                r = RAW2DNG()
                r.options(lib.dngTag(
                    Xres=const.Xres, Yres=const.Yres, bpp=const.bpp,
                    CCM=const.dngCCM_IMX249_2_sRGB,
                    # CCM=lib.CCM2dngCCM(const.CCM_Leica_M9),
                    COMPRESS=const.COMPRESS, FRAMERATE=const.FRAMERATE, WB=const._WB),
                    path="", compress=const.COMPRESS)
                r.convert(bayeredImg, f"output/frame_{frame}.dng")
            elif RAW_FORMAT == "exr":
                lib.writeEXR(f"output/frame_{frame}.exr", bayeredImg)
            else:
                raise NotImplementedError

        if THREADED:
            debayerQueue.put(bayeredImg)
        else:
            bgrImg = workerFunctions.imgProcess(bayeredImg, CCM=CCM, RGB_Gain=RGB_Gain)
            if SAVE and SAVE_TYPE == "compressed":
                writer.write(bgrImg)
            previewImg = cv2.resize(
                bgrImg, (int(const.Xres/RESIZE_FACTOR), int(const.Yres/RESIZE_FACTOR)), cv2.INTER_LINEAR)
            cv2.imshow("img", previewImg)
            key = cv2.waitKey(1) & 0xFF
            # check for 'q' key-press
            if key == ord("q"):
                #  if 'q' key-pressed break out
                break

        # img = PySpin.Image.Create(image_result)
        # img.Save(f"output/{i}.raw", PySpin.SPINNAKER_IMAGE_FILE_FORMAT_RAW)
        image_result.Release()

        t = datetime.now()
        _s = np.round(1 / (t - _t).total_seconds(), 3)
        print(str(_s) + " FPS")
        _t = t

    if not THREADED and SAVE and SAVE_TYPE == "compressed":
        writer.close()
        # safely close writer

    # close process
    if THREADED:
        queues = [
            (debayerQueue, "Debayer"),
            (colorCorrectQueue, "Color"),
            (lumaCorrectQueue, "Luma"),
            (displayQueue, "Display"),
            (saveQueue, "Save"),
        ]
        print("Sending empty items to close queues...")
        for queue in queues:
            print(f"Closing {queue[1]} queue when empty...")
            queue[0].put(None)
            while queue[0].qsize() != 0:
                if queue[0].qsize() == 0:
                    break
                else:
                    time.sleep(1)
            queue[0].close()
            print(f"Closed {queue[1]}")

        print("terminating processes...")
        demosaicingProcess.terminate()
        colorCorrectProcess.terminate()
        lumaCorrectProcess.terminate()
        displayProcess.terminate()
        saveProcess.terminate()

    print("closing camera feed.")
    cam.EndAcquisition()

    cam.DeInit()

    cam_list.Clear()
    system.ReleaseInstance()

    print("Done!")


if __name__ == "__main__":
    main()
