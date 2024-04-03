using OpenCvSharp;
using SpinnakerNET;
using SpinnakerNET.GenApi;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using BitMiracle.LibTiff.Classic;
//using Sdcb.LibRaw;
using RawBayer2DNG;
//using Emgu.CV;
//using NRawTherapee;


namespace Acquisition_CSharp
{
    class Program
    {
        public const int NFRAMES = (int)1e6;
        public const float FPS = 30;
        public const float RESIZE_FACTOR = 1.5f;
        public const ulong TIMEOUT = (ulong)(1 / FPS * 1e3);
        public const int WIDTH = 1920;
        public const int HEIGHT = 1080;
        public const bool THREADED = true;
        public const int PARALLEL_CHUNKS = 6;  // 6 seems optimum
        public const string CV2_WINDOWNAME = "im";

        static int PrintDeviceInfo(INodeMap nodeMap)
        {
            int result = 0;
            try
            {
                Trace.WriteLine("\n*** DEVICE INFORMATION ***\n");
                ICategory category = nodeMap.GetNode<ICategory>("DeviceInformation");
                if (category != null && category.IsReadable)
                {
                    for (int i = 0; i < category.Children.Length; i++)
                    {
                        if (category.Children[i].IsReadable)
                        { Trace.WriteLine(category.Children[i].Name + ": " + category.Children[i].ToString()); }
                        else { Trace.WriteLine(category.Children[i].Name + ": Node not available"); }
                    }
                }
                else { Trace.WriteLine("Device control information not available."); }
            }
            catch (SpinnakerException ex)
            {
                Trace.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }
            return result;
        }
        static Mat ProcessImg(Mat matImg, Mat CCM, float[] RGB_Gain)
        {
            int DMAX = (int)Math.Pow(2, 16);
            int height = matImg.Height;
            int width = matImg.Width;

            Mat _im1 = new Mat(rows: height, cols: width, type: MatType.CV_16UC3);
            Mat im1Reshaped = new Mat(rows: (int)(width * height), cols: 1, type: MatType.CV_32FC3);
            // Go to float data
            matImg.ConvertTo(_im1, MatType.CV_32FC3);
            // Apply CCM
            im1Reshaped = _im1.Reshape(1, height * width);
            im1Reshaped = im1Reshaped * CCM;
            _im1 = im1Reshaped.Reshape(3, height, width);
            // WB
            //_im1.ConvertTo(_im1, MatType.CV_16UC3);
            Mat[] rgbSplit = _im1.Split();
            for (int c = 0; c < 3; c++)
            {
                rgbSplit[c] = rgbSplit[c] * RGB_Gain[c] / RGB_Gain.Max();
            }
            Cv2.Merge(rgbSplit, _im1);

            Cv2.CvtColor(_im1, _im1, ColorConversionCodes.RGB2HLS_FULL);
            Mat[] hlsSplit = _im1.Split();

            //// Luma Max  // TODO have better implementation
            //double max, min;
            //Cv2.MinMaxLoc(hlsSplit[1], maxVal: out max, minVal: out min);
            ////Trace.WriteLine("Min {0} Max {1}", min, max);
            //hlsSplit[1] = (hlsSplit[1] - min) / (max - min) * DMAX;
            //Cv2.MinMaxLoc(hlsSplit[1], maxVal: out a, minVal: out b);
            //Trace.WriteLine("Min {0} Max {1}", min, max);

            double min = 00;
            double max = 35000;
            hlsSplit[1] = (hlsSplit[1] - min) / (max - min) * DMAX;
            //Cv2.MinMaxLoc(hlsSplit[1], maxVal: out max, minVal: out min);
            //Trace.WriteLine("Min: " + min + " Max: " + max);

            // Luma Gamma
            Cv2.Pow(hlsSplit[1] / DMAX, 1 / 2.0f, hlsSplit[1]);
            Cv2.Log(hlsSplit[1] + 1.0f, hlsSplit[1]);
            hlsSplit[1] = hlsSplit[1] * DMAX / Math.Log(2);

            // Sat vs Luma
            // https://github.com/shimat/opencvsharp/wiki/Accessing-Pixel
            // TypeSpecificMat (faster)
            var _lumaMat = new Mat<float>(hlsSplit[1]);
            var _lumaIndexer = _lumaMat.GetIndexer();
            var _satMat = new Mat<float>(hlsSplit[2]);
            var _satIndexer = _satMat.GetIndexer();
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    _satIndexer[y, x] = _satIndexer[y, x] * satCoeff(_lumaIndexer[y, x] / DMAX, 0.5f);
                }
            }
            hlsSplit[1] = _lumaMat;
            Cv2.Merge(hlsSplit, _im1);

            // HSV -> BGR
            Cv2.CvtColor(_im1, _im1, ColorConversionCodes.HLS2BGR_FULL);
            return _im1;
        }

        private static Mat[] ProcessImageQuadrantsParallel(Mat[] matList)
        {
            Mat[] processedMatList = new Mat[PARALLEL_CHUNKS];
            Parallel.For(0, PARALLEL_CHUNKS, i => { processedMatList[i] = ProcessImg(matList[i], getCcmMat(), getLiveRgbGain()); });
            return processedMatList;
        }

        static float[] getRgbGain()
        {
            float[] RGB_Gain = { (float)(0.6), (float)(0.5), (float)(1.0) };
            Vector3 rgbVec = new Vector3(RGB_Gain[0], RGB_Gain[1], RGB_Gain[2]);
            rgbVec = Vector3.Normalize(rgbVec);
            RGB_Gain[0] = rgbVec[0] * (float)Math.Sqrt(3);
            RGB_Gain[1] = rgbVec[1] * (float)Math.Sqrt(3);
            RGB_Gain[2] = rgbVec[2] * (float)Math.Sqrt(3);
            //Mat RGB_Gain_Map = new Mat(3, 1, MatType.CV_32FC1, RGB_Gain);
            //float [] RGB_Gain = {(float)1.8, (float)0.5, (float)2.5};
            //float[] RGB_Gain = { (float)0.60, (float)0.78, (float)1.41 };
            //NDArray RGB_Gain_arr = np.array(RGB_Gain);
            return RGB_Gain;
        }

        static float[] getLiveRgbGain()
        {
            return [
                Cv2.GetTrackbarPos("R", CV2_WINDOWNAME),
                Cv2.GetTrackbarPos("G", CV2_WINDOWNAME),
                Cv2.GetTrackbarPos("B", CV2_WINDOWNAME)
            ];
        }

        static float[,] getCCM()
        {
            float[,] CCM = {
                {(float)1.543012,  (float)-0.263652, (float)-0.279517},
                {(float)-0.261021, (float)1.578747,  (float)-0.317627},
                {(float)0.139825,  (float)-0.70511,  (float)1.565359}
            };
            return CCM;
        }

        static Mat getCcmMat()
        {
            
            Mat CCM_Mat = new Mat(3, 3, MatType.CV_32FC1, getCCM());
            //Cv2.Rotate(CCM_Mat, CCM_Mat, RotateFlags.Rotate180);
            //NDArray CCM_arr = ToNDArray(CCM_Mat);
            return CCM_Mat;
        }

        static Mat imageProcess(ushort[] flatBayeredData)
        {
            Mat im0 = new Mat(HEIGHT, WIDTH, MatType.CV_16UC1);
            Mat im1 = new Mat(rows: HEIGHT, cols: WIDTH, type: MatType.CV_16UC3);
            Mat im1Reshaped = new Mat(rows: (int)(WIDTH * HEIGHT), cols: 1, type: MatType.CV_32FC1);
            // Debayer
            im0 = new Mat(HEIGHT, WIDTH, MatType.CV_16UC1, flatBayeredData);
            Cv2.CvtColor(im0, im1, ColorConversionCodes.BayerBG2RGB_EA);  // move to process img func for speedup ?

            Cv2.Size size = new OpenCvSharp.Size((int)(WIDTH / RESIZE_FACTOR), (int)(HEIGHT / RESIZE_FACTOR));
            if (RESIZE_FACTOR != 1)
            {
                Cv2.Resize(im1, im1, size, interpolation: InterpolationFlags.Cubic);
            }

            if (THREADED)
            {
                Mat[] imageChunks = new Mat[PARALLEL_CHUNKS];
                for (int chunkNum = 0; chunkNum < PARALLEL_CHUNKS; chunkNum++)
                {
                    imageChunks[chunkNum] =
                        new Mat(im1, new OpenCvSharp.Rect(chunkNum * im1.Width / PARALLEL_CHUNKS, 0, im1.Width / PARALLEL_CHUNKS, im1.Height));
                }
                Mat[] processedQuadrants = ProcessImageQuadrantsParallel(imageChunks);
                Mat im_ = processedQuadrants[0];
                for (int i = 1; i < PARALLEL_CHUNKS; i++)
                {
                    Cv2.HConcat([im_, processedQuadrants[i]], im_);
                }
                im1 = im_;
            }
            else
            {
                im1 = ProcessImg(im1, getCcmMat(), getRgbGain());
                //im1 = ProcessImg(new Mat(im1, new Rect(0, 0, (int)(WIDTH / 2), (int)(HEIGHT / 2))), getCcmMat(), getRgbGain());
            }

            // show average
            //int a = 10;
            //Mat crop = new Mat(_im1, new Rect((int)(width / 2 - a), (int)(height/ 2 - a), (int)(a * 2), (int)(a * 2)));
            //Trace.WriteLine(crop.Mean());
            return im1;
        }

        static int AcquireImages(IManagedCamera cam, INodeMap nodeMap, INodeMap nodeMapTLDevice)
        {
            int result = 0;

            //int[] s = new int[256];
            //Mat LumaVsSatLUT = new Mat(1, 256, MatType.CV_8UC1, new Scalar(0));
            //for (int l = 0; l < 256; l++)
            //{
            //    int satMult = (int)(satCoeff((float)(l) / 256, (float)0.5) * 256);
            //    //s[l] = (int)(satMult);
            //    //Trace.WriteLine(satMult);
            //    LumaVsSatLUT.Set(0, l, satMult);
            //}
            //Cv2.Resize(LumaVsSatLUT, LumaVsSatLUT, new Size((int)(width / RESIZE_FACTOR), (int)(height / RESIZE_FACTOR)), interpolation: InterpolationFlags.Lanczos4);
            //Cv2.ImShow("LUT", LumaVsSatLUT);
            Trace.WriteLine("\n*** IMAGE ACQUISITION ***\n");

            IEnum iAcquisitionMode = nodeMap.GetNode<IEnum>("AcquisitionMode");
            IEnumEntry iAcquisitionModeContinuous = iAcquisitionMode.GetEntryByName("Continuous");
            iAcquisitionMode.Value = iAcquisitionModeContinuous.Symbolic;
            Trace.WriteLine("Acquisition mode set to continuous...");

            IEnum iPixelFormat = nodeMap.GetNode<IEnum>("PixelFormat");
            IEnumEntry iPixelFormatBayerRG16 = iPixelFormat.GetEntryByName("BayerRG16");
            iPixelFormat.Value = iPixelFormatBayerRG16.Symbolic;

            IInteger iOffsetX = nodeMap.GetNode<IInteger>("OffsetX");
            iOffsetX.Value = (int)((1920 - WIDTH) / 2);
            IInteger iWidth = nodeMap.GetNode<IInteger>("Width");
            iWidth.Value = WIDTH;
            try
            {
                IInteger iOffsetY = nodeMap.GetNode<IInteger>("OffsetY");
                iOffsetY.Value = (int)((1200 - HEIGHT) / 2);
                IInteger iHeight = nodeMap.GetNode<IInteger>("Height");
                iHeight.Value = HEIGHT;
            }
            catch (Exception e)
            {
                IInteger iHeight = nodeMap.GetNode<IInteger>("Height");
                iHeight.Value = HEIGHT;
                IInteger iOffsetY = nodeMap.GetNode<IInteger>("OffsetY");
                iOffsetY.Value = (int)((1200 - HEIGHT) / 2);
            }

            IFloat iExposureTime = nodeMap.GetNode<IFloat>("ExposureTime");
            iExposureTime.Value = 1 / (2 * FPS) * 1e6;

            IFloat iGain = nodeMap.GetNode<IFloat>("Gain");
            iGain.Value = 0.0;

            IEnum iAcquisitionFrameRateAuto = nodeMap.GetNode<IEnum>("AcquisitionFrameRateAuto");
            iAcquisitionFrameRateAuto.Value = 0;

            IFloat iAcquisisionFrameRate = nodeMap.GetNode<IFloat>("AcquisitionFrameRate");
            iAcquisisionFrameRate.Value = FPS;

            IEnum iVideoMode = nodeMap.GetNode<IEnum>("VideoMode");
            iVideoMode.Value = 7;

            IManagedImageProcessor processor = new ManagedImageProcessor();
            processor.SetColorProcessing(ColorProcessingAlgorithm.HQ_LINEAR);

            cam.BeginAcquisition();
            Trace.WriteLine("Acquiring images...");

            Mat im2 = new Mat(rows: (int)(HEIGHT / RESIZE_FACTOR), cols: (int)(WIDTH / RESIZE_FACTOR), type: MatType.CV_16UC3);

            DateTime t;
            DateTime _t = DateTime.Now;

            Cv2.ImShow(CV2_WINDOWNAME, im2);
            Cv2.CreateTrackbar("R", CV2_WINDOWNAME, count: 200);
            Cv2.SetTrackbarPos("R", CV2_WINDOWNAME, pos: (int)(getRgbGain()[0] * 100));
            Cv2.CreateTrackbar("G", CV2_WINDOWNAME, count: 200);
            Cv2.SetTrackbarPos("G", CV2_WINDOWNAME, pos: (int)(getRgbGain()[1] * 100));
            Cv2.CreateTrackbar("B", CV2_WINDOWNAME, count: 200);
            Cv2.SetTrackbarPos("B", CV2_WINDOWNAME, pos: (int)(getRgbGain()[2] * 100));

            for (int frameNumber = 0; frameNumber < NFRAMES; frameNumber++)
            {
                IManagedImage rawImage = cam.GetNextImage(100);
                if (rawImage.IsIncomplete)
                {
                    Trace.WriteLine("Image incomplete with image status " + rawImage.ImageStatus + "...");
                }
                else
                {

                    byte[] rawData = rawImage.ManagedData;
                    uint bpp = rawImage.BitsPerPixel;
                    ushort[] flatBayeredData = new ushort[HEIGHT * WIDTH];
                    for (int i = 0; i < flatBayeredData.Length; i++)
                    {
                        flatBayeredData[i] = BitConverter.ToUInt16(rawData, i * 2);
                    }

                    //saveImgSpinnaker(rawImage, processor, frameNumber);
                    //createTiff(flatBayeredData, frameNumber);
                    //saveCv2(flatBayeredData, frameNumber);
                    //saveRaw(flatBayeredData, frameNumber);
                    //saveDng(flatBayeredData, frameNumber);

                    Mat im1 = imageProcess(flatBayeredData);

                    im1.ConvertTo(im1, MatType.CV_16UC3);
                    
                    Cv2.ImShow(CV2_WINDOWNAME, im1);
                    int key = Cv2.WaitKey(1);
                    if (key == 113)  // q
                    {
                        rawImage.Release();
                        break;
                    }

                }
                rawImage.Release();
                t = DateTime.Now;
                Trace.WriteLine("FPS: " + Math.Round(1.0 / (t - _t).TotalSeconds, 2));
                Console.WriteLine("FPS: " + Math.Round(1.0 / (t - _t).TotalSeconds, 2));
                _t = DateTime.Now;
            }

            return result;
        }

        // ----------------- FILE SAVING ------------------------------

        static string getProjectDir()
        {
            string workingDirectory = Environment.CurrentDirectory;
            string projectDirectory = Directory.GetParent(workingDirectory).Parent.Parent.Parent.FullName;
            return projectDirectory;
        }

        static void saveRaw(ushort[] flatBayeredData, int frameNumber)
        {
            
            string filename = "test_" + frameNumber + ".raw";
            string fullPath = getProjectDir() + "/output/" + filename;
            byte[] rawBytes = new byte[flatBayeredData.Length * 2];
            Buffer.BlockCopy(flatBayeredData, 0, rawBytes, 0, rawBytes.Length);
            using (var fs = new FileStream(fullPath, FileMode.Create, FileAccess.Write))
            {
                fs.Write(rawBytes, 0, rawBytes.Length);
            }
        }

        static void saveCv2(ushort[] flatBayeredData, int frameNumber)
        {
            string filename = "test_" + frameNumber + ".tiff";
            string fullPath = getProjectDir() + "/output/" + filename;
            Mat im0 = new Mat(HEIGHT, WIDTH, MatType.CV_16UC1, flatBayeredData);
            Cv2.ImWrite(fullPath, im0);
        }

        static void saveImgSpinnaker(IManagedImage rawImage, IManagedImageProcessor processor, int frameNumber)
        {
            string fileName = getProjectDir() + "/output/test_" + frameNumber + ".raw";
            IManagedImage convertedImage = processor.Convert(rawImage, PixelFormatEnums.BGR16);
            convertedImage.Save(fileName);
        }

        static void saveDng(ushort[] flatBayeredData, int frameNumber)
        {
            var dngStream = new dng_stream();
            DNGLosslessEncoder.EncodeLosslessJPEG(flatBayeredData, WIDTH, HEIGHT, 1, 16, 1, 1, dngStream);
            //RawBayer2DNG.MainWindow.Process
        }

        static void createTiff(ushort[] flatBayeredData, int frameNumber)
        {
            string fileName = getProjectDir() + "/output/test_"+frameNumber+".dng";

            //using RawContext r = RawContext.OpenBayerData(new ReadOnlySpan<ushort>(flatBayeredData), WIDTH, HEIGHT, bayerPattern: OpenBayerPattern.RGGB);
            //r.OutputTiff = true;

            // Create a new TIFF file
            using (Tiff tiff = Tiff.Open(fileName, "w"))
            {
                if (tiff == null)
                {
                    Console.WriteLine("Error creating TIFF file.");
                    return;
                }

                // Set TIFF tags
                tiff.SetField(TiffTag.DNGVERSION, [1, 4, 0, 0]);
                tiff.SetField(TiffTag.DNGBACKWARDVERSION, [1, 2, 0, 0]);
                tiff.SetField(TiffTag.MAKE, "SW");
                tiff.SetField(TiffTag.MODEL, "Cam0");
                tiff.SetField(TiffTag.PHOTOMETRIC, 32803);
                tiff.SetField(TiffTag.IMAGEWIDTH, WIDTH); // Set image width
                tiff.SetField(TiffTag.IMAGELENGTH, HEIGHT); // Set image height
                //tiff.SetField(TiffTag.STRIPBYTECOUNTS, WIDTH * HEIGHT * 2);
                tiff.SetField(TiffTag.SAMPLEFORMAT, 1);
                tiff.SetField(TiffTag.BITSPERSAMPLE, 16);
                tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                tiff.SetField(TiffTag.ROWSPERSTRIP, HEIGHT);
                tiff.SetField(TiffTag.ACTIVEAREA, [0, 0, HEIGHT, WIDTH]);
                tiff.SetField(TiffTag.DEFAULTCROPSIZE, [WIDTH, HEIGHT]);
                tiff.SetField(TiffTag.DEFAULTCROPORIGIN, [0, 0]);
                tiff.SetField(TiffTag.CFALAYOUT, [1, 2, 0, 1]);
                
                //tiff.SetField(TiffTag.CFALAYOUT, [2, 1, 1, 0]);
                //tiff.SetField(TiffTag.BLACKLEVEL, 0);
                //tiff.SetField(TiffTag.WHITELEVEL, (ushort)65535);

                byte[] rawBytes = new byte[flatBayeredData.Length * 2];
                Buffer.BlockCopy(flatBayeredData, 0, rawBytes, 0, rawBytes.Length);

                // Write raw Bayer data
                for (int row = 0; row < 1080; row++)
                {
                    tiff.WriteScanline(rawBytes, row);
                }

                // Set DNG-specific tags (adjust as needed)
                //tiff.SetField(TiffTag.DNGVERSION, "1.4.0.0");
                //tiff.SetField(TiffTag.DNGBACKWARDVERSION, "1.3.0.0");
                // Add other DNG tags as needed

                // Save the TIFF file as DNG
                tiff.Close();
            }

        }

        int RunSingleCamera(IManagedCamera cam)
        {
            int result = 0;

            INodeMap nodeMapTLDevice = cam.GetTLDeviceNodeMap();
            result = PrintDeviceInfo(nodeMapTLDevice);
            // Initialize camera
            cam.Init();
            // Retrieve GenICam nodemap
            INodeMap nodeMap = cam.GetNodeMap();

            // Acquire images
            result = result | AcquireImages(cam, nodeMap, nodeMapTLDevice);

            // convert raw frames
            //convertRawFramesToDng();

            return result;
        }

        static void convertRawFramesToDng()
        {
            var rawFiles = Directory.EnumerateFiles(getProjectDir() + "/output", "*.raw");
            foreach (string file in rawFiles)
            {
                string _f = file;
                byte[] contents = File.ReadAllBytes(file);
                ushort[] flatBayeredData = new ushort[contents.Length / 2];
                Buffer.BlockCopy(contents, 0, flatBayeredData, 0, contents.Length);
                int frameNumber;
                frameNumber = int.Parse(file.Split("/")[1].Split("_")[1].Split(".")[0]);
                saveCv2(flatBayeredData, frameNumber);
                //File.Delete(file);
                string tiffFile = file.Replace(".raw", ".tiff");
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = getProjectDir() + "/makeDNG/makeDNG.exe",
                        Arguments = tiffFile + " " + tiffFile.Replace(".tiff", ".dng") + " 3 1 0 0"
                    }
                };
                process.Start();
                process.WaitForExit();
                
                //File.Delete(tiffFile);
            }
        }

        static int Main(string[] args)
        {
            ManagedSystem system = new SpinnakerNET.ManagedSystem();
            ManagedCameraList camList = system.GetCameras();
            Trace.WriteLine("Number of cameras detected: " + camList.Count + "\n\n");
            Program program = new Program();
            int result;

            if (camList.Count == 0)
            {
                // Clear camera list before releasing system
                camList.Clear();
                // Release system
                system.Dispose();
                result = -1;
            }
            else
            {
                result = program.RunSingleCamera(camList[0]);
                Trace.WriteLine("Done aquiring. Closing.");
            }
            // Clear camera list before releasing system
            camList.Clear();
            // Release system
            system.Dispose();
            return result;
        }

        static float satCoeff(float value, float DESAT)
        {
            float LOW = (float)0.0005;
            float HIGH = (float)0.9;
            //float MAXRANGE = (float)Math.Pow(2.0, 16.0);
            float MAXRANGE = (float)1.0;
            float _v = value / MAXRANGE;
            float sat;
            if (value < 0.0 | _v > 1.0)
            {
                sat = (float)0.0;
            }
            else if (_v < LOW)
            {
                sat = 1 / LOW * _v;
            }
            else if (_v > HIGH & _v < 1)
            {
                sat = -(1 / (1 - HIGH)) * _v + 1 / (1 - HIGH);
            }
            else
            {
                sat = (float)1.0;
            }
            return sat * DESAT;
        }
    }
}
