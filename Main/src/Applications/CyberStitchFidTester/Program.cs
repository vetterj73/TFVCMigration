using System;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Threading;
using Cyber.ImageUtils;
using MPanelIO;
using Cyber.DiagnosticUtils;
using Cyber.MPanel;
using MCoreAPI;
using MLOGGER;
using MMosaicDM;
using PanelAlignM;
using SIMMosaicUtils;

namespace CyberStitchFidTester
{
    /// <summary>
    /// The idea behind this program is to run regression tests on CyberStitch.  
    /// The Concept:
    /// 1)  Input a SIM Simulation set, a panel file with a small number of fids and a test file with a large number of fids.
    /// 2)  Output a text file that gives you an indication of how far off the fiducials are in the stitched image 
    /// compared to known locations.
    /// 3)  Extra Credit:  This runs as part of the nightly build.  Each night, we can compare a magic number (the total distance offset)
    /// to the last magic number.  If the Magic Number is increasing, we should be nervous because something we changed is causing problems.
    /// NOTE:  I added the ability to bypass CyberStitch by sending in an image file directly.  This causes some headaches because the image
    /// file may be built with other tools (2dSPI for instance) and therefore may have a different pixel size and stride.  This muddied the 
    /// water, but it should be handled correctly.
    /// </summary>
    class Program
    {
        private const string headerLine = "Panel#, Fid#, X, Y ,XOffset, YOffset, CorrScore, Ambig";

        private const double cPixelSizeInMeters = 1.70e-5;
        private static ManagedMosaicSet _mosaicSetSim = null;
        private static ManagedMosaicSet _mosaicSetIllum = null;
        private static ManagedMosaicSet _mosaicSetProcessing = null;
        private static CPanel _processingPanel = null;
        private static CPanel _fidPanel = null;
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread logger = new LoggingThread(null);
        private static int _cycleCount = 0;
        private static double _allPanelFidDifference = 0.0;
        private static double _dXDiffSqrSumTol = 0.0;//used for the total Xoffset square sum
        private static double _dYDiffSqrSumTol = 0.0;
        private static double _dXRMS = 0.0;//used for the xoffset RMS
        private static double _dYRMS = 0.0;

        // For debug
        private static int _iBufCount = 0;
        private static bool _bSimulating = false;
        private static bool _bBayerPattern = false;
        private static int _iBayerType = 1; // GBRG
        private static StreamWriter writer = null;

        // For output analysis
        private static double[] _dXDiffSum;
        private static double[] _dYDiffSum;
        private static double[] _dXAbsDiffSum;
        private static double[] _dYAbsDiffSum;
        private static double[] _dXDiffSqrSum;
        private static double[] _dYDiffSqrSum;        
        private static int[] _icycleCount;

        /// <summary>
        /// This works similar to CyberStitchTester.  The differences:
        /// 1)  All images are obtained prior to running through CyberStitch.
        /// 2)  Color images are pre-converted to illum images for faster processing.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Gather input data.
            string simulationFile = "";
            string panelFile = "";
            string fidPanelFile = "";
            bool bUseProjective = false;
            bool bSaveStitchedResultsImage = false;
            int numberToRun = 1;
            string unitTestFolder="";
            ManagedFeatureLocationCheck fidChecker;

            //output csv file shows the comparison results
            string outputTextPath = @".\fidsCompareResults.csv";
            string lastOutputTextPath = @".\lastFidsCompareResults.csv";
            string imagePath = "";
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-b")
                    _bBayerPattern = true;
                else if (args[i] == "-f" && i < args.Length - 1)
                    fidPanelFile = args[i + 1];
                else if (args[i] == "-i" && i < args.Length - 1)
                    imagePath = args[i + 1];
                else if (args[i] == "-l" && i < args.Length - 1)
                    lastOutputTextPath = args[i + 1];
                else if (args[i] == "-n" && i < args.Length - 1)
                    numberToRun = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-o" && i < args.Length - 1)
                    outputTextPath = args[i + 1];
                else if (args[i] == "-p" && i < args.Length - 1)
                    panelFile = args[i + 1];
                else if (args[i] == "-r")
                    bSaveStitchedResultsImage = true;
                else if (args[i] == "-s" && i < args.Length - 1)
                    simulationFile = args[i + 1];
                else if (args[i] == "-u" && i < args.Length - 1)
                    unitTestFolder = args[i + 1];
                else if (args[i] == "-w")
                    bUseProjective = true;
                else if (args[i] == "-h" && i < args.Length - 1)
                {
                    ShowHelp();
                    return;
                }
            }

            // Start the logger
            logger.Start("Logger", @"c:\\", "CyberStitch.log", true, -1);
            logger.AddObjectToThreadQueue("CyberStitchFidTester Version: " + Assembly.GetExecutingAssembly().GetName().Version);

            // Open output text file.
            writer = new StreamWriter(outputTextPath);

            bool bImageOnly = false;
            if (File.Exists(imagePath))
                bImageOnly = true;
            if (!bImageOnly && File.Exists(panelFile))
            {
                _processingPanel = LoadProductionFile(panelFile, cPixelSizeInMeters);
                if (_processingPanel == null)
                {
                    Terminate();
                    Console.WriteLine("Could not load Panel File: " + panelFile);
                    return;
                }
            } else if (!bImageOnly) // Panel file must exist...
            {
                Terminate();
                Console.WriteLine("The panel file does not exist..: " + panelFile);
                return;
            }

            Bitmap inputBmp = null;
            if(bImageOnly)
                inputBmp = new Bitmap(imagePath);


            int ifidsNum = 0;
            double pixelSize = cPixelSizeInMeters;
            if (File.Exists(fidPanelFile))
            {
                _fidPanel = LoadProductionFile(fidPanelFile, cPixelSizeInMeters);
                if (bImageOnly && _fidPanel.GetNumPixelsInY() != inputBmp.Width)
                {
                    pixelSize = _fidPanel.PanelSizeY/inputBmp.Width;
                    _fidPanel = LoadProductionFile(fidPanelFile, pixelSize);
                }
                if (_fidPanel == null)
                {
                    Terminate();
                    Console.WriteLine("Could not load Fid Test File: " + fidPanelFile);
                    return;
                }
                fidChecker = new ManagedFeatureLocationCheck(_fidPanel);
                ifidsNum = _fidPanel.NumberOfFiducials;
                _dXDiffSum = new double[ifidsNum];
                _dYDiffSum = new double[ifidsNum];
                _dXAbsDiffSum = new double[ifidsNum];
                _dYAbsDiffSum = new double[ifidsNum];
                _dXDiffSqrSum = new double[ifidsNum];
                _dYDiffSqrSum = new double[ifidsNum];
                _icycleCount = new int[ifidsNum];

                for (int i = 0; i < ifidsNum; i++)
                {
                    _dXDiffSum[i] = 0;
                    _dYDiffSum[i] = 0;
                    _dXAbsDiffSum[i] = 0;
                    _dYAbsDiffSum[i] = 0;
                    _dXDiffSqrSum[i] = 0;
                    _dYDiffSqrSum[i] = 0;
                    _icycleCount[i] = 0;
                }
            }
            else
            {
                Terminate();
                Console.WriteLine("Fid Test File does not exist: " + fidPanelFile);
                return;
            }

            if (bImageOnly)
            {
                // This allows images to directly be sent in instead of using CyberStitch to create them
                CyberBitmapData cbd = new CyberBitmapData();

                cbd.Lock(inputBmp);
                writer.WriteLine(headerLine);
                fidChecker = new ManagedFeatureLocationCheck(_fidPanel);
                RunFiducialCompare(cbd.Scan0, cbd.Stride, writer, fidChecker);
                if (bSaveStitchedResultsImage)
                {
                    _aligner.Save3ChannelImage("c:\\Temp\\FidCompareImage.bmp",
                                         cbd.Scan0, cbd.Stride,
                                          _fidPanel.GetCADBuffer(), _fidPanel.GetNumPixelsInY(),
                                          _fidPanel.GetCADBuffer(),_fidPanel.GetNumPixelsInY(),
                                          _fidPanel.GetNumPixelsInY(), _fidPanel.GetNumPixelsInX());          
                }
                cbd.Unlock();
                Terminate();
                return;
            }


            // Initialize the SIM CoreAPI
            if (!InitializeSimCoreAPI(simulationFile))
            {
                Terminate();
                Console.WriteLine("Could not initialize Core API");
                return;
            }

            // Set up mosaic set
            SetupMosaic(true, false);

            try
            {
                Output("Aligner ChangeProduction");
                // Setup Aligner...
                _aligner.OnLogEntry += OnLogEntryFromClient;
                _aligner.SetAllLogTypes(true);
                _aligner.NumThreads(8);
                //_aligner.LogFiducialOverlaps(true);
                if (bUseProjective)
                    _aligner.UseProjectiveTransform(true);
                if (!_aligner.ChangeProduction(_mosaicSetProcessing, _processingPanel))
                {
                    throw new ApplicationException("Aligner failed to change production ");
                }
            }
            catch (Exception except)
            {
                Output("Error Changing Production: " + except.Message);
                Terminate();
                return;
            }

            bool bDone = false;

            while (!bDone)
            {
                numAcqsComplete = 0;

                _aligner.ResetForNextPanel();
                _mosaicSetSim.ClearAllImages();
                _mosaicSetIllum.ClearAllImages();
                _mosaicSetProcessing.ClearAllImages();
                if (!GatherImages())
                {
                    Output("Issue with StartAcquisition");
                    bDone = true;
                }
                else
                {
                    Output("Waiting for Images...");
                    mDoneEvent.WaitOne();
                }

                // Verify that mosaic is filled in...
                if (!_mosaicSetSim.HasAllImages())
                    Output("The mosaic does not contain all images!");
                else
                {
                    _cycleCount++;
                    CreateIllumImages();
                    RunStitch();

                    //_mosaicSetProcessing.SaveAllStitchedImagesToDirectory("C:\\Temp\\");
                    // for fiducials(used for stitch) location check
                    //_aligner.Save3ChannelImage("c:\\Temp\\StitchFidLocation.bmp",
                    //                          _mosaicSetProcessing.GetLayer(0).GetStitchedBuffer(),
                    //                          _mosaicSetProcessing.GetLayer(1).GetStitchedBuffer(),
                    //                          _processingPanel.GetCADBuffer(),
                    //                          _processingPanel.GetNumPixelsInY(), _processingPanel.GetNumPixelsInX());

                    if (bSaveStitchedResultsImage)
                        _aligner.Save3ChannelImage("c:\\Temp\\FidCompareAfterCycle" + _cycleCount + ".bmp",
                                               _mosaicSetProcessing.GetLayer(0).GetStitchedBuffer(), _processingPanel.GetNumPixelsInY(),
                                               _mosaicSetProcessing.GetLayer(1).GetStitchedBuffer(), _processingPanel.GetNumPixelsInY(),
                                               _fidPanel.GetCADBuffer(), _processingPanel.GetNumPixelsInY(),
                                               _fidPanel.GetNumPixelsInY(), _fidPanel.GetNumPixelsInX());
                    if (_cycleCount == 1)
                    {
                        writer.WriteLine("Units: Microns");
                        //outline is the output file column names
                        writer.WriteLine(headerLine);
                    }

                    if (_fidPanel != null)
                        RunFiducialCompare(_mosaicSetProcessing.GetLayer(0).GetStitchedBuffer(), _fidPanel.GetNumPixelsInY(), writer, fidChecker);
                }
                if (_cycleCount >= numberToRun)
                    bDone = true;
                else
                    mDoneEvent.Reset();
            }

            writer.WriteLine(" Fid#, XOffset Mean, YOffset Mean,XOffset Stdev, YOffset Stdev, Absolute XOffset Mean, Absolute YOffset Mean, Absolute XOffset Stdev, Absolute YOffset Stdev, Number of cycle ");
            for (int i = 0; i < ifidsNum; i++)
            {
                if (_icycleCount[i] == 0)   // If no fiducial is found
                {
                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", i,
                            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", _icycleCount[i]));
                }
                else
                {
                    double dMeanX = _dXDiffSum[i] / _icycleCount[i];
                    double dMeanY = _dYDiffSum[i] / _icycleCount[i];
                    double dSdvX = Math.Sqrt(_dXDiffSqrSum[i] / _icycleCount[i] - dMeanX * dMeanX);
                    double dSdvY = Math.Sqrt(_dYDiffSqrSum[i] / _icycleCount[i] - dMeanY * dMeanY);
                    double dAbsMeanX = _dXAbsDiffSum[i] / _icycleCount[i];
                    double dAbsMeanY = _dYAbsDiffSum[i] / _icycleCount[i];
                    double dAbsSdvX = Math.Sqrt(_dXDiffSqrSum[i] / _icycleCount[i] - dAbsMeanX * dAbsMeanX);
                    double dAbsSdvY = Math.Sqrt(_dYDiffSqrSum[i] / _icycleCount[i] - dAbsMeanY * dAbsMeanY);
                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", i,
                        dMeanX, dMeanY, dSdvX, dSdvY,
                        dAbsMeanX, dAbsMeanY, dAbsSdvX, dAbsSdvY, 
                        _icycleCount[i]));
                }
                
            }
            _dXRMS = Math.Sqrt(_dXDiffSqrSumTol/(_cycleCount*_fidPanel.NumberOfFiducials));
            _dYRMS = Math.Sqrt(_dYDiffSqrSumTol/(_cycleCount*_fidPanel.NumberOfFiducials));
            writer.WriteLine(string.Format("MagicNumber: {0}, Average Offset: {1}, Xoffset RMS:{2}, Yoffset RMS:{3}", _allPanelFidDifference, _allPanelFidDifference / (_cycleCount * _fidPanel.NumberOfFiducials), _dXRMS, _dYRMS));

            if (File.Exists(lastOutputTextPath))
            {
                string[] lines = File.ReadAllLines(lastOutputTextPath);
                string lastLine = lines[lines.Length - 1];
                string[] parts = lastLine.Split(':');
                double lastMagic = 0;
                bool bGood = false;
                if (parts.Length > 1 && double.TryParse(parts[1], out lastMagic))
                {
                    // Check that we are at least as good as last time (to the nearest micron)
                    if (Math.Round(_allPanelFidDifference) <= Math.Round(lastMagic))
                        bGood = true;
                }

                Console.WriteLine("Are we as good as last time: " + (bGood?"Yes!":"No!"));

                if (Directory.Exists(unitTestFolder))
                {
                    string file = Path.Combine(unitTestFolder + Path.GetFileNameWithoutExtension(lastOutputTextPath)) + ".xml";
                    NUnitXmlWriter.WriteResult(file, "CyberStitchFidTester", "MagicNumber", bGood);
                }
            }

            Output("Processing Complete");
            Terminate();
            ManagedCoreAPI.TerminateAPI();
        }

        private static void Terminate()
        {
            if (writer != null)
                writer.Close();

            if(logger != null)
                logger.Kill();
        }

        private static void ShowHelp()
        {
            logger.AddObjectToThreadQueue("CyberStitchFIDTester Command line Options");
            logger.AddObjectToThreadQueue("*****************************************");
            logger.AddObjectToThreadQueue("-b // if bayer pattern");
            logger.AddObjectToThreadQueue("-f <FidTestPanel.xml>");
            logger.AddObjectToThreadQueue("-i <imagePath> instead of running cyberstitch");
            logger.AddObjectToThreadQueue("-l <lastResultsDirectory>");
            logger.AddObjectToThreadQueue("-h Show Help");
            logger.AddObjectToThreadQueue("-l <lastOutput.txt>");
            logger.AddObjectToThreadQueue("-n <#> // Number of panels - defaults to 1");
            logger.AddObjectToThreadQueue("-o <output.txt>");
            logger.AddObjectToThreadQueue("-p <panelfile.xml>");
            logger.AddObjectToThreadQueue("-r // Save stitched results image (c:\\temp\\*.bmp)");
            logger.AddObjectToThreadQueue("-s <SimScenario.xml>");
            logger.AddObjectToThreadQueue("-u <UnitTestFolder>");
            logger.AddObjectToThreadQueue("-w // if projective transform is desired");
            logger.AddObjectToThreadQueue("-----------------------------------------");
        }

        private static void RunFiducialCompare (IntPtr data, int stride, StreamWriter writer, ManagedFeatureLocationCheck fidChecker)
        {
            int iFidNums = _fidPanel.NumberOfFiducials;

            // Cad_x, cad_y, Loc_x, Loc_y, CorrScore, Ambig 
            int iItems = 6;
            double[] dResults = new double[iFidNums*iItems];
            double xDifference = 0;
            double yDifference = 0;
            double fidDifference = 0;
            string sNofid = "N/A";
            //convert meters to microns
            int iUnitCoverter = 1000000;
            // Find fiducial on the board
            fidChecker.CheckFeatureLocation(data, stride, dResults);

            for (int i = 0; i < iFidNums; i++)
            {
                if (dResults[i * iItems + 4] == 0 || dResults[i * iItems + 5] == 1) // Fiducial not found
                {
                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7}",
                        _cycleCount, i,
                        dResults[i * iItems] * iUnitCoverter, dResults[i * iItems + 1] * iUnitCoverter,
                        sNofid, sNofid,
                        dResults[i * iItems + 4], dResults[i * iItems + 5]));
                }
                else
                {
                    _icycleCount[i]++;
                    xDifference = (dResults[i * iItems] - dResults[i * iItems + 2]) * iUnitCoverter;
                    yDifference = (dResults[i * iItems + 1] - dResults[i * iItems + 3]) * iUnitCoverter;
                    fidDifference += Math.Sqrt(xDifference * xDifference + yDifference * yDifference);
                    
                    _dXDiffSum[i] += xDifference;
                    _dYDiffSum[i] += yDifference;
                    _dXAbsDiffSum[i] += Math.Abs(xDifference);
                    _dYAbsDiffSum[i] += Math.Abs(yDifference);
                    _dXDiffSqrSum[i] += Math.Pow(xDifference, 2);
                    _dYDiffSqrSum[i] += Math.Pow(yDifference, 2);

                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7}", 
                        _cycleCount, i,
                        dResults[i * iItems] * iUnitCoverter, dResults[i * iItems + 1] * iUnitCoverter, 
                        xDifference, yDifference,
                        dResults[i * iItems + 4], dResults[i * iItems + 5]));
                    _dXDiffSqrSumTol += _dXDiffSqrSum[i];
                    _dYDiffSqrSumTol += _dYDiffSqrSum[i];
                }
            }

            writer.WriteLine("Total Difference for this Panel: " + fidDifference);
            _allPanelFidDifference += fidDifference;
        }

        private static void RunStitch()
        {
            // Because _mosaicSetProcessing is hooked up to the aligner, things will automatically run.
            for (uint i = 0; i < _mosaicSetIllum.GetNumMosaicLayers(); i++)
            {
                ManagedMosaicLayer pLayer = _mosaicSetIllum.GetLayer(i);

                for (uint j = 0; j < pLayer.GetNumberOfCameras(); j++)
                {
                    for (uint k = 0; k < pLayer.GetNumberOfTriggers(); k++)
                    {
                        _mosaicSetProcessing.AddImage(pLayer.GetTile(j, k).GetImageBuffer(), i, j, k);
                    }
                }
            }
        }

        private static void CreateIllumImages()
        {
            if (_bBayerPattern)
            {
                for (uint i = 0; i < _mosaicSetSim.GetNumMosaicLayers(); i++)
                {
                    ManagedMosaicLayer pLayer = _mosaicSetSim.GetLayer(i);

                    for (uint j = 0; j < pLayer.GetNumberOfCameras(); j++)
                    {
                        for (uint k = 0; k < pLayer.GetNumberOfTriggers(); k++)
                        {
                            // Convert Bayer to luminance
                            pLayer.GetTile(j, k).Bayer2Lum(_iBayerType);

                            _mosaicSetIllum.AddImage(pLayer.GetTile(j, k).GetImageBuffer(), i, j, k);
                        }
                    }
                }
            }
            else
            {
                _mosaicSetIllum.CopyBuffers(_mosaicSetSim); 
            }
        }

        private static bool GatherImages()
        {
            if (!_bSimulating)
            {
                for (int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
                {
                    ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                    if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                        return false;
                }
            }
            else
            {   // launch device one by one in simulation case
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
                if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                    return false;
            }
            return true;
        }

        private static bool InitializeSimCoreAPI(string simulationFile)
        {
            //bool bSimulating = false;
            if (!string.IsNullOrEmpty(simulationFile) && File.Exists(simulationFile))
                _bSimulating = true;

            if (_bSimulating)
            {
                Output("Running with Simulation File: " + simulationFile);
                ManagedCoreAPI.SetSimulationFile(simulationFile);
            }

            ManagedSIMDevice.OnFrameDone += OnFrameDone;
            ManagedSIMDevice.OnAcquisitionDone += OnAcquisitionDone;

            if (ManagedCoreAPI.InitializeAPI() != 0)
            {

                Output("Could not initialize CoreAPI!");
                logger.Kill();
                return false;
            }

            if (ManagedCoreAPI.NumberOfDevices() <= 0)
            {
                Output("There are no SIM Devices attached!");
                logger.Kill();
                return false;
            }

            if (!_bSimulating)
            {
                for (int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
                {
                    ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                    int bufferCount = 128;// (triggerCount + 1) * GetNumberOfEnabledCameras(0) * 2;
                    int desiredCount = bufferCount;
                    d.AllocateFrameBuffers(ref bufferCount);

                    if (desiredCount != bufferCount)
                    {
                        Output("Could not allocate all buffers!  Desired = " + desiredCount + " Actual = " + bufferCount);
                        logger.Kill();
                        return false;
                    }

                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_processingPanel.PanelSizeX, _processingPanel.PanelSizeY, 0, .004);
                    if (cs1 == null)
                    {
                        Output("Could not create capture spec.");
                        return false;
                    }
                }
            }
            return true;
        }

        private static CPanel LoadProductionFile(string panelFile, double pixelSize)
        {
            CPanel panel = null;
            if (!string.IsNullOrEmpty(panelFile))
            {
                try
                {
                    if (panelFile.EndsWith(".srf", StringComparison.CurrentCultureIgnoreCase))
                    {
                        panel = SRFToPanel.parseSRF(panelFile, pixelSize, pixelSize);
                        if (panel == null)
                            throw new ApplicationException("Could not parse the SRF panel file");
                    }
                    else if (panelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
                    {
                        panel = XmlToPanel.CSIMPanelXmlToCPanel(panelFile, pixelSize, pixelSize);
                        if (panel == null)
                            throw new ApplicationException("Could not convert xml panel file");
                    }

                    return panel;
                }
                catch (Exception except)
                {
                    Output("Exception reading Panel file: " + except.Message);
                    logger.Kill();
                }
            }
            return panel;
        }

        private static void OnLogEntryFromClient(MLOGTYPE logtype, string message)
        {
            Console.WriteLine(logtype + " " + message);
            Output(logtype + " " + message);
        }

        /// <summary>
        /// Given a SIM setup and a mosaic for stitching, setup the stich...
        /// </summary>
        private static void SetupMosaic(bool bOwnBuffers, bool bMaskForDiffDevices)
        {
            if (ManagedCoreAPI.NumberOfDevices() <= 0)
            {
                Output("No Device Defined");
                return;
            }
            _mosaicSetSim = new ManagedMosaicSet(_processingPanel.PanelSizeX, _processingPanel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters, bOwnBuffers, false, 0); // not bayer pattern
            _mosaicSetIllum = new ManagedMosaicSet(_processingPanel.PanelSizeX, _processingPanel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters, false, false, 0);
            _mosaicSetProcessing = new ManagedMosaicSet(_processingPanel.PanelSizeX, _processingPanel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters, false, false, 0);
            _mosaicSetSim.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSetSim.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);
            SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSetIllum, bMaskForDiffDevices);
            SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSetProcessing, bMaskForDiffDevices);
            SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSetSim, bMaskForDiffDevices);
        }

        private static void OnLogEntryFromMosaic(MLOGTYPE logtype, string message)
        {
            Output(logtype + " From Mosaic: " + message);
        }

        private static void OnAcquisitionDone(int device, int status, int count)
        {
            Output("OnAcquisitionDone Called!");
            numAcqsComplete++;
            // launch next device in simulation case
            if (_bSimulating && numAcqsComplete < ManagedCoreAPI.NumberOfDevices())
            {
                Thread.Sleep(10000);
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(numAcqsComplete);
                if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                    return;
            }
            if (ManagedCoreAPI.NumberOfDevices() == numAcqsComplete)
                mDoneEvent.Set();
        }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
            // Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
             //    pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));
            _iBufCount++; // for debug
            uint layer = (uint)(pframe.DeviceIndex() * ManagedCoreAPI.GetDevice(0).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex());
            _mosaicSetSim.AddImage(pframe.BufferPtr(), layer, (uint)pframe.CameraIndex(),
                                (uint)pframe.TriggerIndex());
        }

        private static void Output(string str)
        {
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }
    }
}
