using System;
using System.IO;
using System.Reflection;
using System.Threading;
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
    /// </summary>
    class Program
    {
        private const double cPixelSizeInMeters = 1.70e-5;
        private static ManagedMosaicSet _mosaicSetSim = null;
        private static ManagedMosaicSet _mosaicSetIllum = null;
        private static ManagedMosaicSet _mosaicSetProcessing = null;
        private static CPanel _processingPanel = new CPanel(0, 0, cPixelSizeInMeters, cPixelSizeInMeters);
        private static CPanel _fidPanel = new CPanel(0, 0, cPixelSizeInMeters, cPixelSizeInMeters);
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread logger = new LoggingThread(null);
        private static int _cycleCount = 0;
        private static double allPanelFidDifference = 0.0;

        // For debug
        private static int _iBufCount = 0;
        private static bool _bSimulating = false;
        private static bool _bBayerPattern = false;
        private static int _iBayerType = 1; // GBRG

        private static ManagedFeatureLocationCheck fidChecker = null;

        // For output analysis
        private static double[] _dmeanXDiff;
        private static double[] _dmeanYDiff;
        private static double[] _dXDiffTot;
        private static double[] _dYDiffTot;
        private static double[] _dXDiffSqrTol;
        private static double[] _dYDiffSqrTol;
        private static double[] _dXDiffStdev;
        private static double[] _dYDiffStdev;
        private static int[] _icycleCount;

        /// <summary>
        /// This works similar to CyberStitchTester.  The differences:
        /// 1)  All images are obtained prior to running through CyberStitch.
        /// 2)  Color images are pre-converted to illum images for faster processing.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Start the logger
            logger.Start("Logger", @"c:\\", "CyberStitch.log", true, -1);
            logger.AddObjectToThreadQueue("CyberStitchFidTester Version: " + Assembly.GetExecutingAssembly().GetName().Version);

            // Gather input data.
            string simulationFile = "";
            string panelFile = "";
            string fidPanelFile = "";
            bool bUseProjective = false;
            bool bSaveStitchedResultsImage = false;
            int numberToRun = 1;
            string unitTestFolder="";

            //output csv file shows the comparison results
            string outputTextPath = @".\fidsCompareResults.csv";
            string lastOutputTextPath = @".\fidsCompareResults.csv";
            StreamWriter writer = null;
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-b")
                    _bBayerPattern = true;
                else if (args[i] == "-f" && i < args.Length - 1)
                    fidPanelFile = args[i + 1];
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
                    logger.Kill();
                    return;
                }
            }

            _processingPanel = LoadProductionFile(panelFile);
            if (_processingPanel == null)
            {
                logger.Kill();
                Console.WriteLine("Could not load Panel File: " + panelFile);
                return;
            }

            if (File.Exists(fidPanelFile))
            {
                _fidPanel = LoadProductionFile(fidPanelFile);
                if (_fidPanel == null)
                {
                    logger.Kill();
                    Console.WriteLine("Could not load Fid Test File: " + fidPanelFile);
                    return;
                }

                fidChecker = new ManagedFeatureLocationCheck(_fidPanel);
            }


            // Initialize the SIM CoreAPI
            if (!InitializeSimCoreAPI(simulationFile))
            {
                logger.Kill();
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
                _aligner.LogFiducialOverlaps(true);
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
                logger.Kill();
                return;
            }

            bool bDone = false;

            int ifidsNum = _fidPanel.NumberOfFiducials;
            _dXDiffTot = new double[ifidsNum];
            _dYDiffTot = new double[ifidsNum];
            _dmeanXDiff = new double[ifidsNum];
            _dmeanYDiff = new double[ifidsNum];
            _dXDiffSqrTol = new double[ifidsNum];
            _dYDiffSqrTol = new double[ifidsNum];
            _dXDiffStdev = new double[ifidsNum];
            _dYDiffStdev = new double[ifidsNum];
            _icycleCount = new int[ifidsNum];

            for (int i = 0; i < ifidsNum; i++)
            {
                _dmeanXDiff[i] = 0;
                _dmeanYDiff[i] = 0;
                _dXDiffStdev[i] = 0;
                _dYDiffStdev[i] = 0;
                _icycleCount[i] = 0;
                _dXDiffSqrTol[i] = 0;
                _dYDiffSqrTol[i] = 0;
            }

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

                    if (bSaveStitchedResultsImage)
                        _aligner.Save3ChannelImage("c:\\Temp\\FidCompareAfterCycle" + _cycleCount + ".bmp",
                                               _mosaicSetProcessing.GetLayer(0).GetStitchedBuffer(),
                                               _mosaicSetProcessing.GetLayer(1).GetStitchedBuffer(),
                                               _fidPanel.GetCADBuffer(),
                                               _fidPanel.GetNumPixelsInY(), _fidPanel.GetNumPixelsInX());
                    if (_cycleCount == 1)
                    {
                        writer = new StreamWriter(outputTextPath);
                        writer.WriteLine("Units: Microns");
                        //outline is the output file column names
                        string outLine = "Panel#, Fid#, X, Y ,XOffset, YOffset, CorrScore, Ambig";
                        writer.WriteLine(outLine);
                    }

                    if (fidChecker != null)
                    RunFiducialCompare(_mosaicSetProcessing.GetLayer(0).GetStitchedBuffer(), _fidPanel.NumberOfFiducials, writer);
                }
                if (_cycleCount >= numberToRun)
                    bDone = true;
                else
                    mDoneEvent.Reset();
            }

            writer.WriteLine(" Fid#,XOffset Mean, YOffset Mean,XOffset Stdev, YOffset Stdev,Number of cycle ");
            for (int i = 0; i < ifidsNum; i++)
            {
                writer.WriteLine(string.Format("{0},{1},{2},{3},{4},{5}", i, _dmeanXDiff[i], _dmeanYDiff[i], _dXDiffStdev[i], _dYDiffStdev[i],_icycleCount[i]));
            }
            writer.WriteLine(string.Format("MagicNumber: {0}", allPanelFidDifference));

            if (File.Exists(lastOutputTextPath))
            {
                string[] lines = File.ReadAllLines(lastOutputTextPath);
                string lastLine = lines[lines.Length - 1];
                string[] parts = lastLine.Split(':');
                double lastMagic = 0;
                bool bGood = false;
                if (parts.Length > 1 && double.TryParse(parts[1], out lastMagic))
                {
                    if (allPanelFidDifference <= lastMagic)
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
            if (writer != null)
                writer.Close();
            logger.Kill();
            ManagedCoreAPI.TerminateAPI();
        }
        private static void ShowHelp()
        {
            logger.AddObjectToThreadQueue("CyberStitchFIDTester Command line Options");
            logger.AddObjectToThreadQueue("*****************************************");
            logger.AddObjectToThreadQueue("-b // if bayer pattern");
            logger.AddObjectToThreadQueue("-f <FidTestPanel.xml>");
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

        private static void RunFiducialCompare (IntPtr data, int iFidNums, StreamWriter writer)
            {
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
            fidChecker.CheckFeatureLocation(data, dResults);
            for (int i = 0; i < iFidNums*iItems - 5; i++)
            {
                if (dResults[i + 4] == 0 || dResults[i + 5] == 1)
                {
                    writer.WriteLine(string.Format("{0},{1},{2},{3},{4},{5},{6},{7}", _cycleCount, i/6,
                                                   dResults[i]*iUnitCoverter, dResults[i + 1]*iUnitCoverter, sNofid,
                                                   sNofid, dResults[i + 4], dResults[i + 5]));
                    i += 5;
                }
                else
                {
                    _icycleCount[i/6]++;
                    xDifference = (dResults[i] - dResults[i + 2])*iUnitCoverter;
                    yDifference = (dResults[i + 1] - dResults[i + 3])*iUnitCoverter;
                    fidDifference += Math.Sqrt(xDifference*xDifference + yDifference*yDifference);
                    _dXDiffTot[i/6] += Math.Abs(xDifference);
                    _dmeanXDiff[i/6] = _dXDiffTot[i/6]/_icycleCount[i/6];
                    _dYDiffTot[i / 6] += Math.Abs(yDifference);
                    _dmeanYDiff[i/6] = _dYDiffTot[i/6]/_icycleCount[i/6];
                    _dXDiffSqrTol[i/6] += Math.Pow(xDifference, 2);
                    _dYDiffSqrTol[i/6] += Math.Pow(yDifference, 2);
                    _dXDiffStdev[i/6] = Math.Sqrt(_dXDiffSqrTol[i/6]/_icycleCount[i/6] - Math.Pow(_dmeanXDiff[i/6], 2));
                    _dYDiffStdev[i/6] = Math.Sqrt(_dYDiffSqrTol[i/6]/_icycleCount[i/6] - Math.Pow(_dmeanYDiff[i/6], 2));
                    writer.WriteLine(string.Format("{0},{1},{2},{3},{4},{5},{6},{7}", _cycleCount, i/6,
                                                   dResults[i]*iUnitCoverter, dResults[i + 1]*iUnitCoverter, xDifference,
                                                   yDifference, dResults[i + 4], dResults[i + 5]));
                    i += 5;
                }
            }

            writer.WriteLine("Total Difference for this Panel: " + fidDifference);
            allPanelFidDifference += fidDifference;
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

        private static CPanel LoadProductionFile(string panelFile)
        {
            CPanel panel = null;
            if (!string.IsNullOrEmpty(panelFile))
            {
                try
                {
                    if (panelFile.EndsWith(".srf", StringComparison.CurrentCultureIgnoreCase))
                    {
                        panel = SRFToPanel.parseSRF(panelFile, cPixelSizeInMeters, cPixelSizeInMeters);
                        if (panel == null)
                            throw new ApplicationException("Could not parse the SRF panel file");
                    }
                    else if (panelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
                    {
                        panel = XmlToPanel.CSIMPanelXmlToCPanel(panelFile, cPixelSizeInMeters, cPixelSizeInMeters);
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
             Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
                 pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));
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
