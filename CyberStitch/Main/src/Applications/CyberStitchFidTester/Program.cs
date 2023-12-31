﻿using System;
using System.Collections.Generic;
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
    /// The program is to check the accuracy of CyberStitch or load stitched images  
    /// For cyberstitch
    /// 1)  Input a SIM Simulation set, a panel file with a small number of fids and a test file with a large number of features.
    /// For stitched image
    /// 1)  Input stitched images, a panel file with a small number of fids (optional) and a test file with a large number of features.
    /// 2)  Output a text file that gives you an indication of how far off the fiducials are in the stitched image 
    /// compared to nominal/cad locations
    /// 3)  Extra Credit:  This runs as part of the nightly build.  Each night, we can compare a magic number (the average distance offset)
    /// to the last magic number.  If the Magic Number is increasing, we should be nervous because something we changed is causing problems.
    /// NOTE:  Using stitched image causes some headaches because the image
    /// file may be built with other tools (2dSPI for instance) and therefore may have a different pixel size and stride.  This muddied the 
    /// water, but it should be handled correctly.
    /// </summary>
    class ProgramfidChecker
    {
        // Control parameter
        private static double _dPixelSizeInMeters = 1.70e-5; // This will be adjusted if using a Simulated or real device
        private static uint _iInputImageColumns = 2592; // This will be adjusted if using a Simulated or real device
        private static uint _iInputImageRows = 1944; // This will be adjusted if using a Simulated or real device
        private static bool _bDetectPanelEdge = false;
        private static int _iLayerIndex4Edge = 0;
        private static bool _bRtoL = false; // right to left conveyor direction indicator
        private static bool _bFRR = false; // fixed rear rail indicator
        private static bool _bSkipDemosaic = true; // true: skip demosaic for bayer image
        private static bool _bBayerPattern = true;
        private static int _iBayerType = 1; // GBRG
        private static string _simulationFile = "";
        private static string _alignmentPanelFile = "";
        private static string _featureFile = "";
        private static bool _bSimulating = false;
        private static bool _bUseProjective = true;
        private static bool _bUseCameraModel = true;
        private static bool _bUseIterativeCameraModel = false;
        private static bool _bSaveStitchedResultsImage = false;
        private static bool _bUseTwoPassStitch = false;
        private static int _numberToRun = 1;
        private static bool _bMaskForDiffDevices = false;
        

        // For stitched image as input
        private static string _stitchedImagePathPattern = "";
        private static int _iPanelOffsetInCols = 0; // Panel's top left corner in stitched image
        private static int _iPanelOffsetInRows = 0; 
        
        //output csv file shows the comparison results
        private static string _unitTestFolder = "";
        private static string _outputTextPath = @".\StitchAccuracyResults.csv";
        private static string _lastOutputTextPath = @".\LastStitchAccuracyResults.csv";
        
        // Internal variable
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static ManagedMosaicSet _mosaicSet = null;
        private static CPanel _alignmentPanel = null;
        private static CPanel _alignmentPanel_noFid = null;
        private static CPanel _featurePanel = null;
        private static ManagedFeatureLocationCheck _featureChecker = null;
        private static ManagedImageFidAligner _imageFidAligner = null;
        private static LoggingThread _logger = new LoggingThread(null);
        private readonly static ManualResetEvent _mAlignedEvent = new ManualResetEvent(false);
        private static StreamWriter _writer = null;
        private static StreamWriter _finalCompWriter= null;
        private static int _numAcqsComplete = 0;
        private static int _cycleCount = 0;
        private static bool _bStitchedImageOnly = false;
        private static bool _bNoFiducial = false;
        
        // For output analysis
        private const string headerLine = "Panel#, Fid#, X, Y ,XOffset, YOffset, CorrScore, Ambig";
        private static double[] _dXDiffSum;
        private static double[] _dYDiffSum;
        private static double[] _dXAbsDiffSum;
        private static double[] _dYAbsDiffSum;
        private static double[] _dXDiffSqrSum;
        private static double[] _dYDiffSqrSum;        
        private static int[] _icycleCount;
        private static int _iTotalCount = 0;        
        private static double _allPanelFidDifference = 0.0;
        private static double _dXDiffSqrSumTol = 0.0;//used for the total Xoffset square sum
        private static double _dYDiffSqrSumTol = 0.0;
        private static double _dXRMS = 0.0;//used for the xoffset RMS
        private static double _dYRMS = 0.0;            
       
        // For time stamp
        private static DateTime _dtStartTime;
        private static DateTime _dtEndTime;
        private static double _tsRunTime = 0; 
        private static double _tsTotalRunTime = 0;
        private static uint _numThreads = 8;

        private static bool _bUseCoreAPI = true;
       
        /// <summary>
        /// This works similar to CyberStitchTester.  The differences:
        /// 1)  All images are obtained prior to running through CyberStitch.
        /// 2)  Color images are pre-converted to illum images for faster processing.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {

            // Start the logger (showHelp() requires logger to be started)
            _logger.Start("Logger", @"c:\\", "CyberStitch.log", true, -1);
            _logger.AddObjectToThreadQueue("CyberStitchFidTester Version: " + Assembly.GetExecutingAssembly().GetName().Version);
            
            
            // Control parameter inputs
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-nb")
                    _bBayerPattern = false;
                else if (args[i] == "-m")
                    _bMaskForDiffDevices = true;
                else if (args[i] == "-r")
                    _bSaveStitchedResultsImage = true;
                else if (args[i] == "-nw")
                    _bUseProjective = false;
                else if (args[i] == "-fov")
                    _bUseCameraModel = false;
                else if (args[i] == "-iter")
                {
                    _bUseCameraModel = false;
                    _bUseIterativeCameraModel = true;
                }
                else if (args[i] == "-rtol")
                    _bRtoL = true;
                else if (args[i] == "-frr")
                    _bFRR = true;
                else if (args[i] == "-de")
                    _bDetectPanelEdge = true;
                else if (args[i] == "-skipD")
                    _bSkipDemosaic = true;
                else if (args[i] == "-twopass")
                    _bUseTwoPassStitch = true;
                else if (args[i] == "-nf")
                    _bNoFiducial = true;
                else if (args[i] == "-s" && i < args.Length - 1)
                    _simulationFile = args[i + 1];
                else if (args[i] == "-u" && i < args.Length - 1)
                    _unitTestFolder = args[i + 1];
                else if (args[i] == "-n" && i < args.Length - 1)
                    _numberToRun = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-o" && i < args.Length - 1)
                    _outputTextPath = args[i + 1];
                else if (args[i] == "-p" && i < args.Length - 1)
                    _alignmentPanelFile = args[i + 1];
                else if (args[i] == "-f" && i < args.Length - 1)
                    _featureFile = args[i + 1];
                else if (args[i] == "-i" && i < args.Length - 1)
                    _stitchedImagePathPattern = args[i + 1];
                else if (args[i] == "-l" && i < args.Length - 1)
                    _lastOutputTextPath = args[i + 1];
                else if (args[i] == "-le" && i < args.Length - 1)
                    _iLayerIndex4Edge = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-pixsize" && i < args.Length - 1)
                    _dPixelSizeInMeters = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-imgcols" && i < args.Length - 1)
                    _iInputImageColumns = Convert.ToUInt32(args[i + 1]);
                else if (args[i] == "-imgrows" && i < args.Length - 1)
                    _iInputImageRows = Convert.ToUInt32(args[i + 1]);
                else if (args[i] == "-t" && i < args.Length - 1)
                    _numThreads = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-xoffset" && i < args.Length - 1)
                    _iPanelOffsetInCols = Convert.ToInt32(args[i + 1]);
                else if (args[i] == "-yoffset" && i < args.Length - 1)
                    _iPanelOffsetInRows = Convert.ToInt32(args[i + 1]);
                else if (args[i] == "-h" && i < args.Length - 1)
                {
                    ShowHelp();
                    return;
                }
            }

            if (_simulationFile.EndsWith(".csv", StringComparison.CurrentCultureIgnoreCase))
            {
                _bUseCoreAPI = false;

                string sFile = Path.GetDirectoryName(_simulationFile) + "\\Cycle0\\Cam0_Trig0.bmp";
                if (!File.Exists(sFile))
                {
                    Output("No image file exists!");
                    return;
                }
                Bitmap fov = new Bitmap(sFile);
                _iInputImageColumns = (uint)fov.Size.Width;
                _iInputImageRows = (uint)fov.Size.Height;
                // SIM120
                if (_iInputImageColumns == 3664 && _iInputImageRows == 2748)
                    _dPixelSizeInMeters = 1.2e-5;
                // SIM 110
                else if (_iInputImageColumns == 2592 && _iInputImageRows == 1944)
                    _dPixelSizeInMeters = 1.7e-5;
                else
                {
                    Output("Invalid image file !");
                    return;
                }
            }

            // Panel images are from disc or from stitch
            string[] stitchedImagePath = ExpandFilePaths(_stitchedImagePathPattern);
            if (stitchedImagePath.Length > 0)
            {
                _bStitchedImageOnly = true;
            }

            if ((_bStitchedImageOnly || !_bUseCoreAPI) && _dPixelSizeInMeters < 0)
            {
                _logger.AddObjectToThreadQueue("If you're not using CoreAPI, or using stitched images, you must specify pixel size.");
                Terminate(false);
                return;
            }

            if (!_bStitchedImageOnly && _bUseCoreAPI)
            {
                if (!InitializeSimCoreAPI(_simulationFile))
                {
                    Console.WriteLine("Could not initialize Core API");
                    Terminate(false);
                    return;
                }
            }

            // Load offset Fiducial file
            if (!LoadFeatureFile())
            {
                Output("Cannot load offset fiducial file");
                Terminate(false);
                return;
            }

            // Load panel, _alignmentPanel default is null 
            if (File.Exists(_alignmentPanelFile))
                _alignmentPanel = LoadPanelDescription(_alignmentPanelFile, _dPixelSizeInMeters);
            else
            {
                Output("Cannot load alignment fiducial file");
                Terminate(false);
                return;
            }

            // If panel need to be stitched without fiducials
            if (!_bStitchedImageOnly && _bNoFiducial)
            {
                _alignmentPanel_noFid = LoadPanelDescription(_alignmentPanelFile, _dPixelSizeInMeters);
                _alignmentPanel_noFid.ClearFiducials();
            }

            // If stitched image need be adjusted by fiducial
            if ((_bStitchedImageOnly && _alignmentPanel != null) ||
                (!_bStitchedImageOnly && _bNoFiducial))
            {
                _imageFidAligner = new ManagedImageFidAligner(_alignmentPanel);
            }

            // If no stitched image and panel information is not available
            if(!_bStitchedImageOnly && _alignmentPanel == null)
            {
                Console.WriteLine("Could not load Panel File: " + _alignmentPanelFile);
                Terminate(false);
                return;
            }
  
            // Open output text file
            try
            {
                _writer = new StreamWriter(_outputTextPath);
                Output("The report file: " + _outputTextPath);
            }
            catch(Exception e)
            {
                Output("Cannot open outputfile"+e.Message);
                Terminate(false);
                return;
            }

            if (_bStitchedImageOnly) // Panel images are from disc
            { 
                _writer.WriteLine(headerLine);
                int cycleId = 0; // Image already loaded.
               
                while (cycleId < stitchedImagePath.Length)
                { 
                    // Load first image
                    Bitmap inputBmp = new Bitmap(stitchedImagePath[cycleId]);  // should be safe, as _bStitchedImageOnly == (stitchedImagePath.Length > 0)
                    if(inputBmp == null)
                        break;
                    
                    if(inputBmp.PixelFormat != System.Drawing.Imaging.PixelFormat.Format8bppIndexed)
                    {
                        Output("Image " + stitchedImagePath[cycleId] + " is not a grayscale one!");
                        Terminate(false);
                        return;
                    }
                    Console.WriteLine("Comparing fiducials on {0}...", stitchedImagePath[cycleId]);

                    // Adjust image based on fiducial results
                    CyberBitmapData cbd = new CyberBitmapData();
                    cbd.Lock(inputBmp);

                    IntPtr dataPoint = new IntPtr((Int64)cbd.Scan0 + _iPanelOffsetInRows*cbd.Stride + _iPanelOffsetInCols);
                    IntPtr morphedData = _imageFidAligner.MorphImage(dataPoint, cbd.Stride);

                    cbd.Unlock();
                        // It is safe to explicit release memory
                    inputBmp.Dispose();

                    // Calcaulate offsets
                    RunFiducialCompare(morphedData, _featurePanel.GetNumPixelsInY(), _writer);

                    // Save panel image
                    if (_bSaveStitchedResultsImage)
                    {
                        // record debug image
                        string imageFilename = "c:\\Temp\\FeatureCompareImage-" + cycleId + ".bmp";
                        _aligner.Save3ChannelImage(imageFilename,
                            morphedData, _featurePanel.GetNumPixelsInY(),
                            morphedData, _featurePanel.GetNumPixelsInY(),
                           _featurePanel.GetCADBuffer(), _featurePanel.GetNumPixelsInY(),
                            _featurePanel.GetNumPixelsInY(), _featurePanel.GetNumPixelsInX());
                    }
                
                    // Increment cycle and get bitmap, if available
                    cycleId++;

                    // Check terminal condition
                    if (cycleId == _numberToRun)
                        break;
                }
            }
            else  // Panel images are from stitch
            {
                if (_bUseCoreAPI)
                {
                    if (!SetupSIMCaptureSpecs())
                    {
                        Console.WriteLine("Error in setting up capture specs");
                        Terminate(true);
                        return;
                    }
                }

                // SetUp aligner
                try
                {
                    SetupAligner();
                }
                catch (Exception except)
                {
                    Output("Error SetupAligner: " + except.Message);
                    Terminate(true);
                    return;
                }

                // 
                try
                {
                    RunStitchAndRecordResults(_numberToRun);
                }
                catch (Exception except)
                {
                    Output("Error during acquire and stitch: " + except.Message);
                    Terminate(true);
                    return;
                }

            } // End Simulation Mode

            // Last attempt to force proper shutdown...
            try
            {
                WriteResults(_lastOutputTextPath, _outputTextPath, _unitTestFolder);
                Output("Processing Complete");
            }
            catch (Exception except)
            {
                Output("Error during write results: " + except.Message);
            }
            finally
            {
                Terminate(!_bStitchedImageOnly);           
            }
        }


        #region utilities
        
        private static void Terminate(bool bTerminateCore)
        {
            if (_writer != null)
                _writer.Close();

            if (_logger != null)
                _logger.Kill();

            _aligner.Dispose();       

            if(bTerminateCore)
                ManagedCoreAPI.TerminateAPI();
        }

        private static void OnLogEntryFromClient(MLOGTYPE logtype, string message)
        {
            Console.WriteLine(logtype + " " + message);
            Output(logtype + " " + message);
        }

        private static void OnLogEntryFromMosaic(MLOGTYPE logtype, string message)
        {
            Output(logtype + " From Mosaic: " + message);
        }

        private static void Output(string str)
        {
            _logger.AddObjectToThreadQueue(str);
            _logger.AddObjectToThreadQueue(null);
        }

        private static void ShowHelp()
        {
            _logger.AddObjectToThreadQueue("CyberStitchFIDTester Command line Options");
            _logger.AddObjectToThreadQueue("*****************************************");
            _logger.AddObjectToThreadQueue("-b // if bayer pattern");
            _logger.AddObjectToThreadQueue("-f <FidTestPanel.xml>");
            _logger.AddObjectToThreadQueue("-i <stitchedImagePath> instead of running cyberstitch");
            _logger.AddObjectToThreadQueue("-l <lastResultsDirectory>");
            _logger.AddObjectToThreadQueue("-h Show Help");
            _logger.AddObjectToThreadQueue("-l <lastOutput.txt>");
            _logger.AddObjectToThreadQueue("-n <#> // Number of panels - defaults to 1");
            _logger.AddObjectToThreadQueue("-o <output.txt>");
            _logger.AddObjectToThreadQueue("-p <panelfile.xml>");
            _logger.AddObjectToThreadQueue("-r // Save stitched results image (c:\\temp\\*.bmp)");
            _logger.AddObjectToThreadQueue("-s <SimScenario.xml>");
            _logger.AddObjectToThreadQueue("-u <UnitTestFolder>");
            _logger.AddObjectToThreadQueue("-w // if projective transform is desired");
            _logger.AddObjectToThreadQueue("-----------------------------------------");
        }

        private static string[] ExpandFilePaths(string filePathPattern)
        {
            List<string> fileList = new List<string>();

            if (filePathPattern.Length > 0)
            {
                string substitutedFilePathPattern = Environment.ExpandEnvironmentVariables(filePathPattern);

                string directory = Path.GetDirectoryName(substitutedFilePathPattern);
                if (string.IsNullOrEmpty(directory))
                    directory = ".";

                string filePattern = Path.GetFileName(substitutedFilePathPattern);
                if (string.IsNullOrEmpty(filePattern))
                    filePattern = "*";

                foreach (string filePath in Directory.GetFiles(directory, filePattern))
                    fileList.Add(filePath);
            }

            return fileList.ToArray();
        }

        #endregion

        #region load panel file
        private static CPanel LoadPanelDescription(string panelFile, double pixelSize)
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
                    else if (_alignmentPanelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
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
                    _logger.Kill();
                }
            }
            return panel;
        }

        private static bool LoadFeatureFile()
        {
            if (File.Exists(_featureFile))
            {
                double pixelSize = _dPixelSizeInMeters;

                _featurePanel = LoadPanelDescription(_featureFile, pixelSize);

                if (_featurePanel == null)
                {
                    Terminate(false);
                    Console.WriteLine("Could not load Fid Test File: " + _featureFile);
                    return false;
                }
                _featureChecker = new ManagedFeatureLocationCheck(_featurePanel);
                int ifidsNum = _featurePanel.NumberOfFiducials;
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
                Terminate(false);
                Console.WriteLine("Feature file does not exist: " + _featureFile);
                return false;
            }

            return (true);
        }
        #endregion

        #region coreApi and callbacks

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
                _logger.Kill();
                return false;
            }

            if (ManagedCoreAPI.NumberOfDevices() <= 0)
            {
                Output("There are no SIM Devices attached!");
                _logger.Kill();
                return false;
            }

            Output("ManagedCoreAPI # Devices: " + ManagedCoreAPI.NumberOfDevices());

            // Determine Pixel size on SIM.  Make sure they're all consistent.
            if (_dPixelSizeInMeters > 0)
            {
                Output("Warning, Overwriting user defined pixel size from SIM settings.");
            }
            // Determine Pixel size on SIM.  Make sure they're all consistent.
            _dPixelSizeInMeters = -1;
            for (int ix = 0; ix < ManagedCoreAPI.NumberOfDevices(); ix++)
            {
                ManagedSIMDevice device = ManagedCoreAPI.GetDevice(ix);

                if (_dPixelSizeInMeters < 0)
                {
                    _dPixelSizeInMeters = device.NominalPixelSizeX;
                }
                else if (Math.Abs(device.NominalPixelSizeX - _dPixelSizeInMeters) > 0.00001)
                {
                    Output("Pixel Sizes on SIM Device ID " + ix + " don't Match Device 0 " + device.NominalPixelSizeX + " " + _dPixelSizeInMeters);
                    return false;
                }
            }

            // Determine pixels on SIM.  Make sure they're all consistent.
            _iInputImageColumns = 0;
            _iInputImageRows = 0;
            for (int ix = 0; ix < ManagedCoreAPI.NumberOfDevices(); ix++)
            {
                ManagedSIMDevice device = ManagedCoreAPI.GetDevice(ix);
                ManagedSIMCamera camera = device.GetSIMCamera(device.FirstCameraEnabled);
                if (_iInputImageColumns == 0)
                {
                    _iInputImageColumns = (uint)camera.Columns();
                    _iInputImageRows = (uint)camera.Rows();
                }
                else
                {
                    if (_iInputImageColumns != (uint)camera.Columns() || _iInputImageRows != (uint)camera.Rows())
                    {
                        Output("Camera sizes are changing on SIM Device " + ix + " " + _iInputImageColumns + " " + _iInputImageRows + " " + camera.Columns() + " " + camera.Rows());
                        return false;
                    }
                }
            }

            return true;
        }

        private static bool SetupSIMCaptureSpecs() 
        {
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
                        _logger.Kill();
                        return false;
                    }
                    if (_bRtoL)
                    {
                        d.ConveyorRtoL = true;
                    }
                    if (_bFRR)
                    {
                        d.FixedRearRail = true;
                    }

                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_alignmentPanel.PanelSizeX, _alignmentPanel.PanelSizeY, 0, .004);
                    if (cs1 == null)
                    {
                        Output("Could not create capture spec.");
                        return false;
                    }
                }
            }

            return true;
        }

        private static void OnAcquisitionDone(int device, int status, int count)
        {
            Output("OnAcquisitionDone Called!");
            _numAcqsComplete++;
            // launch next device in simulation case
            if (_bSimulating && _numAcqsComplete < ManagedCoreAPI.NumberOfDevices())
            {
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(_numAcqsComplete);
                if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                    Output("Could not start acquisition for device " + device);
            }
        }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
            // Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
            //     pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));

            int device = pframe.DeviceIndex();
            int mosaic_row = SimMosaicTranslator.TranslateTrigger(pframe);
            int mosaic_column = pframe.CameraIndex() - ManagedCoreAPI.GetDevice(device).FirstCameraEnabled;

            uint layer = (uint)(device * ManagedCoreAPI.GetDevice(device).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex());

            _mosaicSet.AddRawImage(pframe.BufferPtr(), layer, (uint)mosaic_column, (uint)mosaic_row);
        }

        #endregion

        #region Setup Aligner

        /// <summary>
        /// Given a SIM setup and a mosaic for stitching, setup the stich...
        /// </summary>
        private static bool SetupMosaicSet()
        {
            if (_bUseCoreAPI)
            {
                if (ManagedCoreAPI.NumberOfDevices() <= 0)
                {
                    Output("No Device Defined");
                    return false;
                }
            }

            _mosaicSet = new ManagedMosaicSet(
                _alignmentPanel.PanelSizeX, _alignmentPanel.PanelSizeY,
                _iInputImageColumns, _iInputImageRows, _iInputImageColumns,
                _dPixelSizeInMeters, _dPixelSizeInMeters,
                true,
                _bBayerPattern, _iBayerType, _bSkipDemosaic);
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);

            //_mosaicSet.SetGaussianDemosaic(true);

            if (_bUseCoreAPI)
                SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSet, _bMaskForDiffDevices);
            else
                SimMosaicTranslator.InitializeMosaicFromNominalTrans(_mosaicSet, _simulationFile, _alignmentPanel.PanelSizeX);

            return true;
        }

        private static void SetupAligner()
        {
            if (!SetupMosaicSet())
            {
                throw new ApplicationException("Failed to setup mossaic set");
            }

            Output("Aligner ChangeProduction");
            // Setup Aligner...
            _aligner.OnLogEntry += OnLogEntryFromClient;
            _aligner.SetAllLogTypes(true);
            _aligner.OnAlignmentDone += OnAlignmentDone;
            _aligner.UseProjectiveTransform(_bUseProjective);

            //_aligner.LogOverlaps(true);
            //_aligner.LogFiducialOverlaps(true);
            //_aligner.LogTransformVectors(true);

            if (_bUseTwoPassStitch)
                _aligner.SetUseTwoPassStitch(true);
            if (_bUseCameraModel)
            {
                _aligner.UseCameraModelStitch(true);
                _aligner.UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
            }
            if (_bUseIterativeCameraModel)
            {
                _aligner.UseCameraModelIterativeStitch(true);
                _aligner.UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
            }

            // Set number of thread to be used in cyberstitch
            _aligner.NumThreads(_numThreads);

            // Always seperate acquistion, demosaic and alignment stages 
            _mosaicSet.SetSeperateProcessStages(true);

            // true: Skip demosaic for Bayer image
            if (_bBayerPattern)
                _aligner.SetSkipDemosaic(_bSkipDemosaic);

            // Must after InitializeSimCoreAPI() before ChangeProduction()
            if (_bUseCoreAPI)
            {
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
                _aligner.SetPanelEdgeDetection(_bDetectPanelEdge, _iLayerIndex4Edge, !d.ConveyorRtoL, !d.FixedRearRail);
            }
            else
                _aligner.SetPanelEdgeDetection(_bDetectPanelEdge, _iLayerIndex4Edge, true, true);

            // Add trigger to trigger overlaps for same layer
            //for (uint i = 0; i < _mosaicSet.GetNumMosaicLayers(); i++)
            //    _mosaicSet.GetCorrelationSet(i, i).SetTriggerToTrigger(true);

            CPanel tempPanel = _alignmentPanel;
                // If need stitch without fiducial
            if (_bNoFiducial)
                tempPanel = _alignmentPanel_noFid;
            if (!_aligner.ChangeProduction(_mosaicSet, tempPanel))
            {
                throw new ApplicationException("Aligner failed to change production");
            }
        }

        #endregion

        #region stitch
        private static void OnAlignmentDone(bool status)
        {
            Output("OnAlignmentDone Called!");
            _mAlignedEvent.Set();
        }

        private static void RunStitchAndRecordResults(int numberToRun)
        {
            bool bDone = false;
            while (!bDone)
            {
                _numAcqsComplete = 0;
                _aligner.ResetForNextPanel();
                _mosaicSet.ClearAllImages();

                if (_bUseCoreAPI)
                {
                    if (!StartSimAcquistion())
                    {
                        Output("Issue with StartAcquisition");
                        break;
                    }
                }
                else
                {
                    // Directly load image from disc
                    string sFolder = Path.GetDirectoryName(_simulationFile);
                    sFolder += "\\Cycle" + _cycleCount;
                    if(!Directory.Exists(sFolder))
                        break;

                    SimMosaicTranslator.LoadAllRawImages(_mosaicSet, sFolder);
                }         
                Output("Waiting for Images...");
                _mAlignedEvent.WaitOne();

                // Release raw buffer, Raw buffer have to hold until demosaic/memoery copy is done
                if (!_bUseCoreAPI)
                    SimMosaicTranslator.ReleaseRawBufs(_mosaicSet);

                _dtStartTime = DateTime.Now;

                // Verify that mosaic is filled in...
 
                _cycleCount++;
                _mAlignedEvent.WaitOne();
                _tsRunTime = _aligner.GetAlignmentTime();
                _tsTotalRunTime += _tsRunTime;

                _dtEndTime = DateTime.Now;

                ManagedPanelFidResultsSet set = _aligner.GetFiducialResultsSet();
                Output("Panel Skew is: " + set.dPanelSkew);
                Output("Panel dPanelXscale is: " + set.dPanelXscale);
                Output("Panel dPanelYscale is: " + set.dPanelYscale);

                IntPtr data = _mosaicSet.GetLayer(0).GetGreyStitchedBuffer();
                // If no fiducial used in alignment
                if(_bNoFiducial)
                {
                    // Adjust stitched image
                    // Since stitched image has been flattened in camera model,
                    // fiducial results need not be corrected by panel height
                    IntPtr tempData = _mosaicSet.GetLayer(0).GetGreyStitchedBuffer();
                    data = _imageFidAligner.MorphImage(tempData, _alignmentPanel.GetNumPixelsInY());
                }

                if (_featurePanel != null)
                    RunFiducialCompare(data, _featurePanel.GetNumPixelsInY(), _writer);

                if (_bSaveStitchedResultsImage && _featurePanel!=null)
                    _aligner.Save3ChannelImage("c:\\Temp\\FeatureCompareAfterCycle" + _cycleCount + ".bmp",
                        data, _alignmentPanel.GetNumPixelsInY(),
                        data, _alignmentPanel.GetNumPixelsInY(),
                        _featurePanel.GetCADBuffer(), _featurePanel.GetNumPixelsInY(),
                        _featurePanel.GetNumPixelsInY(), _featurePanel.GetNumPixelsInX());
                if (_cycleCount == 1)
                {
                    _writer.WriteLine("Units: Microns");
                    //outline is the output file column names
                    _writer.WriteLine(headerLine);
                }

                if (_cycleCount >= numberToRun)
                    bDone = true;
                else
                {
                    _mAlignedEvent.Reset();
                }
            }
        }

        private static bool StartSimAcquistion()
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
        #endregion

        #region offset calculation and result record

        private static void WriteResults(string lastOutputTextPath, string outputTextPath, string unitTestFolder)
        {
            _writer.WriteLine(" Fid#, XOffset Mean, YOffset Mean,XOffset Stdev, YOffset Stdev, Absolute XOffset Mean, Absolute YOffset Mean, Absolute XOffset Stdev, Absolute YOffset Stdev, Number of cycle ");
            for (int i = 0; i < _featurePanel.NumberOfFiducials; i++)
            {
                if (_icycleCount[i] == 0)   // If no fiducial is found
                {
                    _writer.WriteLine(
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
                    _writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", i,
                        dMeanX, dMeanY, dSdvX, dSdvY,
                        dAbsMeanX, dAbsMeanY, dAbsSdvX, dAbsSdvY, 
                        _icycleCount[i]));
                }
                _iTotalCount += _icycleCount[i];

            }
            _dXRMS = Math.Sqrt(_dXDiffSqrSumTol / (_iTotalCount));
            _dYRMS = Math.Sqrt(_dYDiffSqrSumTol / (_iTotalCount));

            _writer.WriteLine(String.Format("Average Panel Process Running time(Unites:Minutes): {0}", _tsTotalRunTime/60/_cycleCount));
            _writer.WriteLine(string.Format("MagicNumber: {0}, Xoffset RMS:{1}, Yoffset RMS:{2}", _allPanelFidDifference, _dXRMS, _dYRMS));
            _writer.WriteLine(string.Format("Average Offset: {0}", _allPanelFidDifference / _iTotalCount));

            if (File.Exists(lastOutputTextPath))
            {
                string[] lines = File.ReadAllLines(lastOutputTextPath);
                if (lines.Length >= 1)
                {
                    string lastLine = lines[lines.Length - 1];
                    string timeRecordLine = lines[lines.Length - 3];
                    string[] averageParts = lastLine.Split(':');
                    string[] timeParts = timeRecordLine.Split(':');
                    double lastAverage = 0;
                    double lastTimeRecord = 0;
                    bool bGood = false;
                    bool bGoodAver = false;
                    bool bGoodTime = false;
                    bool bHeadLine = true;
                    const string headLine =
                        "Test, Previous Average Offset(Microns),  Current Average Offset(Microns),Previous Average Panel Process Running time(Minutes),Current Average Panel Process Running time(Minutes), Test Result";
                    string finalCompCSVPath = Path.Combine(Path.GetDirectoryName(outputTextPath),
                                                           "FinalCompareResults.csv");
                    string testName = Path.GetFileNameWithoutExtension(outputTextPath);
                    string testResult;
                    if (File.Exists(finalCompCSVPath))
                        bHeadLine = false;
                    _finalCompWriter= new StreamWriter(finalCompCSVPath, true);
                    if (averageParts.Length > 1 && double.TryParse(averageParts[1], out lastAverage))
                    {
                        // Check that we are at least as good as last time (to the nearest micron)
                        if (Math.Round(_allPanelFidDifference/_iTotalCount) <= Math.Round(lastAverage))
                            bGoodAver = true;
                    }
                    if (timeParts.Length > 1 && double.TryParse(timeParts[2], out lastTimeRecord))
                    {
                        // Check that we are at least as good as last time(in 20% range)
                        if (_tsTotalRunTime/60/_cycleCount <= lastTimeRecord*1.2)
                            bGoodTime = true;
                    }
                    bGood = bGoodAver && bGoodTime;
                    testResult = (bGood ? "Passed" : "Failed");
                    Console.WriteLine("Are we as good as last time: " + (bGood ? "Yes!" : "No!"));
                    if (bHeadLine) _finalCompWriter.WriteLine(headLine);
                    _finalCompWriter.WriteLine(string.Format("{0},{1},{2},{3},{4},{5}", testName, lastAverage,
                                                            _allPanelFidDifference/_iTotalCount, lastTimeRecord,
                                                            _tsTotalRunTime/60/_cycleCount, testResult));
                    if (_finalCompWriter!= null)
                        _finalCompWriter.Close();
                    if (Directory.Exists(unitTestFolder))
                    {
                        string file =
                            Path.Combine(unitTestFolder + Path.GetFileNameWithoutExtension(lastOutputTextPath)) + ".xml";
                        NUnitXmlWriter.WriteResult(file, "CyberStitchFidTester", "AverageOffset", bGood);
                    }
                }
            }
        }

        private static void RunFiducialCompare(IntPtr data, int stride, StreamWriter writer)
        {
            int iFidNums = _featurePanel.NumberOfFiducials;

            // Cad_x, cad_y, Loc_x, Loc_y, CorrScore, Ambig 
            int iItems = 6;
            double[] dResults = new double[iFidNums*iItems];
            double xDifference = 0;
            double yDifference = 0;
            double fidDifference = 0;
            string sNofid = "N/A";
            //convert meters to microns
            int iUnitCoverter = 1000000;
            // Find features on the board
            _featureChecker.CheckFeatureLocation(data, stride, dResults);
            //Record the processing time
            //DateTime dtEndTime = DateTime.Now;
            //TimeSpan tsRunTime = dtEndTime - _dtStartTime;

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
                    _dXDiffSqrSumTol += Math.Pow(xDifference, 2);
                    _dYDiffSqrSumTol += Math.Pow(yDifference, 2);
                }
            }

            writer.WriteLine("Total Difference for this Panel: " + fidDifference);
            _allPanelFidDifference += fidDifference;
            writer.WriteLine(string.Format("Panel Process Start Time: {0}, Panel Processing end time: {1},Panel process running time: {2}" ,_dtStartTime, _dtEndTime, _tsRunTime));
           // _tsTotalRunTime += tsRunTime;
        } 
        
        #endregion
    }

   
}
