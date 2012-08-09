using System;
using System.IO;
using System.Threading;
using MPanelIO;
using Cyber.DiagnosticUtils;
using Cyber.MPanel;
using MCoreAPI;
using MLOGGER;
using MMosaicDM;
using PanelAlignM;
using SIMMosaicUtils;

namespace CyberStitchTester
{
    class Program
    {
        private static double dPixelSizeInMeters = 1.70e-5;

        // for live mode image capture setup
        //start
        private static double dTriggerOverlapInM = 0.004; //Image capture overlap setup
        private static int iBrightField = 31; // Bright field intensity setup(0-31)
        private static int iDarkField = 31; // Dark field intensity setup(0-31)
        private static double dTriggerStartOffsetInMCS1 = -0.0185; //before sim start offset setup 
        private static double dTriggerStartOffsetInMCS2 = -0.004; //after sim start offset setup
        //end
        private static uint iInputImageColumns = 2592;
        private static uint iInputImageRows = 1944;
        private static ManagedMosaicSet _mosaicSet = null;
        private static CPanel _panel = new CPanel(0, 0, dPixelSizeInMeters, dPixelSizeInMeters); 
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread logger = new LoggingThread(null);
        private static uint _numThreads = 8;
        private static int _cycleCount = 0;

        private static bool _bDetectPanelEdge = false;
        private static bool _bRtoL = false; // right to left conveyor direction indicator
        private static bool _bFRR = false; // fixed rear rail indicator
        private static bool _bUseDualIllumination = true;// default use dual illumination for live mode

        private static int _iBufCount = 0;
        private static bool _bSimulating = false;
        private static bool _bBayerPattern = false;
        private static int _iBayerType = 1; // GBRG
        private static bool _bSkipDemosaic = false;  // true: Skip demosaic for Bayer image
 
        /// <summary>
        /// Use SIM to load up an image set and run it through the stitch tools...
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Start the logger
            logger.Start("Logger", @"c:\\Temp\\", "CyberStitch.log", true, -1);

            // Gather input data.
            string simulationFile = "";
            string panelFile="";
            bool bContinuous = false;
            bool bOwnBuffers = true; /// Must be true because we are release buffers immediately.
            bool bMaskForDiffDevices = false;
            bool bAdjustForHeight = true;
            bool bUseProjective = true;
            bool bUseCameraModel = false;
            bool bUseIterativeCameraModel = false;
            bool bSeperateProcessStages = false;
            bool bUseTwoPassStitch = false;
            
            int numberToRun = 1;
            int iLayerIndex4Edge = 0;

            for(int i=0; i<args.Length; i++)
            {
                if (args[i] == "-c")
                    bContinuous = true;
                else if (args[i] == "-n" && i < args.Length - 1)
                    numberToRun = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-m")
                    bMaskForDiffDevices = true;
                else if (args[i] == "-bayer")
                    _bBayerPattern = true;
                else if (args[i] == "-w")
                    bUseProjective = true;
                else if (args[i] == "-nw")
                    bUseProjective = false;
                else if (args[i] == "-nh")
                    bAdjustForHeight = false;
                else if (args[i] == "-cammod")
                    bUseCameraModel = true;
                else if (args[i] == "-de")
                    _bDetectPanelEdge = true;
                else if (args[i] == "-iter")
                    bUseIterativeCameraModel = true;
                else if (args[i] == "-rtol")
                    _bRtoL = true;
                else if (args[i] == "-frr")
                    _bFRR = true;
                else if (args[i] == "-sps")
                    bSeperateProcessStages = true;
                else if (args[i] == "-twopass")
                    bUseTwoPassStitch = true;
                else if (args[i] == "-skipD")
                    _bSkipDemosaic = true;
                else if (args[i] == "-sillu")
                    _bUseDualIllumination = false;
                else if (args[i] == "-s" && i < args.Length - 1)
                    simulationFile = args[i + 1];
                else if (args[i] == "-t" && i < args.Length - 1)
                    _numThreads = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-p" && i < args.Length - 1)
                    panelFile = args[i + 1];
                else if (args[i] == "-pixsize" && i < args.Length - 1)
                    dPixelSizeInMeters = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-imgcols" && i < args.Length - 1)
                    iInputImageColumns = Convert.ToUInt32(args[i + 1]);
                else if (args[i] == "-imgrows" && i < args.Length - 1)
                    iInputImageRows = Convert.ToUInt32(args[i + 1]);
				else if (args[i] == "-overlap" && i < args.Length - 1)
                    dTriggerOverlapInM = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-brightfield" && i < args.Length - 1)
                    iBrightField = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-darkfield" && i < args.Length - 1)
                    iDarkField = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-sim1startoffset" && i < args.Length - 1)
                   dTriggerStartOffsetInMCS1 = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-sim2startoffset" && i < args.Length - 1)
                    dTriggerStartOffsetInMCS2 = Convert.ToDouble(args[i + 1]);
            }

            // Setup the panel based on panel file
            if (!ChangeProductionFile(panelFile))
            {
                logger.Kill();
                return;
            }

            // Initialize the SIM CoreAPI
            if (!InitializeSimCoreAPI(simulationFile))
            {
                logger.Kill();
                return;
            }
            
            // Set up mosaic set
            SetupMosaic(bOwnBuffers, bMaskForDiffDevices);

            // Set up logger for aligner
            _aligner.OnLogEntry += OnLogEntryFromClient;
            _aligner.SetAllLogTypes(true);
            //_aligner.LogTransformVectors(true);

            // Set up aligner delegate
            _aligner.OnAlignmentDone += OnAlignmentDone;

            // Set up production for aligner
            try
            {
                _aligner.NumThreads(_numThreads);
                _mosaicSet.SetThreadNumber(_numThreads);
                //_mosaicSet.SetThreadNumber(1);
                //_aligner.LogOverlaps(true);
                //_aligner.LogFiducialOverlaps(true);
                //_aligner.UseCyberNgc4Fiducial();
                //_aligner.LogPanelEdgeDebugImages(true);
                _aligner.UseProjectiveTransform(bUseProjective);
                if (bUseTwoPassStitch)
                    _aligner.SetUseTwoPassStitch(true);
                if (bUseCameraModel)
                {
                    _aligner.UseCameraModelStitch(true);
                    _aligner.UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
                }
                if (bUseIterativeCameraModel)
                {
                    _aligner.UseCameraModelIterativeStitch(true);
                    _aligner.UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
                }
                _mosaicSet.SetSeperateProcessStages(bSeperateProcessStages);

                // Must after InitializeSimCoreAPI() before ChangeProduction()
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
                _aligner.SetPanelEdgeDetection(_bDetectPanelEdge, iLayerIndex4Edge, !d.ConveyorRtoL, !d.FixedRearRail);

                // true: Skip demosaic for Bayer image
                if (_bBayerPattern)
                    _aligner.SetSkipDemosaic(_bSkipDemosaic);

                // Add trigger to trigger overlaps for same layer
                //for (uint i = 0; i < _mosaicSet.GetNumMosaicLayers(); i++)
                //    _mosaicSet.GetCorrelationSet(i, i).SetTriggerToTrigger(true);

                if (!_aligner.ChangeProduction(_mosaicSet, _panel))
                {
                    throw new ApplicationException("Aligner failed to change production ");
                }
            }
            catch (Exception except)
            {
                Output("Error: " + except.Message);
                logger.Kill();
                return;
            }

            // Calculate indices
            uint iLayerIndex1 = 0;
            uint iLayerIndex2 = 0;
            switch (_mosaicSet.GetNumMosaicLayers())
            {
                case 1:
                    iLayerIndex1 = 0;
                    iLayerIndex2 = 0;
                    _bUseDualIllumination = false;    // Single layer case
                    break;

                case 2:
                    iLayerIndex1 = 0;
                    iLayerIndex2 = 1;
                    break;

                case 4:
                    iLayerIndex1 = 2;
                    iLayerIndex2 = 3;
                    break;
            }

            // Set component height if it exist
            double dMaxHeight = 0;
            if (bAdjustForHeight)
                dMaxHeight = _panel.GetMaxComponentHeight();

            if (dMaxHeight > 0)
            {
                bool bSmooth = true;
                IntPtr heightBuf = _panel.GetHeightImageBuffer(bSmooth);
                uint iSpan = (uint)_panel.GetNumPixelsInY();
                double dHeightRes = _panel.GetHeightResolution();
                double dPupilDistance = 0.3702;
                // Need modified based on layers that have component 
                _mosaicSet.GetLayer(iLayerIndex1).SetComponentHeightInfo(heightBuf, iSpan, dHeightRes, dPupilDistance);
                _mosaicSet.GetLayer(iLayerIndex2).SetComponentHeightInfo(heightBuf, iSpan, dHeightRes, dPupilDistance);
            }

            bool bDone = false;
            while(!bDone)
            {
                numAcqsComplete = 0;

                _aligner.ResetForNextPanel();
               
                _mosaicSet.ClearAllImages();
                Output("Begin stitch cycle...");
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
                if (!_mosaicSet.HasAllImages())
                    Output("The mosaic does not contain all images!");
                else
                {
                    Output("End stitch cycle");

                    _cycleCount++;                   
                    // After a panel is stitched and before aligner is reset for next panel
                    ManagedPanelFidResultsSet fidResultSet = _aligner.GetFiducialResultsSet();

                    Output("Begin morph");

                    if (_bBayerPattern) // for bayer pattern
                    {
                       /* if (Directory.Exists("c:\\temp\\jrhResults\\Cycle_" + (_cycleCount - 1)) == false)
                        {
                            Directory.CreateDirectory("c:\\temp\\jrhResults\\Cycle_" + (_cycleCount - 1));
                        }

                        if (_mosaicSet.SaveAllStitchedImagesToDirectory("c:\\temp\\jrhResults\\Cycle_" + (_cycleCount - 1) + "\\") == false)
                            Output("Could not save mosaic images");
                        */
                    }
                   
                    _aligner.Save3ChannelImage("c:\\temp\\Aftercycle" + _cycleCount + ".bmp",
                        _mosaicSet.GetLayer(iLayerIndex1).GetGreyStitchedBuffer(),
                        _mosaicSet.GetLayer(iLayerIndex2).GetGreyStitchedBuffer(),
                        _panel.GetCADBuffer(), //heightBuf,
                        _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                    if (_bUseDualIllumination)
                    {
                        //*
                        _aligner.Save3ChannelImage("c:\\temp\\Beforecycle" + _cycleCount + ".bmp",
                         _mosaicSet.GetLayer(0).GetGreyStitchedBuffer(),
                         _mosaicSet.GetLayer(1).GetGreyStitchedBuffer(),
                         _panel.GetCADBuffer(), //heightBuf,
                         _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                        _aligner.Save3ChannelImage("c:\\temp\\Brightcycle" + _cycleCount + ".bmp",
                         _mosaicSet.GetLayer(0).GetGreyStitchedBuffer(),
                         _mosaicSet.GetLayer(iLayerIndex1).GetGreyStitchedBuffer(),
                         _panel.GetCADBuffer(), //heightBuf,
                         _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                        _aligner.Save3ChannelImage("c:\\temp\\Darkcycle" + _cycleCount + ".bmp",
                         _mosaicSet.GetLayer(1).GetGreyStitchedBuffer(),
                         _mosaicSet.GetLayer(iLayerIndex2).GetGreyStitchedBuffer(),
                         _panel.GetCADBuffer(), //heightBuf,
                          _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());
                     //*/
                    }
                     

                    // Get fiducial information
                    ManagedPanelFidResultsSet set = _aligner.GetFiducialResultsSet();

                    // Get the stitch grid 
                    // Must after get stitched image of the same layer
                    int[] pCols = new int[_mosaicSet.GetLayer(iLayerIndex1).GetNumberOfCameras() + 1];
                    int[] pRows = new int[_mosaicSet.GetLayer(iLayerIndex1).GetNumberOfTriggers() + 1];
                    _mosaicSet.GetLayer(iLayerIndex1).GetStitchGrid(pCols, pRows);

                    // For image patch test
                    //IntPtr pPoint = _mosaicSet.GetLayer(iLayerIndex1).GetStitchedBuffer();
                    //ManagedFOVPreferSelected select = new ManagedFOVPreferSelected();
                    //_mosaicSet.GetLayer(iLayerIndex1).GetImagePatch(pPoint, 100, 0, 0, 100, 100, select);

                    Output("End morph");
                }

                // should we do another cycle?
                if (!bContinuous && _cycleCount >= numberToRun)
                    bDone = true;
                else
                    mDoneEvent.Reset();
            }

            Output("Processing Complete");
            logger.Kill();

            _aligner.Dispose();

            ManagedCoreAPI.TerminateAPI();
        }

        private static bool GatherImages()
        {
            if (!_bSimulating)
            {
                
                for (int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
                {
                    ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                    Output("Begin SIM" + i + " acquisition");
                    if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                        return false;
                }
             
                
            }
            else
            {   // launch device one by one in simulation case
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
                Output("Begin SIM0 acquisition");
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
                    int bufferCount = 256;// (triggerCount + 1) * GetNumberOfEnabledCameras(0) * 2;
                    int desiredCount = bufferCount;
                    d.AllocateFrameBuffers(ref bufferCount);

                    if(desiredCount != bufferCount)
                    {
                        Output("Could not allocate all buffers!  Desired = " + desiredCount + " Actual = " + bufferCount);
                        logger.Kill();
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

                    d.RemoveCaptureSpecs();
                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_panel.PanelSizeX, _panel.PanelSizeY, dTriggerStartOffsetInMCS1, dTriggerOverlapInM);
                    if (cs1 == null)
                    {
                        Output("Could not create capture spec.");
                        return false;
                    }

                    cs1.GetIllumination().BrightFieldIntensity(iBrightField);
                    if (!_bUseDualIllumination)
                    {
                        cs1.GetIllumination().DarkFieldIntensity(iDarkField);
                    }
                    else
                    {
                        cs1.GetIllumination().DarkFieldIntensity(0);

                        ManagedSIMCaptureSpec cs2 = d.SetupCaptureSpec(_panel.PanelSizeX, _panel.PanelSizeY, dTriggerStartOffsetInMCS2, dTriggerOverlapInM);
                        cs2.GetIllumination().BrightFieldIntensity(0);
                        cs2.GetIllumination().DarkFieldIntensity(iDarkField);
                    }
                }
            }

            return true;
        }

        private static bool ChangeProductionFile(string panelFile)
        {
            if (!string.IsNullOrEmpty(panelFile))
            {
                try
                {
                    if (panelFile.EndsWith(".srf", StringComparison.CurrentCultureIgnoreCase))
                    {
                        _panel = SRFToPanel.parseSRF(panelFile, dPixelSizeInMeters, dPixelSizeInMeters);
                        if (_panel == null)
                            throw new ApplicationException("Could not parse the SRF panel file");
                    }
                    else if (panelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
                    {
                        _panel = XmlToPanel.CSIMPanelXmlToCPanel(panelFile, dPixelSizeInMeters, dPixelSizeInMeters);
                        if (_panel == null)
                            throw new ApplicationException("Could not convert xml panel file");
                    }
                }
                catch (Exception except)
                {
                    Output("Exception reading Panel file: " + except.Message);
                    logger.Kill();
                    return false;
                }
            }
            return true;
        }

        private static void OnLogEntryFromClient( MLOGTYPE logtype, string message)
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
            _mosaicSet = new ManagedMosaicSet(
                _panel.PanelSizeX, _panel.PanelSizeY, 
                iInputImageColumns, iInputImageRows, iInputImageColumns, 
                dPixelSizeInMeters, dPixelSizeInMeters, 
                bOwnBuffers,
                _bBayerPattern, _iBayerType, _bSkipDemosaic);
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);
            SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSet, bMaskForDiffDevices);
        }

        private static void OnLogEntryFromMosaic(MLOGTYPE logtype, string message)
        {
            Output(logtype + " From Mosaic: " + message);
        }

        private static void OnAcquisitionDone(int device, int status, int count)
        {
            if(0 == device)
                Output("End SIM"+device+" acquisition");
            numAcqsComplete++;
            // lauch next device in simulation case
            if (_bSimulating && numAcqsComplete < ManagedCoreAPI.NumberOfDevices())
            {
//                Thread.Sleep(10000);
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(numAcqsComplete);
                Output("Begin SIM" + numAcqsComplete + " acquisition");
                if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                    return;
            }
            //if (ManagedCoreAPI.NumberOfDevices() == numAcqsComplete)
            //    mDoneEvent.Set();
        }

        private static void OnAlignmentDone(bool status)
        {
            Output("End cyberstitch alignment");
            mDoneEvent.Set();
        }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
            //Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
             //   pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));
            _iBufCount++; // for debug

            int device = pframe.DeviceIndex();
            int mosaic_row = SimMosaicTranslator.TranslateTrigger(pframe);
            int mosaic_column = pframe.CameraIndex() - ManagedCoreAPI.GetDevice(device).FirstCameraEnabled;

            uint layer = (uint)(pframe.DeviceIndex() * ManagedCoreAPI.GetDevice(device).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex());

            _mosaicSet.AddRawImage(pframe.BufferPtr(), layer, (uint)mosaic_column, (uint)mosaic_row);

            ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
            d.ReleaseFrameBuffer(pframe);
        }

        private static void Output(string str)
        {
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }
    }
}
