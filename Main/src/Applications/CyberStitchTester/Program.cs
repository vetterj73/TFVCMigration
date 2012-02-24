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
        private const double cPixelSizeInMeters = 1.70e-5;
        private static ManagedMosaicSet _mosaicSet = null;
        private static ManagedMosaicSet _mosaicSetCopy = null;
        private static CPanel _panel = new CPanel(0, 0, cPixelSizeInMeters, cPixelSizeInMeters); 
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread logger = new LoggingThread(null);
        private static uint _numThreads = 8;
        private static int _cycleCount = 0;

        private static bool _bRtoL = false; // right to left conveyor direction indicator
        private static bool _bFRR = false; // fixed rear rail indicator

        // For debug
        private static int _iBufCount = 0;
        private static bool _bSimulating = false;
        private static bool _bBayerPattern = false;
        private static int _iBayerType = 1; // GBRG
 
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
            bool bOwnBuffers = false;
            bool bMaskForDiffDevices = false;
            bool bAdjustForHeight = true;
            bool bUseProjective = false;
            bool bUseCameraModel = false;
            int numberToRun = 1;

            for(int i=0; i<args.Length; i++)
            {
                if (args[i] == "-b")
                    bOwnBuffers = true;
                if (args[i] == "-c")
                    bContinuous = true;
                else if (args[i] == "-n" && i < args.Length - 1)
                    numberToRun = Convert.ToInt16(args[i + 1]);
                if (args[i] == "-m")
                    bMaskForDiffDevices = true;
                if (args[i] == "-bayer")
                    _bBayerPattern = true;
                if (args[i] == "-w")
                    bUseProjective = true;
                if (args[i] == "-nh")
                    bAdjustForHeight = false;
                if (args[i] == "-cammod")
                    bUseCameraModel = true;
                if (args[i] == "-rtol")
                    _bRtoL = true;
                if (args[i] == "-frr")
                    _bFRR = true;
                if (args[i] == "-s" && i < args.Length - 1)
                    simulationFile = args[i + 1];
                if (args[i] == "-t" && i < args.Length - 1)
                    _numThreads = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-p" && i < args.Length - 1)
                    panelFile = args[i + 1];
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

            // Set up production for aligner
            try
            {
                _aligner.NumThreads(_numThreads);
                //_aligner.LogOverlaps(true);
                //_aligner.LogFiducialOverlaps(true);
                //_aligner.UseCyberNgc4Fiducial();
                if(bUseProjective)
                    _aligner.UseProjectiveTransform(true);
                if (bUseCameraModel)
                {
                    _aligner.UseCameraModelStitch(true);
                    _aligner.UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
                }

                Output("Before ChangeProduction");
                if (!_aligner.ChangeProduction(_mosaicSet, _panel))
                {
                    throw new ApplicationException("Aligner failed to change production ");
                }
                Output("After ChangeProduction");
            }
            catch (Exception except)
            {
                Output("Error: " + except.Message);
                logger.Kill();
                return;
            }

            bool bDone = false;
            while(!bDone)
            {
                numAcqsComplete = 0;

                _aligner.ResetForNextPanel();
               
                _mosaicSet.ClearAllImages();
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
                    _cycleCount++;                   
                    // After a panel is stitched and before aligner is reset for next panel
                    ManagedPanelFidResultsSet fidResultSet = _aligner.GetFiducialResultsSet();
                    
                    // Calculate indices
                    uint iLayerIndex1 = 0;
                    uint iLayerIndex2 = 0;
                    switch (_mosaicSet.GetNumMosaicLayers())
                    {
                        case 1:
                            iLayerIndex1 = 0;
                            iLayerIndex2 = 0;
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

                    if (_bBayerPattern) // for bayer pattern
                    {
                        if (Directory.Exists("c:\\temp\\jrhResults\\Cycle_" + (_cycleCount - 1)) == false)
                        {
                            Directory.CreateDirectory("c:\\temp\\jrhResults\\Cycle_" + (_cycleCount - 1));
                        }

                        if (_mosaicSet.SaveAllStitchedImagesToDirectory("c:\\temp\\jrhResults\\Cycle_" + (_cycleCount - 1) + "\\") == false)
                            Output("Could not save mosaic images");

                        /* for debug 
                        _aligner.Save3ChannelImage("c:\\temp\\Aftercycle" + _cycleCount + ".bmp",
                            _mosaicSet.GetLayer(iLayerIndex1).GetGreyStitchedBuffer(),
                            _mosaicSet.GetLayer(iLayerIndex2).GetGreyStitchedBuffer(),
                            _panel.GetCADBuffer(), //heightBuf,
                            _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());
                        //*/
                    }
                    else // for gray scale
                    {
                        // Save a 3 channel image with CAD data...
                        _aligner.Save3ChannelImage("c:\\temp\\Aftercycle" + _cycleCount + ".bmp",
                            _mosaicSet.GetLayer(iLayerIndex1).GetStitchedBuffer(),
                            _mosaicSet.GetLayer(iLayerIndex2).GetStitchedBuffer(),
                            _panel.GetCADBuffer(), 
                            _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                        /*
                        _aligner.Save3ChannelImage("c:\\temp\\Beforecycle" + _cycleCount + ".bmp",
                            _mosaicSet.GetLayer(0).GetStitchedBuffer(),
                            _mosaicSet.GetLayer(1).GetStitchedBuffer(),
                            _panel.GetCADBuffer(), //heightBuf,
                            _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                        _aligner.Save3ChannelImage("c:\\temp\\Brightcycle" + _cycleCount + ".bmp",
                            _mosaicSet.GetLayer(0).GetStitchedBuffer(),
                            _mosaicSet.GetLayer(iLayerIndex1).GetStitchedBuffer(),
                            _panel.GetCADBuffer(), //heightBuf,
                            _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                        _aligner.Save3ChannelImage("c:\\temp\\Darkcycle" + _cycleCount + ".bmp",
                            _mosaicSet.GetLayer(1).GetStitchedBuffer(),
                            _mosaicSet.GetLayer(iLayerIndex2).GetStitchedBuffer(),
                            _panel.GetCADBuffer(), //heightBuf,
                            _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());
                        //*/

                        // Get the stitch grid 
                        // Must after get stitched image of the same layer
                        int[] pCols = new int[_mosaicSet.GetLayer(iLayerIndex1).GetNumberOfCameras() + 1];
                        int[] pRows = new int[_mosaicSet.GetLayer(iLayerIndex1).GetNumberOfTriggers() + 1];
                        _mosaicSet.GetLayer(iLayerIndex1).GetStitchGrid(pCols, pRows);

                        /*/ Testing a copy of mosaic...
                        _mosaicSetCopy.CopyBuffers(_mosaicSet);
                        _mosaicSetCopy.CopyTransforms(_mosaicSet);
                        _aligner.Save3ChannelImage("c:\\temp\\3channelresultcyclecopy" + _cycleCount + ".bmp",
                             _mosaicSetCopy.GetLayer(iLayerIndex1).GetStitchedBuffer(),
                             _mosaicSetCopy.GetLayer(iLayerIndex2).GetStitchedBuffer(),
                             _panel.GetCADBuffer(),
                             _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());
                         */
                    }                       
                }

                // should we do another cycle?
                if (!bContinuous && _cycleCount >= numberToRun)
                    bDone = true;
                else
                    mDoneEvent.Reset();
            }

            Output("Processing Complete");
            logger.Kill();
            ManagedCoreAPI.TerminateAPI();
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

                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_panel.PanelSizeX, _panel.PanelSizeY, 0, .004);
                    if (cs1 == null)
                    {
                        Output("Could not create capture spec.");
                        return false;
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
                        _panel = SRFToPanel.parseSRF(panelFile, cPixelSizeInMeters, cPixelSizeInMeters);
                        if (_panel == null)
                            throw new ApplicationException("Could not parse the SRF panel file");
                    }
                    else if (panelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
                    {
                        _panel = XmlToPanel.CSIMPanelXmlToCPanel(panelFile, cPixelSizeInMeters, cPixelSizeInMeters);
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
            _mosaicSet = new ManagedMosaicSet(_panel.PanelSizeX, _panel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters, bOwnBuffers, _bBayerPattern, _iBayerType);
            _mosaicSetCopy = new ManagedMosaicSet(_panel.PanelSizeX, _panel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters, bOwnBuffers, _bBayerPattern, _iBayerType);
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);
            SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSetCopy, bMaskForDiffDevices);
            SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSet, bMaskForDiffDevices);
        }

        private static void OnLogEntryFromMosaic(MLOGTYPE logtype, string message)
        {
            Output(logtype + " From Mosaic: " + message);
        }

        private static void OnAcquisitionDone(int device, int status, int count)
        {
            Output("OnAcquisitionDone Called!");
            numAcqsComplete++;
            // lauch next device in simulation case
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
           //     pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));
            _iBufCount++; // for debug

            int device = pframe.DeviceIndex();
            int mosaic_row = SimMosaicTranslator.TranslateTrigger(pframe);
            int mosaic_column = pframe.CameraIndex() - ManagedCoreAPI.GetDevice(device).FirstCameraEnabled;

            uint layer = (uint)(pframe.DeviceIndex() * ManagedCoreAPI.GetDevice(device).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex());

            _mosaicSet.AddRawImage(pframe.BufferPtr(), layer, (uint)mosaic_column, (uint)mosaic_row);
        }

        private static void Output(string str)
        {
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }
    }
}
