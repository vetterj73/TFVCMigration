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
        // For live mode image capture setup
        private static double _dTriggerOverlapInM = 0.004; //Image capture overlap setup
        private static int _iBrightField = 31; // Bright field intensity setup(0-31)
        private static int _iDarkField = 31; // Dark field intensity setup(0-31)
        private static double _dTriggerStartOffsetInMCS1 = -0.0185; //before sim start offset setup 
        private static double _dTriggerStartOffsetInMCS2 = -0.004; //after sim start offset setup
        private static bool _bRtoL = false; // right to left conveyor direction indicator
        private static bool _bFRR = false; // fixed rear rail indicator
        private static bool _bUseDualIllumination = true;// default use dual illumination for live mode
        
        // Control parameters
        private static double _dPixelSizeInMeters = -1;
        private static uint _iInputImageColumns = 0; // 2952 for SIM 110
        private static uint _iInputImageRows = 0; // 1944 for SIM 110
        private static bool _bBayerPattern = false;
        private static int _iBayerType = 1; // GBRG
        private static bool _bSkipDemosaic = false;  // true: Skip demosaic for Bayer image
        private static bool _bDetectPanelEdge = false;
        private static int _iLayerIndex4Edge = 0;
        private static uint _numThreads = 8;
        private static string _simulationFile = "";
        private static string _panelFile = "";
        private static bool _bContinuous = false;
        private static int _numberToRun = 1;
        private static bool _bOwnBuffers = true; /// Must be true because we are release buffers immediately.
        private static bool _bMaskForDiffDevices = false;
        private static bool _bAdjustForHeight = true;
        private static bool _bUseProjective = true;
        private static bool _bUseCameraModel = false;
        private static bool _bUseIterativeCameraModel = false;
        private static bool _bSeperateProcessStages = false;
        private static bool _bUseTwoPassStitch = false;
        private static bool _bNoFiducial = false;
        
        // Internal variable
        private static ManagedMosaicSet _mosaicSet = null;
        private static CPanel _panel = null; 
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread _logger = new LoggingThread(null);
        
        private static int _numAcqsComplete = 0;
        private readonly static AutoResetEvent _mDoneEvent = new AutoResetEvent(false);
        private static bool _bSimulating = false;
        private static uint _iLayerIndex1 = 0;
        private static uint _iLayerIndex2 = 0;

        private static bool _bUseCoreAPI = true;

        /// <summary>
        /// Use SIM to load up an image set and run it through the stitch tools...
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Start the _logger
            _logger.Start("Logger", @"c:\\Temp\\", "CyberStitch.log", true, -1);

            // Input control parameter
            for(int i=0; i<args.Length; i++)
            {
                if (args[i] == "-c")
                    _bContinuous = true;
                else if (args[i] == "-n" && i < args.Length - 1)
                    _numberToRun = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-m")
                    _bMaskForDiffDevices = true;
                else if (args[i] == "-b")
                    _bBayerPattern = true;
                else if (args[i] == "-w")
                    _bUseProjective = true;
                else if (args[i] == "-nw")
                    _bUseProjective = false;
                else if (args[i] == "-nh")
                    _bAdjustForHeight = false;
                else if (args[i] == "-nf")
                    _bNoFiducial = true;
                else if (args[i] == "-cammod")
                    _bUseCameraModel = true;
                else if (args[i] == "-de")
                    _bDetectPanelEdge = true;
                else if (args[i] == "-iter")
                    _bUseIterativeCameraModel = true;
                else if (args[i] == "-rtol")
                    _bRtoL = true;
                else if (args[i] == "-frr")
                    _bFRR = true;
                else if (args[i] == "-sps")
                    _bSeperateProcessStages = true;
                else if (args[i] == "-twopass")
                    _bUseTwoPassStitch = true;
                else if (args[i] == "-skipD")
                    _bSkipDemosaic = true;
                else if (args[i] == "-sillu")
                    _bUseDualIllumination = false;
                else if (args[i] == "-s" && i < args.Length - 1)
                    _simulationFile = args[i + 1];
                else if (args[i] == "-t" && i < args.Length - 1)
                    _numThreads = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-p" && i < args.Length - 1)
                    _panelFile = args[i + 1];
				else if (args[i] == "-overlap" && i < args.Length - 1)
                    _dTriggerOverlapInM = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-brightfield" && i < args.Length - 1)
                    _iBrightField = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-darkfield" && i < args.Length - 1)
                    _iDarkField = Convert.ToUInt16(args[i + 1]);
                else if (args[i] == "-sim1startoffset" && i < args.Length - 1)
                   _dTriggerStartOffsetInMCS1 = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-sim2startoffset" && i < args.Length - 1)
                    _dTriggerStartOffsetInMCS2 = Convert.ToDouble(args[i + 1]);
            }

            if (_simulationFile.EndsWith(".csv", StringComparison.CurrentCultureIgnoreCase))
            {
                _bUseCoreAPI = false;
                Output("Using a simualtion file that ends in .csv is not presently supported.");
                Output("Because we have no way of knowing the pixel size with this approach.  Yet.  JRH");
                _logger.Kill();
                return;
            }

            if (_bUseCoreAPI)
            {
                // Initialize the SIM CoreAPI
                // Includes determining nominal pixel size of SIM.
                if (!InitializeSimCoreAPI())
                {
                    _logger.Kill();
                    return;
                }
            }




            // Setup the panel based on panel file
            if (!LoadPanelDecription())
            {
                _logger.Kill();
                return;
            }

            // Remove fiducial information from panel
            if (_bNoFiducial)
                _panel.ClearFiducials();

            // Finish SIM configuration (Must be done after panel defined, but before Mosaic and Aligner defined, as they
            // use SIM configs to set up themselves.)
            if (!SetupSIM())
            {
                Output("SIM Failed to configure!");
                return;
            }

            // Set up mosaic set
            if (!SetupMosaic())
            {
                Output("Failed to setup mosaic!");
                return;
            }

            // Set up logger for aligner
            if (!SetupAligner())
            {
                Output("Failed to setup aligner!");
                return;
            }


            // Run stitch
            RunStitch();

            Output("Processing Complete");
            _logger.Kill();

            _aligner.Dispose();

            ManagedCoreAPI.TerminateAPI();
        }

        private static void OnLogEntryFromMosaic(MLOGTYPE logtype, string message)
        {
            Output(logtype + " From Mosaic: " + message);
        }

        private static void OnLogEntryFromClient(MLOGTYPE logtype, string message)
        {
            Console.WriteLine(logtype + " " + message);
            Output(logtype + " " + message);
        }

        private static void Output(string str)
        {
            _logger.AddObjectToThreadQueue(str);
            _logger.AddObjectToThreadQueue(null);
        }

        #region Load panel
        private static bool LoadPanelDecription()
        {
            if (!string.IsNullOrEmpty(_panelFile))
            {
                try
                {
                    if (_panelFile.EndsWith(".srf", StringComparison.CurrentCultureIgnoreCase))
                    {
                        _panel = SRFToPanel.parseSRF(_panelFile, _dPixelSizeInMeters, _dPixelSizeInMeters);
                        if (_panel == null)
                            throw new ApplicationException("Could not parse the SRF panel file");
                    }
                    else if (_panelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
                    {
                        _panel = XmlToPanel.CSIMPanelXmlToCPanel(_panelFile, _dPixelSizeInMeters, _dPixelSizeInMeters);
                        if (_panel == null)
                            throw new ApplicationException("Could not convert xml panel file");
                    }
                }
                catch (Exception except)
                {
                    Output("Exception reading Panel file: " + except.Message);
                    _logger.Kill();
                    return false;
                }
            }
            return true;
        }
        #endregion

        #region coreAPI and callbacks
        private static void OnAcquisitionDone(int device, int status, int count)
        {
            if (0 == device)
                Output("End SIM" + device + " acquisition");
            _numAcqsComplete++;
            // lauch next device in simulation case
            if (_bSimulating && _numAcqsComplete < ManagedCoreAPI.NumberOfDevices())
            {
                //                Thread.Sleep(10000);
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(_numAcqsComplete);
                Output("Begin SIM" + _numAcqsComplete + " acquisition");
                if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                    return;
            }
        }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
            //Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
            //   pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));

            int device = pframe.DeviceIndex();
            int mosaic_row = SimMosaicTranslator.TranslateTrigger(pframe);
            int mosaic_column = pframe.CameraIndex() - ManagedCoreAPI.GetDevice(device).FirstCameraEnabled;

            uint layer = (uint)(pframe.DeviceIndex() * ManagedCoreAPI.GetDevice(device).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex());

            _mosaicSet.AddRawImage(pframe.BufferPtr(), layer, (uint)mosaic_column, (uint)mosaic_row);

            ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
            d.ReleaseFrameBuffer(pframe);
        }

        private static bool InitializeSimCoreAPI()
        {
            //bool bSimulating = false;
            if (!string.IsNullOrEmpty(_simulationFile))
            {
                if (File.Exists(_simulationFile))
                    _bSimulating = true;
                else
                {
                    Output("Simulation file >> " + _simulationFile + " << does not exist, assuming you want to not run in simulation mode.");
                    _bSimulating = false;
                }
            }

            if (_bSimulating)
            {
                Output("Running with Simulation File: " + _simulationFile);
                ManagedCoreAPI.SetSimulationFile(_simulationFile);
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

            // Determine Pixel size on SIM.  Make sure they're all consistent.
            _dPixelSizeInMeters = -1;
            for (int ix = 0; ix < ManagedCoreAPI.NumberOfDevices(); ix++)
            {
                if (_dPixelSizeInMeters < 0)
                {
                    _dPixelSizeInMeters = Math.Round(1000000 * ManagedCoreAPI.GetDevice(ix).AveragePixelSizeX) / 1000000;
                    double tmp = Math.Round(1000000 * ManagedCoreAPI.GetDevice(ix).AveragePixelSizeY) / 1000000;
                    if (tmp != _dPixelSizeInMeters)
                    {
                        Output("Pixel Sizes don't match on SIM Device ID " + ix + _dPixelSizeInMeters + " " + tmp);
                        return false;
                    }
                }
                else
                {
                    double tmpX = Math.Round(1000000 * ManagedCoreAPI.GetDevice(ix).AveragePixelSizeX) / 1000000;
                    double tmpY = Math.Round(1000000 * ManagedCoreAPI.GetDevice(ix).AveragePixelSizeY) / 1000000;
                    if (tmpX != tmpY)
                    {
                        Output("Pixel Sizes don't match on SIM Device ID " + ix + " " + tmpX + " " + tmpY);
                        return false;
                    }
                    else if (tmpX != _dPixelSizeInMeters)
                    {
                        Output("Pixel Sizes on SIM Device ID " + ix + " don't Match Device 0 " + tmpX + " " + _dPixelSizeInMeters);
                        return false;
                    }
                }
            }

            // Determine pixels on SIM.  Make sure they're all consistent.
            _iInputImageColumns = 0; //  2592;
            _iInputImageRows = 0; //  1944;
            for (int ix = 0; ix < ManagedCoreAPI.NumberOfDevices(); ix++)
            {
                ManagedSIMDevice device = ManagedCoreAPI.GetDevice(ix);
                for (int jx = 0; jx < device.NumberOfCameras; jx++)
                {
                    ManagedSIMCamera camera = device.GetSIMCamera(jx);
                    if (_iInputImageColumns == 0)
                    {
                        _iInputImageColumns = (uint) camera.Columns();
                        _iInputImageRows = (uint) camera.Rows();
                    }
                    else
                    {
                        if (_iInputImageColumns != (uint)camera.Columns()
                            || _iInputImageRows != (uint)camera.Rows())
                        {
                            Output("Camera sizes are changing on SIM Device " + ix + " " + _iInputImageColumns + " " + _iInputImageRows + " " + camera.Columns() + " " + camera.Rows());
                            return false;
                        }
                    }
                }
            }



            return true;
        }

        private static bool SetupSIM()
        {
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

                    d.RemoveCaptureSpecs();
                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_panel.PanelSizeX, _panel.PanelSizeY, _dTriggerStartOffsetInMCS1, _dTriggerOverlapInM);
                    if (cs1 == null)
                    {
                        Output("Could not create capture spec.");
                        return false;
                    }

                    cs1.GetIllumination().BrightFieldIntensity(_iBrightField);
                    if (!_bUseDualIllumination)
                    {
                        cs1.GetIllumination().DarkFieldIntensity(_iDarkField);
                    }
                    else
                    {
                        cs1.GetIllumination().DarkFieldIntensity(0);

                        ManagedSIMCaptureSpec cs2 = d.SetupCaptureSpec(_panel.PanelSizeX, _panel.PanelSizeY, _dTriggerStartOffsetInMCS2, _dTriggerOverlapInM);
                        cs2.GetIllumination().BrightFieldIntensity(0);
                        cs2.GetIllumination().DarkFieldIntensity(_iDarkField);
                    }
                }
            }

            return true;
        }
        #endregion

        #region mosaicSet and Alinger setup
        /// <summary>
        /// Given a SIM setup and a mosaic for stitching, setup the stich...
        /// </summary>
        private static bool SetupMosaic()
        {
            if (_bUseCoreAPI && ManagedCoreAPI.NumberOfDevices() <= 0)
            {
                Output("No Device Defined");
                return false;
            }

            _mosaicSet = new ManagedMosaicSet(
                _panel.PanelSizeX, _panel.PanelSizeY, 
                _iInputImageColumns, _iInputImageRows, _iInputImageColumns, 
                _dPixelSizeInMeters, _dPixelSizeInMeters, 
                _bOwnBuffers,
                _bBayerPattern, _iBayerType, _bSkipDemosaic);
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);

            //_mosaicSet.SetGaussianDemosaic(true);

            // Fill mosaicset
            if (_bUseCoreAPI)
                SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSet, _bMaskForDiffDevices);
            else
                SimMosaicTranslator.InitializeMosaicFromNominalTrans(_mosaicSet, _simulationFile, _panel.PanelSizeX);

            return (true);
        }

        private static bool SetupAligner()
        {
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
                _aligner.LogFiducialOverlaps(true);
                //_aligner.UseCyberNgc4Fiducial();
                //_aligner.LogPanelEdgeDebugImages(true);
                _aligner.UseProjectiveTransform(_bUseProjective);
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
                _mosaicSet.SetSeperateProcessStages(_bSeperateProcessStages);

                // Must after InitializeSimCoreAPI() before ChangeProduction()
                if (_bUseCoreAPI)
                {
                    ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
                    _aligner.SetPanelEdgeDetection(_bDetectPanelEdge, _iLayerIndex4Edge, !d.ConveyorRtoL, !d.FixedRearRail);
                }
                else
                {
                    _aligner.SetPanelEdgeDetection(_bDetectPanelEdge, _iLayerIndex4Edge, true, true);
                }

                // true: Skip demosaic for Bayer image
                if (_bBayerPattern)
                    _aligner.SetSkipDemosaic(_bSkipDemosaic);

                /* for debug
                _mosaicSet.SetFiducailCadLoc(0, 5e-3, 10e-3);
                _mosaicSet.SetFiducailCadLoc(1, 160e-3, 10e-3);
                _mosaicSet.SetFiducailCadLoc(2, 160e-3, 120e-3);
                //*/

                /* for debug
                _mosaicSet.SetFiducailCadLoc(0, 4.915e-3, 9.71e-3);
                _mosaicSet.SetFiducailCadLoc(1, 159.9e-3, 9.71e-3);
                _mosaicSet.SetFiducailCadLoc(2, 159.9e-3, 119.95e-3);
                //*/
                
                if (!_aligner.ChangeProduction(_mosaicSet, _panel))
                {
                    throw new ApplicationException("Aligner failed to change production ");
                }
            }
            catch (Exception except)
            {
                Output("Error: " + except.Message);
                _logger.Kill();
                return false;
            }

            // Calculate indices
            switch (_mosaicSet.GetNumMosaicLayers())
            {
                case 1:
                    _iLayerIndex1 = 0;
                    _iLayerIndex2 = 0;
                    _bUseDualIllumination = false;    // Single layer case
                    break;

                case 2:
                    _iLayerIndex1 = 0;
                    _iLayerIndex2 = 1;
                    break;

                case 4:
                    _iLayerIndex1 = 2;
                    _iLayerIndex2 = 3;
                    break;
            }

            // Set component height if it exist
            double dMaxHeight = 0;
            if (_bAdjustForHeight)
                dMaxHeight = _panel.GetMaxComponentHeight();

            if (dMaxHeight > 0)
            {
                bool bSmooth = true;
                IntPtr heightBuf = _panel.GetHeightImageBuffer(bSmooth);
                uint iSpan = (uint)_panel.GetNumPixelsInY();
                double dHeightRes = _panel.GetHeightResolution();
                double dPupilDistance = 0.3702;
                // Need modified based on layers that have component 
                _mosaicSet.GetLayer(_iLayerIndex1).SetComponentHeightInfo(heightBuf, iSpan, dHeightRes, dPupilDistance);
                _mosaicSet.GetLayer(_iLayerIndex2).SetComponentHeightInfo(heightBuf, iSpan, dHeightRes, dPupilDistance);
            }

            return (true);
        }
        #endregion

        #region Run stitch 
        private static bool AcquireSIMImages()
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

        private static void OnAlignmentDone(bool status)
        {
            Output("End cyberstitch alignment");
            _mDoneEvent.Set();
        }

        private static void RunStitch()
        {
            int iCycleCount = 0; 

            bool bDone = false;
            while (!bDone)
            {
                _numAcqsComplete = 0;

                _aligner.ResetForNextPanel();

                _mosaicSet.ClearAllImages();

                /* for debug
                _mosaicSet.SetFiducialFovLoc(0,
                    0, 6, 0,
                    2503.607, 436.062);
                _mosaicSet.SetFiducialFovLoc(1,
                    0, 0, 0,
                    2437.501, 716.265);
                _mosaicSet.SetFiducialFovLoc(2,
                    0, 0, 3,
                    1867.331, 763.484);
                 //*/

                Output("Begin stitch cycle...");
                if (_bUseCoreAPI)
                {   // acquire image by coreAPI
                    if (!AcquireSIMImages())
                    {
                        Output("Issue with StartAcquisition");
                        break;
                    }
                }
                else
                {   // Directly load image from disc
                    string sFolder = Path.GetDirectoryName(_simulationFile);
                    sFolder += "\\Cycle" + iCycleCount;
                    if(!Directory.Exists(sFolder))
                        break;

                    SimMosaicTranslator.LoadAllRawImages(_mosaicSet, sFolder);
                }
                Output("Waiting for Images...");
                _mDoneEvent.WaitOne();

                // Release raw buffer, Raw buffer have to hold until demosaic/memoery copy is done
                if (!_bUseCoreAPI)
                    SimMosaicTranslator.ReleaseRawBufs(_mosaicSet);

                // Verify that mosaic is filled in...

                Output("End stitch cycle");

                iCycleCount++;
                // After a panel is stitched and before aligner is reset for next panel
                ManagedPanelFidResultsSet fidResultSet = _aligner.GetFiducialResultsSet();
                if (_bUseCameraModel || _bUseIterativeCameraModel)
                {
                    double[] zCof = new double[16];
                    if (!_aligner.GetCamModelPanelHeight(0, zCof))
                        Output("Failed to get panel Z Coff for camera model!");
                }

                Output("Begin morph");

                if (_bBayerPattern) // for bayer pattern
                {   /*
                    if (Directory.Exists("c:\\temp\\jrhResults\\Cycle_" + (iCycleCount - 1)) == false)
                    {
                        Directory.CreateDirectory("c:\\temp\\jrhResults\\Cycle_" + (iCycleCount - 1));
                    }
                    
                    if (_mosaicSet.SaveAllStitchedImagesToDirectory("c:\\temp\\jrhResults\\Cycle_" + (iCycleCount - 1) + "\\") == false)
                        Output("Could not save mosaic images");
                    //*/

                    //if (_mosaicSet.SaveAllStitchedImagesToDirectory("c:\\temp\\Cycle_" + (iCycleCount - 1)) == false)
                    //    Output("Could not save mosaic images");
                }

                _aligner.Save3ChannelImage("c:\\temp\\Aftercycle" + iCycleCount + ".bmp",
                    _mosaicSet.GetLayer(_iLayerIndex1).GetGreyStitchedBuffer(),
                    _mosaicSet.GetLayer(_iLayerIndex2).GetGreyStitchedBuffer(),
                    _panel.GetCADBuffer(), //heightBuf,
                    _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                if (_bUseDualIllumination)
                {
                    /*
                    _aligner.Save3ChannelImage("c:\\temp\\Beforecycle" + iCycleCount + ".bmp",
                        _mosaicSet.GetLayer(0).GetGreyStitchedBuffer(),
                        _mosaicSet.GetLayer(1).GetGreyStitchedBuffer(),
                        _panel.GetCADBuffer(), //heightBuf,
                        _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                    _aligner.Save3ChannelImage("c:\\temp\\Brightcycle" + iCycleCount + ".bmp",
                        _mosaicSet.GetLayer(0).GetGreyStitchedBuffer(),
                        _mosaicSet.GetLayer(_iLayerIndex1).GetGreyStitchedBuffer(),
                        _panel.GetCADBuffer(), //heightBuf,
                        _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                    _aligner.Save3ChannelImage("c:\\temp\\Darkcycle" + iCycleCount + ".bmp",
                        _mosaicSet.GetLayer(1).GetGreyStitchedBuffer(),
                        _mosaicSet.GetLayer(_iLayerIndex2).GetGreyStitchedBuffer(),
                        _panel.GetCADBuffer(), //heightBuf,
                        _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());
                    //*/
                }

                // Get fiducial information
                ManagedPanelFidResultsSet set = _aligner.GetFiducialResultsSet();

                // Get the stitch grid 
                // Must after get stitched image of the same layer
                int[] pCols = new int[_mosaicSet.GetLayer(_iLayerIndex1).GetNumberOfCameras() + 1];
                int[] pRows = new int[_mosaicSet.GetLayer(_iLayerIndex1).GetNumberOfTriggers() + 1];
                _mosaicSet.GetLayer(_iLayerIndex1).GetStitchGrid(pCols, pRows);

                // For image patch test
                //IntPtr pPoint = _mosaicSet.GetLayer(_iLayerIndex1).GetStitchedBuffer();
                //ManagedFOVPreferSelected select = new ManagedFOVPreferSelected();
                //_mosaicSet.GetLayer(_iLayerIndex1).GetImagePatch(pPoint, 100, 0, 0, 100, 100, select);

                Output("End morph");

                // should we do another cycle?
                if (!_bContinuous && iCycleCount >= _numberToRun)
                    bDone = true;
            }
        }

        #endregion
    }
}
