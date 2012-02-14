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

namespace MultiAlignerTester
{
    class Program
    {
        private const int iNumAligner = 2;
        private static int _iCurAligner = 0;
        private const double cPixelSizeInMeters = 1.70e-5;
        private static ManagedMosaicSet [] _mosaicSets = new ManagedMosaicSet[iNumAligner];
        private static ManagedMosaicSet[] _mosaicSetCopys = new ManagedMosaicSet[iNumAligner];
        private static CPanel _panel = new CPanel(0, 0, cPixelSizeInMeters, cPixelSizeInMeters); 
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment [] _aligners = new ManagedPanelAlignment[iNumAligner];
        private static LoggingThread logger = new LoggingThread(null);
        private static uint _numThreads = 4;
        private static int _cycleCount = 0;
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

            for (int i = 0; i < iNumAligner; i++)
            {
                _aligners[i] = new ManagedPanelAlignment();

                // Set up logger for aligner
                _aligners[i].OnLogEntry += OnLogEntryFromClient;
                _aligners[i].SetAllLogTypes(true);
                //_aligners[i].LogTransformVectors(true);

                // Set up production for aligner
                try
                {
                    _aligners[i].NumThreads(_numThreads);
                    //_aligner.LogOverlaps(true);
                    _aligners[i].LogFiducialOverlaps(true);
                    //_aligner.UseCyberNgc4Fiducial();
                    if (bUseProjective)
                        _aligners[i].UseProjectiveTransform(true);
                    if (bUseCameraModel)
                    {
                        _aligners[i].UseCameraModelStitch(true);
                        _aligners[i].UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
                    }

                    Output("Before ChangeProduction");
                    if (!_aligners[i].ChangeProduction(_mosaicSets[i], _panel))
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
            }

            bool bDone = false;
            while(!bDone)
            {
                numAcqsComplete = 0;
                _iCurAligner = 0;

                for (int i = 0; i < iNumAligner; i++)
                {
                    _aligners[i].ResetForNextPanel();
                    _mosaicSets[i].ClearAllImages();
                    _mosaicSetCopys[i].ClearAllImages();
                }
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

                if (_mosaicSetCopys[0].HasAllImages() && _mosaicSetCopys[1].HasAllImages())
                {
                    for (uint i = 0; i < _mosaicSets[0].GetNumMosaicLayers(); i++)
                    {
                        ManagedMosaicLayer pRefLayer = _mosaicSetCopys[0].GetLayer(i);

                        for (uint j = 0; j < pRefLayer.GetNumberOfCameras(); j++)
                        {
                            for (uint k = 0; k < pRefLayer.GetNumberOfTriggers(); k++)
                            {
                                for (uint iIndex = 0; iIndex < iNumAligner; iIndex++)
                                {
                                    ManagedMosaicLayer pLayer = _mosaicSetCopys[iIndex].GetLayer(i);

                                    if (_bBayerPattern)
                                        _mosaicSets[iIndex].AddYCrCbImage(pLayer.GetTile(j, k).GetImageBuffer(), i, j, k);
                                    else
                                        _mosaicSets[iIndex].AddRawImage(pLayer.GetTile(j, k).GetImageBuffer(), i, j, k);

                                }
                            }
                        }
                    }
                }


                // Verify that mosaic is filled in...
                if (_mosaicSets[0].HasAllImages() && _mosaicSets[1].HasAllImages())
                {
                    _cycleCount++;                   
                    // After a panel is stitched and before aligner is reset for next panel
                    ManagedPanelFidResultsSet fidResultSet = _aligners[0].GetFiducialResultsSet();

                    for (int i = 0; i < iNumAligner; i++)
                    {
                        //* for debug 
                        _aligners[i].Save3ChannelImage("c:\\temp\\Aftercycle" + _cycleCount + "_" + i + ".bmp",
                            _mosaicSets[i].GetLayer(2).GetGreyStitchedBuffer(),
                            _mosaicSets[i].GetLayer(3).GetGreyStitchedBuffer(),
                            _panel.GetCADBuffer(), //heightBuf,
                            _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());
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
            for(int i=0; i<iNumAligner; i++)
            {
                _mosaicSets[i] = new ManagedMosaicSet(_panel.PanelSizeX, _panel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters, bOwnBuffers, _bBayerPattern, _iBayerType);
                _mosaicSetCopys[i] = new ManagedMosaicSet(_panel.PanelSizeX, _panel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters, bOwnBuffers, _bBayerPattern, _iBayerType);
                _mosaicSets[i].OnLogEntry += OnLogEntryFromMosaic;
                _mosaicSets[i].SetLogType(MLOGTYPE.LogTypeDiagnostic, true);
                SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSets[i], bMaskForDiffDevices);
                SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSetCopys[i], bMaskForDiffDevices);
            }
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
            {
                _iCurAligner++;
                if (_iCurAligner == iNumAligner)
                    mDoneEvent.Set();
                else
                {
                    numAcqsComplete = 0;
                    GatherImages();
                }
            }
        }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
           // Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
           //     pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));
            _iBufCount++; // for debug

            uint layer = (uint)(pframe.DeviceIndex()*ManagedCoreAPI.GetDevice(0).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex());
            _mosaicSetCopys[_iCurAligner].AddRawImage(pframe.BufferPtr(), layer, (uint)pframe.CameraIndex(),
                                (uint)pframe.TriggerIndex());
        }

        private static void Output(string str)
        {
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }
    }
}
