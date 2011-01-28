using System;
using System.Drawing.Imaging;
using System.IO;
using System.Threading;
using CPanelIO;
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
        private static CPanel _panel = new CPanel(0, 0, cPixelSizeInMeters, cPixelSizeInMeters); 
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread logger = new LoggingThread(null);
        private static int _cycleCount = 0;
        // For debug
        private static int _iBufCount = 0;

        /// <summary>
        /// Use SIM to load up an image set and run it through the stitch tools...
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Start the logger
            logger.Start("Logger", @"c:\\", "CyberStitch.log", true, -1);

            // Gather input data.
            string simulationFile = "";
            string panelFile="";
            bool bContinuous = false;
            for(int i=0; i<args.Length; i++)
            {
                if (args[i] == "-c")
                    bContinuous = true;
                if (args[i] == "-s" && i < args.Length - 1)
                    simulationFile = args[i + 1];
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
            SetupMosaic();

            // Set up logger for aligner
            _aligner.OnLogEntry += OnLogEntryFromClient;
            _aligner.SetAllLogTypes(true);

            // Set up production for aligner
            try
            {
                if(!_aligner.ChangeProduction(_mosaicSet, _panel))
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

            bool bDone = false;
            while(!bDone)
            {
                numAcqsComplete = 0;

                _aligner.ResetForNextPanel();
               
                _mosaicSet.ClearAllImages();
                if (!GatherImages())
                    bDone = true;
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

                    // Save the images...
                    for (uint i = 0; i < _mosaicSet.GetNumMosaicLayers(); i++)
                    {
                        ManagedMosaicLayer layer = _mosaicSet.GetLayer(i);

                        IntPtr buffer = layer.GetStitchedBuffer();

                        Cyber.ImageUtils.ImageSaver.SaveToFile
                        (
                            _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX(), _panel.GetNumPixelsInY(),
                            buffer, "c:\\temp\\stitchedcyle" + _cycleCount + "_" + i + ".png", PixelFormat.Format8bppIndexed, ImageFormat.Png
                        );
                    }

                    // Save a 3 channel image with CAD data...
                    _aligner.Save3ChannelImage("c:\\temp\\3channelresultcycle" + _cycleCount + ".bmp",
                        _mosaicSet.GetLayer(2).GetStitchedBuffer(),
                        _mosaicSet.GetLayer(3).GetStitchedBuffer(),
                        _panel.GetCADBuffer(),
                        _panel.GetNumPixelsInY(), _panel.GetNumPixelsInX());

                }

                // should we do another cycle?
                if (!bContinuous)
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
            for(int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
            {
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE);
            }
            return true;
        }

        private static bool InitializeSimCoreAPI(string simulationFile)
        {
            bool bSimulating = false;
            if (!string.IsNullOrEmpty(simulationFile) && File.Exists(simulationFile))
                bSimulating = true;

            if (bSimulating)
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

            if (!bSimulating)
            {
                for (int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
                {
                    ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);

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
        private static void SetupMosaic()
        {
            if (ManagedCoreAPI.NumberOfDevices() <= 0)
            {
                Output("No Device Defined");
                return;
            }
            _mosaicSet = new ManagedMosaicSet(_panel.PanelSizeX, _panel.PanelSizeY, 2592, 1944, 2592, cPixelSizeInMeters, cPixelSizeInMeters);
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);

            SimMosaicTranslator.InitializeMosaicFromCurrentSimConfig(_mosaicSet);
        }

        private static void OnLogEntryFromMosaic(MLOGTYPE logtype, string message)
        {
            Output(logtype + " From Mosaic: " + message);
        }

        private static void OnAcquisitionDone(int device, int status, int count)
        {
            Output("OnAcquisitionDone Called!");
            numAcqsComplete++;
            if (ManagedCoreAPI.NumberOfDevices() == numAcqsComplete)
                mDoneEvent.Set();
         }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
            Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
                pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));
            _iBufCount++; // for debug

            uint layer = (uint)(pframe.DeviceIndex()*ManagedCoreAPI.GetDevice(0).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex());
            _mosaicSet.AddImage(pframe.BufferPtr(), layer, (uint)pframe.CameraIndex(),
                                (uint)pframe.TriggerIndex());
        }

        private static void Output(string str)
        {
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }
    }
}
