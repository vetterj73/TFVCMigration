using System;
using System.IO;
using System.Threading;
using CPanelIO;
using Cyber.DiagnosticUtils;
using Cyber.SPIAPI;
using MCoreAPI;
using MLOGGER;
using MMosaicDM;
using SIMAPI;
using PanelAlignM;

namespace CyberStitchTester
{
    class Program
    {
        private const double cCameraOverlap = .4;
        private const double cTriggerOverlap = .4;
        private static ManagedMosaicSet _mosaicSet = null;
        private static CPanel _panel = new CPanel(0, 0); 
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread logger = new LoggingThread(null);
        private static int _iDeviceDoneCount = 0;

        private static int iBufCount = 0;

        /// <summary>
        /// Use SIM to load up an image set and run it through the stitch tools...
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            logger.Start("Logger", @"c:\\", "CyberStitch.log", true, -1);
            string simulationFile = "";
            string panelFile="";

            for(int i=0; i<args.Length; i++)
            {
                if (args[i] == "-s" && i < args.Length - 1)
                {
                    simulationFile = args[i + 1];
                }
                else if (args[i] == "-p" && i < args.Length - 1)
                {
                    panelFile = args[i + 1];
                }

            }

            if (!string.IsNullOrEmpty(panelFile))
            {
                try
                {
                    if (panelFile.EndsWith(".srf", StringComparison.CurrentCultureIgnoreCase))
                    {
                        if (!SRFToPanel.parseSRF(panelFile, _panel))
                            throw new ApplicationException("Could not parse the SRF panel file");
                    }
                    else if (panelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
                    {
                        if(!XmlToPanel.CSIMPanelXmlToCPanel(panelFile, ref _panel))
                            throw new ApplicationException("Could not convert xml panel file");
                    }
                }
                catch (Exception except)
                {
                    Output("Exception reading Panel file: " + except.Message);
                    logger.Kill();
                    return;
                }
            }


            bool bSimulating = false;
            if (!string.IsNullOrEmpty(simulationFile) && File.Exists(simulationFile))
                bSimulating = true;

            if(bSimulating)
            {
                Output("Running with Simulation File: " + simulationFile);
                ManagedCoreAPI.SetSimulationFile(simulationFile);
            }

            ManagedSIMDevice.OnFrameDone += OnFrameDone;
            ManagedSIMDevice.OnAcquisitionDone += OnAcquisitionDone;

            if(ManagedCoreAPI.InitializeAPI() != 0)
            {
                Output("Could not initialize CoreAPI!");
                logger.Kill();
                return;
            }

            if(ManagedCoreAPI.NumberOfDevices() <=0)
            {
                Output("There are no SIM Devices attached!");
                logger.Kill();
                return;
            }

            if(!bSimulating)
            {
                for(int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
                {
                    ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                    
                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_panel.PanelSizeX, _panel.PanelSizeY, 0, .004);
                    if(cs1==null)
                    {
                        Output("Could not create capture spec.");        
                    }
                }
            }

            SetupMosaic();

            _aligner.OnLogEntry += OnLogEntryFromClient;
            _aligner.SetAllLogTypes(true);

            try
            {
                _aligner.SetPanel(_mosaicSet, _panel);

            }
            catch (Exception except)
            {
                
                Output("Error during SetPanel: " + except.Message);
            }

            for(int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
            {
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE);
            }

            Output("Waiting for Images...");
            mDoneEvent.WaitOne();

            // Now - Correlate and Stitch...
            Output("All Done!");

            logger.Kill();
        }

        private static void OnLogEntryFromClient( MLOGTYPE logtype, string message)
        {
            Console.WriteLine(logtype + " " + message);
            DateTime dataTime = DateTime.Now;
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
            _mosaicSet = new ManagedMosaicSet(_panel.PanelSizeX, _panel.PanelSizeY, 2592, 1944, 2592, 1.69e-5, 1.69e-5);
            _mosaicSet.OnImageAdded += OnImageAddedToMosaic;
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);

            for(int i=0; i<ManagedCoreAPI.NumberOfDevices(); i++)
                AddDeviceToMosaic(ManagedCoreAPI.GetDevice(i));

            SetupCorrelateFlags();
        }

        private static void AddDeviceToMosaic(ManagedSIMDevice d)
        {

            if (d.NumberOfCaptureSpecs <=0)
            {
                Output("No Capture Specs defined");
                return;
            }

            /// @todo - this should be made part of the SIM Device....
            int numCameras = 0;
            for (int i = 0; i < d.NumberOfCameras; i++)
                if (d.GetSIMCamera(i).Status() == (CameraStatus)1)
                    numCameras++;

            for (int i = 0; i < d.NumberOfCaptureSpecs; i++)
            {
                ManagedSIMCaptureSpec pSpec = d.GetSIMCaptureSpec(i);
                ManagedMosaicLayer layer = _mosaicSet.AddLayer(numCameras, pSpec.NumberOfTriggers, false);

                if (layer == null)
                {
                    Output("Could not create Layer: " + i);
                    return;
                }

                // Use camera zero as reference
                ManagedSIMCamera camera0 = d.GetSIMCamera(0);
                /// Set up the transform parameters...
                for (int j = 0; j < numCameras; j++)
                {
                    ManagedSIMCamera camera = d.GetSIMCamera(j);
                    for (int k = 0; k < pSpec.NumberOfTriggers; k++)
                    {
                        ManagedMosaicTile mmt = layer.GetTile(j, k);

                        if (mmt == null)
                        {
                            Output("Could not access tile at: " + j + ", " + k);
                            return;
                        }
                        double dTmep = pSpec.GetTriggerAtIndex(k);

                        // First camera center in X
                        double dTrigOffset = pSpec.GetTriggerAtIndex(k) + pSpec.XOffset();
                        double xOffset = _panel.PanelSizeX- dTrigOffset - camera0.Pixelsize.X * camera0.Rows()/2;
                        // The camera center in X
                        xOffset += (camera.CenterOffset.X - camera0.CenterOffset.X);
                        // The camera's origin in X
                        xOffset -= (camera.Pixelsize.X * camera.Rows() / 2);

                        // First camera center in Y
                        double yOffset = (-d.YOffset + camera0.Pixelsize.Y * camera0.Columns() / 2); ;
                        // The camera center in Y
                        yOffset += (camera.CenterOffset.Y - camera0.CenterOffset.Y);
                        // The camera orign in Y
                        yOffset -= (camera.Pixelsize.Y * camera.Columns() / 2);

                        // Trigger offset is initial offset + triggerIndex * overlap...
                        mmt.SetTransformParameters(camera.Pixelsize.X, camera.Pixelsize.Y,
                            camera.Rotation,
                            xOffset, yOffset);
                    }
                }
            }
        }

        private static void SetupCorrelateFlags()
        { 
            for (int i = 0; i < _mosaicSet.GetNumMosaicLayers(); i++)
            {
                for (int j = 0; j < _mosaicSet.GetNumMosaicLayers(); j++)
                {
                    ManagedCorrelationFlags flag =_mosaicSet.GetCorrelationSet(i, j);
                    if (i == j)
                    {
                        flag.SetCameraToCamera(true);
                        if(_mosaicSet.GetNumMosaicLayers() == 1)
                            flag.SetTriggerToTrigger(true); // For one illumination
                        else
                            flag.SetTriggerToTrigger(false);
                    }
                    else
                    {
                        flag.SetTriggerToTrigger(true);
                        flag.SetCameraToCamera(false);
                    }
                    flag.SetMaskNeeded(false);
                }
            }
        }

        private static void OnLogEntryFromMosaic(MLOGTYPE logtype, string message)
        {
            Output(logtype + " From Mosaic: " + message);
        }

        /// <summary>
        /// NOTE:  This is not desired in the real application - it is just a way to test the mosaic.
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="cameraIndex"></param>
        /// <param name="triggerIndex"></param>
        private static void OnImageAddedToMosaic(int layerIndex, int cameraIndex, int triggerIndex)
        {
            iBufCount++;
            Output("Image was added to the Mosaic!!!!!!!" + iBufCount);
        }

        private static void OnAcquisitionDone(int device, int status, int count)
        {
            try
            {
                Output("OnAcquisitionDone Called!");
                numAcqsComplete++;

                // For two illuminations
                if (_mosaicSet.GetNumMosaicLayers() == 2)
                {
                    int iNumTrigs = _mosaicSet.GetLayer(0).GetNumberOfTriggers() + _mosaicSet.GetLayer(1).GetNumberOfTriggers();
                    for (int j = 0; j < iNumTrigs; j++)
                    {
                        int iIllum = j % 2;
                        int iTrig = j / 2;
                        for (int iCam = 0; iCam < _mosaicSet.GetLayer(iIllum).GetNumberOfCameras(); iCam++)
                        {
                            if (!_aligner.AddImage(iIllum, iTrig, iCam))
                            {
                                Output("Failed to add image!");
                                return;
                            }
                        }
                    }
                }

                // For one illumination
                if (_mosaicSet.GetNumMosaicLayers() == 1)
                {
                    for (int iTrig = 0; iTrig < _mosaicSet.GetLayer(0).GetNumberOfTriggers(); iTrig++)
                    {
                        for (int iCam = 0; iCam < _mosaicSet.GetLayer(0).GetNumberOfCameras(); iCam++)
                        {
                            if (!_aligner.AddImage(0, iTrig, iCam))
                            {
                                Output("Failed to add image!");
                                return;
                            }
                        }
                    }
                }

                int iNum = ManagedCoreAPI.NumberOfDevices();

                // For four illuminationas
                if (_mosaicSet.GetNumMosaicLayers() == 4)
                {  
                    // before two illuminaitons
                    if (_iDeviceDoneCount == 0)
                    { 
                        int iNumTrigs = _mosaicSet.GetLayer(0).GetNumberOfTriggers() + _mosaicSet.GetLayer(1).GetNumberOfTriggers();
                        for (int j = 0; j < iNumTrigs; j++)
                        {
                            int iIllum = j % 2;
                            int iTrig = j / 2;
                            for (int iCam = 0; iCam < _mosaicSet.GetLayer(iIllum).GetNumberOfCameras(); iCam++)
                            {
                                if (!_aligner.AddImage(iIllum, iTrig, iCam))
                                {
                                    Output("Failed to add image!");
                                    return;
                                }
                            }
                        }

                        _iDeviceDoneCount = 1;
                        return;
                    }

                    // After two illuminaitons
                    if (_iDeviceDoneCount == 1)
                    {
                        int iNumTrigs = _mosaicSet.GetLayer(2).GetNumberOfTriggers() + _mosaicSet.GetLayer(3).GetNumberOfTriggers();
                        for (int j = 0; j < iNumTrigs; j++)
                        {
                            int iIllum = j % 2 + 2;
                            int iTrig = j / 2;
                            for (int iCam = 0; iCam < _mosaicSet.GetLayer(iIllum).GetNumberOfCameras(); iCam++)
                            {
                                if (!_aligner.AddImage(iIllum, iTrig, iCam))
                                {
                                    Output("Failed to add image!");
                                    return;
                                }
                            }
                        }
                    }
                }
            
                if (ManagedCoreAPI.NumberOfDevices() == numAcqsComplete)
                mDoneEvent.Set();

            }
            catch (Exception except)
            {
                Output("Error during OnAcquisitionDone: " + except.Message);
            }

         }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
            Output("Got an Image!");
            int layer = pframe.DeviceIndex()*ManagedCoreAPI.GetDevice(0).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex();
            _mosaicSet.AddImage(pframe.BufferPtr(), layer, pframe.CameraIndex(),
                                pframe.TriggerIndex());
        }

        private static void Output(string str)
        {
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }
    }
}
