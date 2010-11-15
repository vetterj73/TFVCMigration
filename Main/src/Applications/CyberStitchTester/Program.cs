using System;
using System.IO;
using System.Threading;
using MCoreAPI;
using MLOGGER;
using MMosaicDM;
using SIMAPI;

namespace CyberStitchTester
{
    class Program
    {
        private const double cCameraOverlap = .4;
        private const double cTriggerOverlap = .4;
        private static ManagedMosaicSet _mosaicSet = null;
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static double _panelWidth = 200;   // in mm
        private static double _panelHeight = 200;  // in mm
        /// <summary>
        /// Use SIM to load up an image set and run it through the stitch tools...
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {

            string simulationFile = "";

            for(int i=0; i<args.Length; i++)
            {
                if(args[i] == "-w" && i<args.Length-1)
                {
                    _panelWidth = Convert.ToInt16(args[i + 1]);
                }
                else if (args[i] == "-h" && i < args.Length - 1)
                {
                    _panelHeight = Convert.ToInt16(args[i + 1]);
                }
                else if (args[i] == "-s" && i < args.Length - 1)
                {
                    simulationFile = args[i + 1];
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
                return;
            }

            if(ManagedCoreAPI.NumberOfDevices() <=0)
            {
                Output("There are no SIM Devices attached!");
                return;
            }

            if(!bSimulating)
            {
                for(int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
                {
                    ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                    
                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_panelWidth/1000.0, _panelHeight/1000.0, 0, .004);
                    if(cs1==null)
                    {
                        Output("Could not create capture spec.");        
                    }
                }
            }

            SetupMosaic();

            for(int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
            {
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(i);
                d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE);
            }

            Output("Waiting for Images...");
            mDoneEvent.WaitOne();

            // Now - Correlate and Stitch...
            Output("All Done!");
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
            _mosaicSet = new ManagedMosaicSet(_panelWidth / 1000.0, _panelHeight / 1000.0, 2592, 1944, 2592, 1.69e-5, 1.69e-5);
            _mosaicSet.OnImageAdded += OnImageAddedToMosaic;
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);

            for(int i=0; i<ManagedCoreAPI.NumberOfDevices(); i++)
                AddDeviceToMosaic(ManagedCoreAPI.GetDevice(i));
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
                ManagedMosaicLayer layer = _mosaicSet.AddLayer(.2, pSpec.XOffset(), numCameras, .004, pSpec.NumberOfTriggers, .004, false);

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
                    ManagedSIMCamera camera = d.GetSIMCamera(i);
                    for (int k = 0; k < pSpec.NumberOfTriggers; k++)
                    {
                        ManagedMosaicTile mmt = layer.GetTile(j, k);

                        if (mmt == null)
                        {
                            Output("Could not access tile at: " + j + ", " + k);
                            return;
                        }

                        // First camera center in X
                        double xOffset = _panelWidth - pSpec.GetTriggerAtIndex(k) - camera0.Pixelsize.X * camera0.Rows()/2;
                        // The camera center in X
                        xOffset += (camera.CenterOffset.X - camera0.CenterOffset.X);
                        // The camera's origin in X
                        xOffset -= (camera.Pixelsize.X * camera.Rows() / 2);

                        // First camera center in Y
                        double yOffset = (-d.YOffset + camera0.Pixelsize.Y * camera0.Columns() / 2); ;
                        // The camera center in Y
                        yOffset = (camera.CenterOffset.Y - camera0.CenterOffset.X);
                        // The camera orign in Y
                        yOffset -= (camera.Pixelsize.Y * camera.Columns() / 2);

                        // Trigger offset is initial offset + triggerIndex * overlap...
                        mmt.SetTransformParameters(camera.Pixelsize.X, camera.Pixelsize.Y,
                            xOffset, yOffset, camera.Rotation);
                    }
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
            Output("Image was added to the Mosaic!!!!!!!");
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
            Output("Got an Image!");
            int layer = pframe.DeviceIndex()*ManagedCoreAPI.GetDevice(0).NumberOfCaptureSpecs +
                        pframe.CaptureSpecIndex();
            _mosaicSet.AddImage(pframe.BufferPtr(), layer, pframe.CameraIndex(),
                                pframe.TriggerIndex());
        }

        private static void Output(string str)
        {
            Console.WriteLine(str);
        }
    }
}
