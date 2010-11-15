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
        private static ManagedMosaicSet _mosaicSet = null;
        private readonly static ManualResetEvent mDoneEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        /// <summary>
        /// Use SIM to load up an image set and run it through the stitch tools...
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            int panelWidth = 200;
            int panelHeight = 200;
            string simulationFile = "";

            for(int i=0; i<args.Length; i++)
            {
                if(args[i] == "-w" && i<args.Length-1)
                {
                    panelWidth = Convert.ToInt16(args[i + 1]);
                }
                else if (args[i] == "-h" && i < args.Length - 1)
                {
                    panelHeight = Convert.ToInt16(args[i + 1]);
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
                    
                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(panelWidth/100.0, panelHeight/100.0, 0, .004);
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
            _mosaicSet = new ManagedMosaicSet(.200, .250, 2592, 1944, 2592, .00017, .00017);
            _mosaicSet.OnImageAdded += OnImageAddedToMosaic;
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);

            for(int i=0; i<ManagedCoreAPI.NumberOfDevices(); i++)
                AddDeviceToMosaic(ManagedCoreAPI.GetDevice(i));
        }

        private static void AddDeviceToMosaic(ManagedSIMDevice d)
        {
            ManagedSIMCaptureSpec pSpec = d.GetSIMCaptureSpec(0);
            if (pSpec == null)
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
                ManagedMosaicLayer layer = _mosaicSet.AddLayer(.2, i * .2, numCameras, .003, pSpec.NumberOfTriggers, .004, false);

                if (layer == null)
                {
                    Output("Could not create Layer: " + i);
                    return;
                }

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

                        // Trigger offset is initial offset + triggerIndex * overlap...
                        double triggerOffset = .2;
                        mmt.SetTransformParameters(camera.Pixelsize.X, camera.Pixelsize.Y,
                            triggerOffset, d.YOffset + camera.CenterOffset.Y, camera.Rotation);
                    }
                }
            }
        }

        private static void OnLogEntryFromMosaic(string message, MLOGTYPE logtype)
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
