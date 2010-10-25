using System;
using System.IO;
using System.Threading;
using MCoreAPI;
using MCyberStitch;

namespace CyberStitchTester
{
    class Program
    {
        private static ManagedMosaicSet _mosaicSet;
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
        }

        private static void Output(string str)
        {
            Console.WriteLine(str);
        }
    }
}
