using System;
using System.IO;
using System.Threading;
using CPanelIO;
using Cyber.DiagnosticUtils;
using Cyber.MPanel;
using MCoreAPI;
using MLOGGER;
using SIMCalibrator;

namespace SIMCalibratorTester
{
    class Program
    {
        private const double cPixelSizeInMeters = 1.70e-5;
        private static CPanel _panel = new CPanel(0, 0, cPixelSizeInMeters, cPixelSizeInMeters);
        private static LoggingThread logger = new LoggingThread(null);
        private static PositionCalibrator _positionCalibrator = null;
        private static ManualResetEvent _calDoneEvent = new ManualResetEvent(false);
        /// <summary>
        /// Use SIM to load up an image set and run it through the stitch tools...
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Start the logger
            logger.Start("Logger", @"c:\\", "SimCalibratorTester.log", true, -1);

            // Gather input data.
            string simulationFile = "";
            string panelFile = "";
            for (int i = 0; i < args.Length; i++)
            {
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

            bool bSimulating = false;
            if (!string.IsNullOrEmpty(simulationFile) && File.Exists(simulationFile))
                bSimulating = true;

            _positionCalibrator = new PositionCalibrator(_panel, ManagedCoreAPI.GetDevice(0),
                bSimulating);

            _positionCalibrator.CalibrationComplete += CalibrationComplete;
/*            Bitmap bmp = posCal.AquireRowImage();

            if (bmp == null)
            {
                Output("Row Image is not valid!");
            }
            else
            {
                bmp.Save("c:\\temp\\rowImage.png");
            }
*/
            _positionCalibrator.StartAcquisition();

            _calDoneEvent.WaitOne();

            Output("Processing Complete");
            logger.Kill();
            ManagedCoreAPI.TerminateAPI();
        }

        private static void CalibrationComplete(CalibrationStatus status)
        {
            Output("Calibration Completed with status " + status);

            if(status == CalibrationStatus.CalibrationNotInTolerance)
            {
                string msg = string.Format("XOffset={0}, YOffset={1}, Velocity={2}",
                                           _positionCalibrator.GetXOffsetInMeters(),
                                           _positionCalibrator.GetYOffsetInMeters(),
                                           _positionCalibrator.GetVelocityOffsetInMetersPerSecond());
                Output(msg);
            }

            _calDoneEvent.Set();
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
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }
    }
}
