using System;
using System.IO;
using System.Reflection;
using System.Threading;
using MPanelIO;
using Cyber.DiagnosticUtils;
using Cyber.MPanel;
using MCoreAPI;
using MLOGGER;
using SIMCalibrator;

namespace SIMCalibratorTester
{
    /// <summary>
    /// This program can be used to do positional calibration on a SIM Device.  It has two purposes:
    /// 1)  Perform and test calibration.
    /// 2)  Demonstrate how to use calibration.
    /// </summary>
    class Program
    {
        private const double cPixelSizeInMeters = 1.70e-5;
        private static CPanel _panel = new CPanel(0, 0, cPixelSizeInMeters, cPixelSizeInMeters);
        private static LoggingThread logger = new LoggingThread(null);
        private static PositionCalibrator _positionCalibrator = null;
        private static ManualResetEvent _calDoneEvent = new ManualResetEvent(false);
        private static bool isColor = false;
        
        static void Main(string[] args)
        {
            // Gather input data.
            string simulationFile = "";
            string panelFile = "";
            double xFidSearchArea = .008;
            double yFidSearchArea = .008;
            bool WaitForKeyboardInput = false;
            bool LoggingActive = false;
            int deviceIndex = 0;

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-s" && i < args.Length - 1)
                    simulationFile = args[i + 1];
                else if (args[i] == "-p" && i < args.Length - 1)
                    panelFile = args[i + 1];
                else if (args[i] == "-fx" && i < args.Length - 1)
                    xFidSearchArea = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-fy" && i < args.Length - 1)
                    yFidSearchArea = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-d" && i < args.Length - 1)
                    deviceIndex = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-k")
                    WaitForKeyboardInput = true;
                else if (args[i] == "-l")
                    LoggingActive = true;
                else if (args[i] == "-c")
                    isColor = true;
                else if (args[i] == "-h")
                {
                    ShowHelp();
                    return;
                }
            }

            // Start the logger
            logger.Start("Logger", @"c:\\", "SimCalibratorTester.log", true, -1);
            Output("SIMAPICalibratorTester Version:" + Assembly.GetExecutingAssembly().GetName().Version);

            // Setup the panel based on panel file
            if (!ChangeProductionFile(panelFile))
            {
                ShowHelp();
                logger.Kill();
                return;
            }

            // Initialize the SIM CoreAPI
            if (!InitializeSimCoreAPI(simulationFile))
            {
                logger.Kill();
                return;
            }

            if(deviceIndex < 0 || deviceIndex >= ManagedCoreAPI.NumberOfDevices())
            {
                Output("The DeviceIndex is not in range of the number of devices: " + ManagedCoreAPI.NumberOfDevices());
                logger.Kill();
                return;       
            }

            bool bSimulating = false;
            if (!string.IsNullOrEmpty(simulationFile) && File.Exists(simulationFile))
                bSimulating = true;

            _positionCalibrator = new PositionCalibrator(_panel, ManagedCoreAPI.GetDevice(deviceIndex),
                bSimulating, xFidSearchArea, yFidSearchArea, LoggingActive, isColor);

            _positionCalibrator.LogEvent += OnLogEntryFromClient;

            Output("SIM SETTINGS BEFORE CALIBRATION");
            OutputSIMCalibrationSettings(deviceIndex);

            _positionCalibrator.CalibrationComplete += CalibrationComplete;
            _positionCalibrator.StartAcquisition();
            _calDoneEvent.WaitOne();

            Output("SIM SETTINGS AFTER CALIBRATION");
            OutputSIMCalibrationSettings(deviceIndex);

            Output("Processing Complete");

            if (WaitForKeyboardInput)
            {
                Console.WriteLine("Press any key to terminate the program.");
                Console.ReadKey();
            }

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

        private static void Output(string str)
        {
            logger.AddObjectToThreadQueue(str);
            logger.AddObjectToThreadQueue(null);
        }

        private static void OutputSIMCalibrationSettings(int device)
        {
            ManagedSIMDevice d = ManagedCoreAPI.GetDevice(device);
            if (d == null)
                return;

            Output("Home Offset:  " + d.HomeOffset);
            Output("Y Offset:  " + d.YOffset);
            Output("Conveyor Velocity:  " + d.ConveyorVelocity);
        }

        static void ShowHelp()
        {
            Console.WriteLine("SIMCalibratorTester Command Line Arguments:");
            Console.WriteLine("=========================================================================");
            Console.WriteLine("-c Used if color calibration...");
            Console.WriteLine("-d <int> optional device index. Defaults to 0");
            Console.WriteLine("-fx <double> optional X fiducial search area in meters. Defaults to .008");
            Console.WriteLine("-fy <double> optional Y fiducial search area in meters. Defaults to .008");
            Console.WriteLine("-k to wait for keyboard input when complete.  Defaults to false");
            Console.WriteLine("-l Activate Logging.  Defaults to false");
            Console.WriteLine("-p <panelFile> required panel file path (xml or srf).");
            Console.WriteLine("-s <simulationFile> optional xml simulation file.");
            Console.WriteLine("-h to show this help");
            Console.WriteLine("NOTE:  Panel file must contain at least one fiducial!");
        }
    }
}