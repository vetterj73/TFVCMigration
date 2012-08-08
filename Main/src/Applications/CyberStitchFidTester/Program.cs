using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Threading;
using Cyber.ImageUtils;
using MPanelIO;
using Cyber.DiagnosticUtils;
using Cyber.MPanel;
using MCoreAPI;
using MLOGGER;
using MMosaicDM;
using PanelAlignM;
using SIMMosaicUtils;

namespace CyberStitchFidTester
{
    /// <summary>
    /// The idea behind this program is to run regression tests on CyberStitch.  
    /// The Concept:
    /// 1)  Input a SIM Simulation set, a panel file with a small number of fids and a test file with a large number of fids.
    /// 2)  Output a text file that gives you an indication of how far off the fiducials are in the stitched image 
    /// compared to known locations.
    /// 3)  Extra Credit:  This runs as part of the nightly build.  Each night, we can compare a magic number (the average distance offset)
    /// to the last magic number.  If the Magic Number is increasing, we should be nervous because something we changed is causing problems.
    /// NOTE:  I added the ability to bypass CyberStitch by sending in an image file directly.  This causes some headaches because the image
    /// file may be built with other tools (2dSPI for instance) and therefore may have a different pixel size and stride.  This muddied the 
    /// water, but it should be handled correctly.
    /// </summary>
    class Program
    {
        private const string headerLine = "Panel#, Fid#, X, Y ,XOffset, YOffset, CorrScore, Ambig";

        private static double dPixelSizeInMeters = 1.70e-5;
        private static uint iInputImageColumns = 2592;
        private static uint iInputImageRows = 1944;
        private static ManagedMosaicSet _mosaicSet = null;
        private static CPanel _processingPanel = null;
        private static CPanel _fidPanel = null;
        private readonly static ManualResetEvent mCollectedEvent = new ManualResetEvent(false);
        private readonly static ManualResetEvent mAlignedEvent = new ManualResetEvent(false);
        private static int numAcqsComplete = 0;
        private static ManagedPanelAlignment _aligner = new ManagedPanelAlignment();
        private static LoggingThread logger = new LoggingThread(null);
        private static int _cycleCount = 0;
        private static double _allPanelFidDifference = 0.0;
        private static double _dXDiffSqrSumTol = 0.0;//used for the total Xoffset square sum
        private static double _dYDiffSqrSumTol = 0.0;
        private static double _dXRMS = 0.0;//used for the xoffset RMS
        private static double _dYRMS = 0.0;
        private static bool _bDetectPanelEdge = false;
        private static bool _bRtoL = false; // right to left conveyor direction indicator
        private static bool _bFRR = false; // fixed rear rail indicator
        private static bool _bSkipDemosaic = false; // true: skip demosaic for bayer image

        // For debug
        private static int _iBufCount = 0;
        private static bool _bSimulating = false;
        private static bool _bBayerPattern = false;
        private static int _iBayerType = 1; // GBRG
        private static StreamWriter writer = null;
        private static StreamWriter finalCompWriter = null;

        // For output analysis
        private static double[] _dXDiffSum;
        private static double[] _dYDiffSum;
        private static double[] _dXAbsDiffSum;
        private static double[] _dYAbsDiffSum;
        private static double[] _dXDiffSqrSum;
        private static double[] _dYDiffSqrSum;        
        private static int[] _icycleCount;
        private static int _iTotalCount = 0;

        //For time stamp
        private static DateTime _dtStartTime;
        private static DateTime _dtEndTime;
        private static double _tsRunTime = 0; 
        private static double _tsTotalRunTime = 0;
       
        /// <summary>
        /// This works similar to CyberStitchTester.  The differences:
        /// 1)  All images are obtained prior to running through CyberStitch.
        /// 2)  Color images are pre-converted to illum images for faster processing.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // Gather input data.
            string simulationFile = "";
            string panelFile = "";
            string fidPanelFile = "";
            bool bUseProjective = false;
            bool bUseCameraModel = false;
            bool bSaveStitchedResultsImage = false;
            bool bUseIterativeCameraModel = false;
            bool bUseTwoPassStitch = false;
            int numberToRun = 1;
            string unitTestFolder="";
            double dCalScale = 1.0;
            int iLayerIndex4Edge = 0;
            bool bMaskForDiffDevices = false;
            ManagedFeatureLocationCheck fidChecker;
         
            //output csv file shows the comparison results
            string outputTextPath = @".\fidsCompareResults.csv";
            string lastOutputTextPath = @".\lastFidsCompareResults.csv";
            string imagePathPattern = "";
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-b")
                    _bBayerPattern = true;
                else if (args[i] == "-f" && i < args.Length - 1)
                    fidPanelFile = args[i + 1];
                else if (args[i] == "-i" && i < args.Length - 1)
                    imagePathPattern = args[i + 1];
                else if (args[i] == "-l" && i < args.Length - 1)
                    lastOutputTextPath = args[i + 1];
                else if (args[i] == "-m")
                    bMaskForDiffDevices = true;
                else if (args[i] == "-n" && i < args.Length - 1)
                    numberToRun = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-o" && i < args.Length - 1)
                    outputTextPath = args[i + 1];
                else if (args[i] == "-p" && i < args.Length - 1)
                    panelFile = args[i + 1];
                else if (args[i] == "-r")
                    bSaveStitchedResultsImage = true;
                else if (args[i] == "-s" && i < args.Length - 1)
                    simulationFile = args[i + 1];
                else if (args[i] == "-u" && i < args.Length - 1)
                    unitTestFolder = args[i + 1];
                else if (args[i] == "-scale" && i < args.Length - 1)
                    dCalScale = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-w")
                    bUseProjective = true;
                else if (args[i] == "-nw")
                    bUseProjective = false;
                else if (args[i] == "-cammod")
                    bUseCameraModel = true;
                else if (args[i] == "-iter")
                    bUseIterativeCameraModel = true;
                else if (args[i] == "-rtol")
                    _bRtoL = true;
                else if (args[i] == "-frr")
                    _bFRR = true;
                else if (args[i] == "-de")
                    _bDetectPanelEdge = true;
                else if (args[i] == "-bSkipD")
                    _bSkipDemosaic = true;
                else if (args[i] == "-le" && i < args.Length - 1)
                    iLayerIndex4Edge = Convert.ToInt16(args[i + 1]);
                else if (args[i] == "-pixsize" && i < args.Length - 1)
                    dPixelSizeInMeters = Convert.ToDouble(args[i + 1]);
                else if (args[i] == "-imgcols" && i < args.Length - 1)
                    iInputImageColumns = Convert.ToUInt32(args[i + 1]);
                else if (args[i] == "-imgrows" && i < args.Length - 1)
                    iInputImageRows = Convert.ToUInt32(args[i + 1]);
                else if (args[i] == "-twopass")
                    bUseTwoPassStitch = true;

                else if (args[i] == "-h" && i < args.Length - 1)
                {
                    ShowHelp();
                    return;
                }
            }

            // Start the logger
            logger.Start("Logger", @"c:\\", "CyberStitch.log", true, -1);
            logger.AddObjectToThreadQueue("CyberStitchFidTester Version: " + Assembly.GetExecutingAssembly().GetName().Version);

            // Open output text file.
            writer = new StreamWriter(outputTextPath);

            Output("The report file: " + outputTextPath);

            string[] imagePath = ExpandFilePaths(imagePathPattern);
            bool bImageOnly = false;
            if(imagePath.Length > 0)
                bImageOnly = true;

            if (!bImageOnly && File.Exists(panelFile))
            {
                _processingPanel = LoadProductionFile(panelFile, dPixelSizeInMeters);
                if (_processingPanel == null)
                {
                    Terminate(false);
                    Console.WriteLine("Could not load Panel File: " + panelFile);
                    return;
                }
            } 
            else if (!bImageOnly) // Panel file must exist...
            {
                Terminate(false);
                Console.WriteLine("The panel file does not exist..: " + panelFile);
                return;
            }

            Bitmap inputBmp = null;
            if(bImageOnly)
                inputBmp = new Bitmap(imagePath[0]);  // should be safe, as bImageOnly == (imagePath.Length > 0)

            int ifidsNum = 0;
            double pixelSize = dPixelSizeInMeters;
            if (File.Exists(fidPanelFile))
            {
                _fidPanel = LoadProductionFile(fidPanelFile, dPixelSizeInMeters);
                if (bImageOnly && _fidPanel.GetNumPixelsInY() != inputBmp.Width)
                {
                    pixelSize = _fidPanel.PanelSizeY/inputBmp.Width;
                    _fidPanel = LoadProductionFile(fidPanelFile, pixelSize);
                }
                if (_fidPanel == null)
                {
                    Terminate(false);
                    Console.WriteLine("Could not load Fid Test File: " + fidPanelFile);
                    return;
                }
                fidChecker = new ManagedFeatureLocationCheck(_fidPanel);
                ifidsNum = _fidPanel.NumberOfFiducials;
                _dXDiffSum = new double[ifidsNum];
                _dYDiffSum = new double[ifidsNum];
                _dXAbsDiffSum = new double[ifidsNum];
                _dYAbsDiffSum = new double[ifidsNum];
                _dXDiffSqrSum = new double[ifidsNum];
                _dYDiffSqrSum = new double[ifidsNum];
                _icycleCount = new int[ifidsNum];

                for (int i = 0; i < ifidsNum; i++)
                {
                    _dXDiffSum[i] = 0;
                    _dYDiffSum[i] = 0;
                    _dXAbsDiffSum[i] = 0;
                    _dYAbsDiffSum[i] = 0;
                    _dXDiffSqrSum[i] = 0;
                    _dYDiffSqrSum[i] = 0;
                    _icycleCount[i] = 0;
                }
            }
            else
            {
                Terminate(false);
                Console.WriteLine("Fid Test File does not exist: " + fidPanelFile);
                return;
            }

            if (bImageOnly)
            {
                int cycleId = 0; // Image already loaded.

                writer.WriteLine(headerLine);

                while (inputBmp != null)
                {
                    Console.WriteLine("Comparing fiducials on {0}...", imagePath[cycleId]);

                    // This allows images to directly be sent in instead of using CyberStitch to create them
                    CyberBitmapData cbd = new CyberBitmapData();

                    cbd.Lock(inputBmp);
                    fidChecker = new ManagedFeatureLocationCheck(_fidPanel);
                    RunFiducialCompare(cbd.Scan0, cbd.Stride, writer, fidChecker);
                    if (bSaveStitchedResultsImage)
                    {
                        string imageFilename = "FidCompareImage-" + cycleId + ".bmp";
                        _aligner.Save3ChannelImage(imageFilename,
                                             cbd.Scan0, cbd.Stride,
                                              _fidPanel.GetCADBuffer(), _fidPanel.GetNumPixelsInY(),
                                              _fidPanel.GetCADBuffer(), _fidPanel.GetNumPixelsInY(),
                                              _fidPanel.GetNumPixelsInY(), _fidPanel.GetNumPixelsInX());
                    }
                    cbd.Unlock();

                    //
                    // Increment cycle and get bitmap, if available
                    cycleId++;

                    if (cycleId < imagePath.Length)
                        inputBmp = new Bitmap(imagePath[cycleId]);
                    else
                        inputBmp = null;
                }
            } // End Stitched Image Inspection Mode
            else
            {
                // Initialize the SIM CoreAPI
                if (!InitializeSimCoreAPI(simulationFile))
                {
                    Terminate(false);
                    Console.WriteLine("Could not initialize Core API");
                    return;
                }

                // Must after InitializeSimCoreAPI() before ChangeProduction()
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(0);
                _aligner.SetPanelEdgeDetection(_bDetectPanelEdge, iLayerIndex4Edge, !d.ConveyorRtoL, !d.FixedRearRail); 

                // Set up mosaic set
                SetupMosaic(true, bMaskForDiffDevices);

                try
                {
                    Output("Aligner ChangeProduction");
                    // Setup Aligner...
                    _aligner.OnLogEntry += OnLogEntryFromClient;
                    _aligner.SetAllLogTypes(true);
                    _aligner.OnAlignmentDone += OnAlignmentDone;
                    _aligner.NumThreads(8);
                    _aligner.UseProjectiveTransform(bUseProjective);
                    if (dCalScale != 1.0)
                        _aligner.SetCalibrationWeight(dCalScale);
                    
                    if (bUseTwoPassStitch)
                        _aligner.SetUseTwoPassStitch(true);
                    if (bUseCameraModel)
                    {
                        _aligner.UseCameraModelStitch(true);
                        _aligner.UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
                    }

                    if (bUseIterativeCameraModel)
                    {
                        _aligner.UseCameraModelIterativeStitch(true);
                        _aligner.UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
                    }

                    // Always seperate acquistion, demosaic and alignment stages 
                    _mosaicSet.SetSeperateProcessStages(true);

                    // true: Skip demosaic for Bayer image
                    if(_bBayerPattern)
                        _aligner.SetSkipDemosaic(_bSkipDemosaic);


                    // Add trigger to trigger overlaps for same layer
                    //for (uint i = 0; i < _mosaicSet.GetNumMosaicLayers(); i++)
                    //    _mosaicSet.GetCorrelationSet(i, i).SetTriggerToTrigger(true);

                    if (!_aligner.ChangeProduction(_mosaicSet, _processingPanel))
                    {
                        throw new ApplicationException("Aligner failed to change production ");
                    }
                }
                catch (Exception except)
                {
                    Output("Error Changing Production: " + except.Message);
                    Terminate(true);
                    return;
                }

                try
                {
                    RunAcquireAndStitch(numberToRun, fidChecker, bSaveStitchedResultsImage);
                }
                catch (Exception except)
                {
                    Output("Error during acquire and stitch: " + except.Message);
                    Terminate(true);
                    return;
                }

            } // End Simulation Mode

            // Last attempt to force proper shutdown...
            try
            {
                WriteResults(lastOutputTextPath, outputTextPath, unitTestFolder);
                Output("Processing Complete");
            }
            catch (Exception except)
            {
                Output("Error during write results: " + except.Message);
            }
            finally
            {
                Terminate(!bImageOnly);           
            }
        }

        private static void Terminate(bool bTerminateCore)
        {
            if (writer != null)
                writer.Close();

            if (logger != null)
                logger.Kill();

            _aligner.Dispose();       

            if(bTerminateCore)
                ManagedCoreAPI.TerminateAPI();
        }

        private static void RunAcquireAndStitch(int numberToRun, ManagedFeatureLocationCheck fidChecker, bool bSaveStitchedResultsImage)
        {
            bool bDone = false;
            while (!bDone)
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
                    mCollectedEvent.WaitOne();
                }

                // Verify that mosaic is filled in...
                if (!_mosaicSet.HasAllImages())
                    Output("The mosaic does not contain all images!");
                else
                {
                    _cycleCount++;
                    mAlignedEvent.WaitOne();
                    _tsRunTime = _aligner.GetAlignmentTime();
                    _tsTotalRunTime += _tsRunTime;

                    ManagedPanelFidResultsSet set = _aligner.GetFiducialResultsSet();
                    Output("Panel Skew is: " + set.dPanelSkew);
                    Output("Panel dPanelXscale is: " + set.dPanelXscale);
                    Output("Panel dPanelYscale is: " + set.dPanelYscale);

                    uint iIndex1 = 0;
                    uint iIndex2 = 1;
                    if (_mosaicSet.GetNumMosaicLayers() == 1)
                        iIndex2 = 0;

                    if (bSaveStitchedResultsImage)
                        _aligner.Save3ChannelImage("c:\\Temp\\FidCompareAfterCycle" + _cycleCount + ".bmp",
                                               _mosaicSet.GetLayer(iIndex1).GetGreyStitchedBuffer(), _processingPanel.GetNumPixelsInY(),
                                               _mosaicSet.GetLayer(iIndex2).GetGreyStitchedBuffer(), _processingPanel.GetNumPixelsInY(),
                                               _fidPanel.GetCADBuffer(), _processingPanel.GetNumPixelsInY(),
                                               _fidPanel.GetNumPixelsInY(), _fidPanel.GetNumPixelsInX());
                    if (_cycleCount == 1)
                    {
                        writer.WriteLine("Units: Microns");
                        //outline is the output file column names
                        writer.WriteLine(headerLine);
                    }

                    if (_fidPanel != null)
                        RunFiducialCompare(_mosaicSet.GetLayer(0).GetGreyStitchedBuffer(), _fidPanel.GetNumPixelsInY(), writer, fidChecker);
                }
                if (_cycleCount >= numberToRun)
                    bDone = true;
                else
                {
                    mCollectedEvent.Reset();
                    mAlignedEvent.Reset();
                }
            }
        }

        private static void WriteResults(string lastOutputTextPath, string outputTextPath, string unitTestFolder)
        {
            writer.WriteLine(" Fid#, XOffset Mean, YOffset Mean,XOffset Stdev, YOffset Stdev, Absolute XOffset Mean, Absolute YOffset Mean, Absolute XOffset Stdev, Absolute YOffset Stdev, Number of cycle ");
            for (int i = 0; i < _fidPanel.NumberOfFiducials; i++)
            {
                if (_icycleCount[i] == 0)   // If no fiducial is found
                {
                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", i,
                            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", _icycleCount[i]));
                }
                else
                {
                    double dMeanX = _dXDiffSum[i] / _icycleCount[i];
                    double dMeanY = _dYDiffSum[i] / _icycleCount[i];
                    double dSdvX = Math.Sqrt(_dXDiffSqrSum[i] / _icycleCount[i] - dMeanX * dMeanX);
                    double dSdvY = Math.Sqrt(_dYDiffSqrSum[i] / _icycleCount[i] - dMeanY * dMeanY);
                    double dAbsMeanX = _dXAbsDiffSum[i] / _icycleCount[i];
                    double dAbsMeanY = _dYAbsDiffSum[i] / _icycleCount[i];
                    double dAbsSdvX = Math.Sqrt(_dXDiffSqrSum[i] / _icycleCount[i] - dAbsMeanX * dAbsMeanX);
                    double dAbsSdvY = Math.Sqrt(_dYDiffSqrSum[i] / _icycleCount[i] - dAbsMeanY * dAbsMeanY);
                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", i,
                        dMeanX, dMeanY, dSdvX, dSdvY,
                        dAbsMeanX, dAbsMeanY, dAbsSdvX, dAbsSdvY, 
                        _icycleCount[i]));
                }
                _iTotalCount += _icycleCount[i];

            }
            _dXRMS = Math.Sqrt(_dXDiffSqrSumTol / (_iTotalCount));
            _dYRMS = Math.Sqrt(_dYDiffSqrSumTol / (_iTotalCount));

            writer.WriteLine(String.Format("Average Panel Process Running time(Unites:Minutes): {0}", _tsTotalRunTime/60/_cycleCount));
            writer.WriteLine(string.Format("MagicNumber: {0}, Xoffset RMS:{1}, Yoffset RMS:{2}", _allPanelFidDifference, _dXRMS, _dYRMS));
            writer.WriteLine(string.Format("Average Offset: {0}", _allPanelFidDifference / _iTotalCount));

            if (File.Exists(lastOutputTextPath))
            {
                string[] lines = File.ReadAllLines(lastOutputTextPath);
                if (lines.Length >= 1)
                {
                    string lastLine = lines[lines.Length - 1];
                    string timeRecordLine = lines[lines.Length - 3];
                    string[] averageParts = lastLine.Split(':');
                    string[] timeParts = timeRecordLine.Split(':');
                    double lastAverage = 0;
                    double lastTimeRecord = 0;
                    bool bGood = false;
                    bool bGoodAver = false;
                    bool bGoodTime = false;
                    bool bHeadLine = true;
                    const string headLine =
                        "Test, Previous Average Offset(Microns),  Current Average Offset(Microns),Previous Average Panel Process Running time(Minutes),Current Average Panel Process Running time(Minutes), Test Result";
                    string finalCompCSVPath = Path.Combine(Path.GetDirectoryName(outputTextPath),
                                                           "FinalCompareResults.csv");
                    string testName = Path.GetFileNameWithoutExtension(outputTextPath);
                    string testResult;
                    if (File.Exists(finalCompCSVPath))
                        bHeadLine = false;
                    finalCompWriter = new StreamWriter(finalCompCSVPath, true);
                    if (averageParts.Length > 1 && double.TryParse(averageParts[1], out lastAverage))
                    {
                        // Check that we are at least as good as last time (to the nearest micron)
                        if (Math.Round(_allPanelFidDifference/_iTotalCount) <= Math.Round(lastAverage))
                            bGoodAver = true;
                    }
                    if (timeParts.Length > 1 && double.TryParse(timeParts[2], out lastTimeRecord))
                    {
                        // Check that we are at least as good as last time(in 20% range)
                        if (_tsTotalRunTime/60/_cycleCount <= lastTimeRecord*1.2)
                            bGoodTime = true;
                    }
                    bGood = bGoodAver && bGoodTime;
                    testResult = (bGood ? "Passed" : "Failed");
                    Console.WriteLine("Are we as good as last time: " + (bGood ? "Yes!" : "No!"));
                    if (bHeadLine) finalCompWriter.WriteLine(headLine);
                    finalCompWriter.WriteLine(string.Format("{0},{1},{2},{3},{4},{5}", testName, lastAverage,
                                                            _allPanelFidDifference/_iTotalCount, lastTimeRecord,
                                                            _tsTotalRunTime/60/_cycleCount, testResult));
                    if (finalCompWriter != null)
                        finalCompWriter.Close();
                    if (Directory.Exists(unitTestFolder))
                    {
                        string file =
                            Path.Combine(unitTestFolder + Path.GetFileNameWithoutExtension(lastOutputTextPath)) + ".xml";
                        NUnitXmlWriter.WriteResult(file, "CyberStitchFidTester", "AverageOffset", bGood);
                    }
                }
            }
        }

        private static void ShowHelp()
        {
            logger.AddObjectToThreadQueue("CyberStitchFIDTester Command line Options");
            logger.AddObjectToThreadQueue("*****************************************");
            logger.AddObjectToThreadQueue("-b // if bayer pattern");
            logger.AddObjectToThreadQueue("-f <FidTestPanel.xml>");
            logger.AddObjectToThreadQueue("-i <imagePath> instead of running cyberstitch");
            logger.AddObjectToThreadQueue("-l <lastResultsDirectory>");
            logger.AddObjectToThreadQueue("-h Show Help");
            logger.AddObjectToThreadQueue("-l <lastOutput.txt>");
            logger.AddObjectToThreadQueue("-n <#> // Number of panels - defaults to 1");
            logger.AddObjectToThreadQueue("-o <output.txt>");
            logger.AddObjectToThreadQueue("-p <panelfile.xml>");
            logger.AddObjectToThreadQueue("-r // Save stitched results image (c:\\temp\\*.bmp)");
            logger.AddObjectToThreadQueue("-s <SimScenario.xml>");
            logger.AddObjectToThreadQueue("-u <UnitTestFolder>");
            logger.AddObjectToThreadQueue("-w // if projective transform is desired");
            logger.AddObjectToThreadQueue("-----------------------------------------");
        }

        static string[] ExpandFilePaths(string filePathPattern)
        {
            List<string> fileList = new List<string>();

            if (filePathPattern.Length > 0)
            {
                string substitutedFilePathPattern = System.Environment.ExpandEnvironmentVariables(filePathPattern);

                string directory = Path.GetDirectoryName(substitutedFilePathPattern);
                if (directory.Length == 0)
                    directory = ".";

                string filePattern = Path.GetFileName(substitutedFilePathPattern);

                foreach (string filePath in Directory.GetFiles(directory, filePattern))
                    fileList.Add(filePath);
            }

            return fileList.ToArray();
        }

        private static void RunFiducialCompare(IntPtr data, int stride, StreamWriter writer, ManagedFeatureLocationCheck fidChecker)
        {
            int iFidNums = _fidPanel.NumberOfFiducials;

            // Cad_x, cad_y, Loc_x, Loc_y, CorrScore, Ambig 
            int iItems = 6;
            double[] dResults = new double[iFidNums*iItems];
            double xDifference = 0;
            double yDifference = 0;
            double fidDifference = 0;
            string sNofid = "N/A";
            //convert meters to microns
            int iUnitCoverter = 1000000;
            // Find fiducial on the board
            fidChecker.CheckFeatureLocation(data, stride, dResults);
            //Record the processing time
            //DateTime dtEndTime = DateTime.Now;
            //TimeSpan tsRunTime = dtEndTime - _dtStartTime;

            for (int i = 0; i < iFidNums; i++)
            {
                if (dResults[i * iItems + 4] == 0 || dResults[i * iItems + 5] == 1) // Fiducial not found
                {
                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7}",
                        _cycleCount, i,
                        dResults[i * iItems] * iUnitCoverter, dResults[i * iItems + 1] * iUnitCoverter,
                        sNofid, sNofid,
                        dResults[i * iItems + 4], dResults[i * iItems + 5]));
                }
                else
                {
                    _icycleCount[i]++;
                    xDifference = (dResults[i * iItems] - dResults[i * iItems + 2]) * iUnitCoverter;
                    yDifference = (dResults[i * iItems + 1] - dResults[i * iItems + 3]) * iUnitCoverter;
                    fidDifference += Math.Sqrt(xDifference * xDifference + yDifference * yDifference);
                    
                    _dXDiffSum[i] += xDifference;
                    _dYDiffSum[i] += yDifference;
                    _dXAbsDiffSum[i] += Math.Abs(xDifference);
                    _dYAbsDiffSum[i] += Math.Abs(yDifference);
                    _dXDiffSqrSum[i] += Math.Pow(xDifference, 2);
                    _dYDiffSqrSum[i] += Math.Pow(yDifference, 2);

                    writer.WriteLine(
                        string.Format("{0},{1},{2},{3},{4},{5},{6},{7}", 
                        _cycleCount, i,
                        dResults[i * iItems] * iUnitCoverter, dResults[i * iItems + 1] * iUnitCoverter, 
                        xDifference, yDifference,
                        dResults[i * iItems + 4], dResults[i * iItems + 5]));
                    _dXDiffSqrSumTol += Math.Pow(xDifference, 2);
                    _dYDiffSqrSumTol += Math.Pow(yDifference, 2);
                }
            }

            writer.WriteLine("Total Difference for this Panel: " + fidDifference);
            _allPanelFidDifference += fidDifference;
            writer.WriteLine(string.Format("Panel Process Start Time: {0}, Panel Processing end time: {1},Panel process running time: {2}" ,_dtStartTime,_dtEndTime,_tsRunTime));
           // _tsTotalRunTime += tsRunTime;
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

                    if (desiredCount != bufferCount)
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

                    ManagedSIMCaptureSpec cs1 = d.SetupCaptureSpec(_processingPanel.PanelSizeX, _processingPanel.PanelSizeY, 0, .004);
                    if (cs1 == null)
                    {
                        Output("Could not create capture spec.");
                        return false;
                    }
                }
            }
            return true;
        }

        private static CPanel LoadProductionFile(string panelFile, double pixelSize)
        {
            CPanel panel = null;
            if (!string.IsNullOrEmpty(panelFile))
            {
                try
                {
                    if (panelFile.EndsWith(".srf", StringComparison.CurrentCultureIgnoreCase))
                    {
                        panel = SRFToPanel.parseSRF(panelFile, pixelSize, pixelSize);
                        if (panel == null)
                            throw new ApplicationException("Could not parse the SRF panel file");
                    }
                    else if (panelFile.EndsWith(".xml", StringComparison.CurrentCultureIgnoreCase))
                    {
                        panel = XmlToPanel.CSIMPanelXmlToCPanel(panelFile, pixelSize, pixelSize);
                        if (panel == null)
                            throw new ApplicationException("Could not convert xml panel file");
                    }

                    return panel;
                }
                catch (Exception except)
                {
                    Output("Exception reading Panel file: " + except.Message);
                    logger.Kill();
                }
            }
            return panel;
        }

        private static void OnLogEntryFromClient(MLOGTYPE logtype, string message)
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
            _mosaicSet = new ManagedMosaicSet(
                _processingPanel.PanelSizeX, _processingPanel.PanelSizeY, 
                iInputImageColumns, iInputImageRows, iInputImageColumns, 
                dPixelSizeInMeters, dPixelSizeInMeters, 
                bOwnBuffers, 
                _bBayerPattern, _iBayerType, _bSkipDemosaic);
            _mosaicSet.OnLogEntry += OnLogEntryFromMosaic;
            _mosaicSet.SetLogType(MLOGTYPE.LogTypeDiagnostic, true);
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
            // launch next device in simulation case
            if (_bSimulating && numAcqsComplete < ManagedCoreAPI.NumberOfDevices())
            {
                ManagedSIMDevice d = ManagedCoreAPI.GetDevice(numAcqsComplete);
                if (d.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                    return;
            }
            if (ManagedCoreAPI.NumberOfDevices() == numAcqsComplete)
                mCollectedEvent.Set();
        }

        private static void OnAlignmentDone(bool status)
        {
            Output("OnAlignmentDone Called!");
            mAlignedEvent.Set();
        }

        private static void OnFrameDone(ManagedSIMFrame pframe)
        {
            // Output(string.Format("Got an Image:  Device:{0}, ICS:{1}, Camera:{2}, Trigger:{3}",
            //     pframe.DeviceIndex(), pframe.CaptureSpecIndex(), pframe.CameraIndex(), pframe.TriggerIndex()));
            _iBufCount++; // for debug

            int device = pframe.DeviceIndex();
            int mosaic_row = SimMosaicTranslator.TranslateTrigger(pframe);
            int mosaic_column = pframe.CameraIndex() - ManagedCoreAPI.GetDevice(device).FirstCameraEnabled;

            uint layer = (uint)(device * ManagedCoreAPI.GetDevice(device).NumberOfCaptureSpecs +
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
