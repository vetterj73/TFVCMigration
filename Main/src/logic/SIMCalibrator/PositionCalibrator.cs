using System;
using System.Drawing;
using System.Threading;
using Cyber.ImageUtils;
using Cyber.MPanel;
using MCoreAPI;
using MLOGGER;
using MMosaicDM;
using PanelAlignM;
using SIMMosaicUtils;

namespace SIMCalibrator
{
    /// <summary>
    /// When the client starts a calibration acquisition, there are 4 possible outcomes:
    /// </summary>
    public enum CalibrationStatus
    {
        AquisitionFailed,            // Could not get images
        CalibrationNotLegitimate,    // Got images, but can't find fiducials from the panel input
        CalibrationNotInTolerance,   // Found fiducials out of tolerance
        CalibrationInTolerance,      // Found fiducials in tolerance (no further work needed).
    }

    public delegate void CalibrationComplete(CalibrationStatus status);

    /// <summary>
    /// This class is used to adjust the positional calibration of a SIM device based on collected images
    /// of a panel.  For the purposes of this class, the Panel can be any CPanel Object.  However, 
    /// keep in mind that fiducials are the only thing this class uses to determine positional calibration.  
    /// Also, keep in mind that results will vary based on the input panel (i.e. - this class does the best 
    /// it can with what it is provided, if you give it a bad calibration target, you will likely get bad
    /// results).
    /// XOffset, YOffset and Conveyor Speed are calibrated with this class.
    /// </summary>
    public class PositionCalibrator
    {
        private const double cPixelSizeInMeters = 1.70e-5;
        private const double cYInTolerance = .0005;
        private const double cXInTolerance = .0005;
        
        private CPanel _panel;
        private ManagedSIMDevice _device;
        private ManagedMosaicSet _mosaicSet;
        private readonly static ManualResetEvent _doneEvent = new ManualResetEvent(false);
        private uint _layerIndex = 0;
        private ManagedPanelAlignment _panelAligner;

        private double _yOffsetInMeters=0.0;
        private double _xOffsetInMeters = 0.0;
        private double _velocityOffsetInMetersPerSecond = 0.0;
        private CalibrationStatus _calibrationStatus = CalibrationStatus.AquisitionFailed;
        private bool _waitingForImages = false;
        private ManagedFidInfo _fidClosestToLeadingEdge;
        private ManagedFidInfo _fidFarthestToLeadingEdge;


        /// <summary>
        /// Fired after images are acquired and calibration is verified.
        /// </summary>
        public event CalibrationComplete CalibrationComplete;

        /// <summary>
        /// Constructor:  Given a valid CPanel Object and a valid SIM Device
        /// </summary>
        /// <param name="panel"></param>
        /// <param name="device"></param>
        /// <param name="bSimulating"></param>
        public PositionCalibrator(CPanel panel, ManagedSIMDevice device, bool bSimulating)
        {
            if (panel == null)
                throw new ApplicationException("The input panel is null!");

            if (device == null)
                throw new ApplicationException("The input device is null!");

            // Events fired for images.
            ManagedSIMDevice.OnFrameDone += FrameDone;
            ManagedSIMDevice.OnAcquisitionDone += AcquisitionDone;

            _panel = panel;
            _device = device;
            SetupCaptureSpecs(bSimulating);

            // Sets up the mosaic from the device...
            SetupMosaic();

            _panelAligner = new ManagedPanelAlignment();
            _panelAligner.OnLogEntry += OnLogEntryFromClient;
            _panelAligner.SetAllLogTypes(true);
            _panelAligner.NumThreads(8);
            _panelAligner.FiducialSearchExpansionXInMeters(.008); // 8 mm in X
            _panelAligner.FiducialSearchExpansionYInMeters(.008); // 8 mm in Y
            _panelAligner.LogFiducialOverlaps(true);
            _panelAligner.ChangeProduction(_mosaicSet, _panel);
        }

        /// <summary>
        /// Acquire a row image with the device.  This will return null if the device is
        /// null.
        /// </summary>
        /// <returns></returns>
        public Bitmap AquireRowImage()
        {        
            // If we started with a mosaic - just return the first row in the mosaic...
            if (_device == null)
                return StitchRowImageFromMosaic();

            _waitingForImages = true;
            _doneEvent.Reset();
            if (_device.StartAcquisition(ACQUISITION_MODE.SINGLE_TRIGGER_MODE) != 0)
                throw new ApplicationException("Could not start a Row Acquisition");

            // Wait for all images to be gathered...
            _doneEvent.WaitOne();

            // Stitch together using basic stitcher (for now).
            return StitchRowImageFromMosaic();
        }

        /// <summary>
        /// Starts the acquisition on the device.  This is an async call.  When the acquisition is
        /// complete, the CalibrationAcquisitionComplete event will be sent to listeners.
        /// </summary>
        public void StartAcquisition()
        {
            if (_device == null)
                return;

            _panelAligner.ResetForNextPanel();

            ResetForAcquisition();

            _waitingForImages = false;
            _doneEvent.Reset();
            if (_device.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE) != 0)
                throw new ApplicationException("Could not start a Panel Capture Acquisition");
        }

        /// <summary>
        /// Currren
        /// </summary>
        private void CalculateCalibrationResults()
        {
            _calibrationStatus = CalibrationStatus.CalibrationNotLegitimate;

            _yOffsetInMeters = 0.0;
            _xOffsetInMeters = 0.0;
            _velocityOffsetInMetersPerSecond = 0.0;
            int numFids = _panelAligner.GetNumberOfFidsProcessed();
            string[] fids = new string[numFids];

            int numFidsUsed = 0;

            for (uint i = 0; i < numFids; i++)
            {
                ManagedFidInfo fidM = _panelAligner.GetFidAtIndex(i);

                if (fidM == null)
                    throw new ApplicationException("Invalid Fid at position " + i);

                if (fidM.CorrelationScore() < .85)
                    continue;

                numFidsUsed++;

                if (_fidClosestToLeadingEdge == null || 
                    _fidClosestToLeadingEdge.GetNominalXPosition()>fidM.GetNominalXPosition())
                    _fidClosestToLeadingEdge = fidM;

                if (_fidFarthestToLeadingEdge == null || 
                    _fidFarthestToLeadingEdge.GetNominalXPosition()<fidM.GetNominalXPosition())
                    _fidFarthestToLeadingEdge = fidM;


                _yOffsetInMeters += fidM.RowDifference() * _mosaicSet.GetNominalPixelSizeY();
                _xOffsetInMeters += fidM.ColumnDifference() * _mosaicSet.GetNominalPixelSizeX();

                fids[i] = string.Format("Fiducial Info: x={0}, y={1}, colOffset={2}, rowOffsetx={3}, score={4}",
                    fidM.GetNominalXPosition(), fidM.GetNominalYPosition(), fidM.ColumnDifference(),
                    fidM.RowDifference(), fidM.CorrelationScore());
            }

            if (numFidsUsed > 0)
            {
                _yOffsetInMeters /= numFidsUsed;
                _xOffsetInMeters /= numFidsUsed;

 //               if (_yOffsetInMeters <= cYInTolerance && _xOffsetInMeters <= cXInTolerance)
   //                 _calibrationStatus = CalibrationStatus.CalibrationInTolerance;
     //           else
                {
                    _calibrationStatus = CalibrationStatus.CalibrationNotInTolerance;
                    AdjustCalibrationBasedOnLastAcquisition();                   
                }
            } 

            System.IO.File.WriteAllLines("c:\\fidInfo.txt", fids);
            FireCalibrationComplete(_calibrationStatus);
        }

        private void OnLogEntryFromClient(MLOGTYPE logtype, string message)
        {
            Console.WriteLine(logtype + ": " + message);
        }

        /// <summary>
        /// Adjust the settings (XOffset, YOffset and Speed) based on the last acquisition.
        /// The idea here is that the client would adjust request an adjustment to parameters
        /// and then try again.  This will only be allowed if 
        /// AquisitionStatus >= AquisitionLegitimate
        /// </summary>
        private void AdjustCalibrationBasedOnLastAcquisition()
        {
            // Always Adjust Y...
            _device.YOffset = _device.YOffset - _yOffsetInMeters;
            
            _velocityOffsetInMetersPerSecond = GetVelocityOffset();

            if (Math.Abs(_velocityOffsetInMetersPerSecond) > .001)
            {
                // If Speed needs setting Off, only use the first fid for home offset adjustment...
                // Other will be further off...
                _xOffsetInMeters = _fidClosestToLeadingEdge.ColumnDifference() * _mosaicSet.GetNominalPixelSizeX();
                _device.HomeOffset = _device.HomeOffset + _xOffsetInMeters;               
                _device.ConveyorVelocity = _device.ConveyorVelocity + _velocityOffsetInMetersPerSecond;
            }
            else
            {
                _device.HomeOffset = _device.HomeOffset + _xOffsetInMeters;               
            }
        }

        private double GetVelocityOffset()
        {
            if (_fidClosestToLeadingEdge == null || _fidFarthestToLeadingEdge == null)
                return 0.0;

            // If Fids are not far apart, we can't adjust speed...
            if (Math.Abs(_fidFarthestToLeadingEdge.GetNominalXPosition() - _fidClosestToLeadingEdge.GetNominalXPosition()) < .009)
                return 0.0;

            // We can try to calculate an offset for speed...
            double nominalDistance = _fidFarthestToLeadingEdge.GetNominalXPosition() -
                                     _fidClosestToLeadingEdge.GetNominalXPosition();


            double actualDistance =
                (_fidFarthestToLeadingEdge.GetNominalXPosition() + _fidFarthestToLeadingEdge.ColumnDifference() * _mosaicSet.GetNominalPixelSizeX()) -
                (_fidClosestToLeadingEdge.GetNominalXPosition() + _fidClosestToLeadingEdge.ColumnDifference() * _mosaicSet.GetNominalPixelSizeX());

            double ratio =  nominalDistance/actualDistance;

            return _device.ConveyorVelocity - _device.ConveyorVelocity*ratio;
        }

        private void ResetForAcquisition()
        {
            _yOffsetInMeters = 0.0;
            _xOffsetInMeters = 0.0;
            _velocityOffsetInMetersPerSecond = 0.0;
            _calibrationStatus = CalibrationStatus.AquisitionFailed;
            _waitingForImages = false;
            _fidClosestToLeadingEdge = null;
            _fidFarthestToLeadingEdge = null;
        }

        /// <summary>
        /// Gets the difference between the current SIM Setting and what the Calibrator is suggesting
        /// it should be.  This would be for UI (display) purposes during calibration.
        /// </summary>
        /// <returns></returns>
        public double GetYOffsetInMeters()
        {
            return _yOffsetInMeters;
        }

        /// <summary>
        /// Gets the difference between the current SIM Setting and what the Calibrator is suggesting
        /// it should be.  This would be for UI (display) purposes during calibration.
        /// </summary>
        /// <returns></returns>
        public double GetXOffsetInMeters()
        {
            return _xOffsetInMeters;
        }

        /// <summary>
        /// Gets the difference between the current SIM Setting and what the Calibrator is suggesting
        /// it should be.  This would be for UI (display) purposes during calibration.
        /// </summary>
        /// <returns></returns>
        public double GetVelocityOffsetInMetersPerSecond()
        {
            return _velocityOffsetInMetersPerSecond;
        }

        /// <summary>
        /// Fires a messages to client that lets them know that calibration is complete
        /// </summary>
        /// <param name="calStatus"></param>
        protected void FireCalibrationComplete(CalibrationStatus calStatus)
        {
            if (CalibrationComplete == null)
                return;

            CalibrationComplete(calStatus);
        }

        private void SetupCaptureSpecs(bool bSimulating)
        {
            if (!bSimulating)
            {
                // @todo - talk to hogan about what the number of buffers should be....
                // Chicken and egg problem with allocation... you don't know how many buffer to use
                // until you setup capture specs and you can't set up capture specs until you have buffers.
                int bufferCount = 128;
                int desiredCount = bufferCount;
                _device.AllocateFrameBuffers(ref bufferCount);

                if(desiredCount != bufferCount)
                    throw new ApplicationException("Could not allocate buffers...");

                ManagedSIMCaptureSpec cs1 = _device.SetupCaptureSpec(_panel.PanelSizeX, _panel.PanelSizeY, 0, .004);
                if (cs1 == null)
                {
                    throw new ApplicationException("Could not setup captureSpec for calibration");
                }
            }
            else
            {
                // This is needed to initialize row acquisition in Simulation Mode...
                // @todo - change CoreAPI to make this not needed (perhaps)
                _waitingForImages = true;
                _device.StartAcquisition(ACQUISITION_MODE.CAPTURESPEC_MODE);
                // Wait for all images to be gathered...
                _doneEvent.WaitOne();
            }
        }

        private void SetupMosaic()
        {
            ManagedSIMCamera cam = _device.GetSIMCamera(0);
            _mosaicSet = new ManagedMosaicSet(_panel.PanelSizeX, _panel.PanelSizeY, (uint)cam.Columns(), (uint)cam.Rows(), (uint)cam.Columns(), cPixelSizeInMeters, cPixelSizeInMeters);
            SimMosaicTranslator.AddDeviceToMosaic(_device, _mosaicSet);
            SimMosaicTranslator.SetCorrelationFlagsFIDOnly(_mosaicSet);
        }

        private void AcquisitionDone(int device, int status, int count)
        {
            if (_waitingForImages)
               _doneEvent.Set();
            else
            {
                CalculateCalibrationResults();
            }
        }

        private void FrameDone(ManagedSIMFrame pframe)
        {
            if (_mosaicSet == null)
                return;

            _mosaicSet.AddImage(pframe.BufferPtr(), 0,
                                (uint) pframe.CameraIndex(), (uint) pframe.TriggerIndex());
        }

        /// <summary>
        /// @TODO - Do we want to rely on BasicStitcher for this?
        /// </summary>
        /// <returns></returns>
        private Bitmap StitchRowImageFromMosaic()
        {
            // NOTE from Alan.  I am currently using the BasicStitcher for this (same as existing 2DSPI).
            // This adds a dependency on CyberCommon that I don't like.
            // @todo... Todd has also expressed an interest in using the camera cal for this stitching... Necessary?
            BasicStitcher stitcher = new BasicStitcher();
            stitcher.Initialize(_mosaicSet.GetLayer(0).GetNumberOfCameras(),
                0, 1, (int)_mosaicSet.GetImageWidthInPixels(), 
                (int)_mosaicSet.GetImageLengthInPixels(), 0, 0, false);

            for (int i = 0; i < _mosaicSet.GetLayer(0).GetNumberOfCameras(); i++)
                stitcher.AddTile(_mosaicSet.GetLayer(_layerIndex).GetTile((uint) i, 0).GetImageBuffer(), i, 0);

            return stitcher.CurrentBitmap;
        }
    }
}
