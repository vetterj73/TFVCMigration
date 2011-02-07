using Cyber.MPanel;
using MCoreAPI;

namespace SIMCalibrator
{
    /// <summary>
    /// When the client starts a calibration acquisition, there are 4 possible outcomes:
    /// </summary>
    public enum AquisitionStatus
    {
        AquisitionFailed,           // Could not get images
        AquisitionNotLegitimate,    // Got images, but can't find fiducials user suggests
        AquisitionNotInTolerance,   // Found fiducials out of tolerance
        AquisitionInTolerance,      // Found fiducials in tolerance (no further work needed).
    }

    public delegate void CalibrationAcquisitionComplete(AquisitionStatus status);

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
        /// <summary>
        /// Fired after images are acquired and calibration is verified.
        /// </summary>
        public event CalibrationAcquisitionComplete CalibrationAcquisitionComplete;

        /// <summary>
        /// Constructor:  Given a valid CPanel Object and a valid SIM Device
        /// </summary>
        /// <param name="panel"></param>
        /// <param name="device"></param>
        PositionCalibrator(CPanel panel, ManagedSIMDevice device)
        {
        }

        /// <summary>
        /// Starts the acquisition on the device.  This is an async call.  When the acquisition is
        /// complete, the CalibrationAcquisitionComplete event will be sent to listeners.
        /// </summary>
        public void StartAcquisition()
        {
        }

        /// <summary>
        /// Adjust the settings (XOffset, YOffset and Speed) based on the last acquisition.
        /// The idea here is that the client would adjust request an adjustment to parameters
        /// and then try again.  This will only be allowed if 
        /// AquisitionStatus >= AquisitionLegitimate
        /// </summary>
        public void AdjustCalibrationBasedOnLastAcquisition()
        {
        }

        /// <summary>
        /// Gets the difference between the current SIM Setting and what the Calibrator is suggesting
        /// it should be.  This would be for UI (display) purposes during calibration.
        /// </summary>
        /// <returns></returns>
        public double GetYOffsetDifference()
        {
            return 0.0;
        }

        /// <summary>
        /// Gets the difference between the current SIM Setting and what the Calibrator is suggesting
        /// it should be.  This would be for UI (display) purposes during calibration.
        /// </summary>
        /// <returns></returns>
        public double GetXOffsetDifference()
        {
            return 0.0;
        }

        /// <summary>
        /// Gets the difference between the current SIM Setting and what the Calibrator is suggesting
        /// it should be.  This would be for UI (display) purposes during calibration.
        /// </summary>
        /// <returns></returns>
        public double GetSpeedDifference()
        {
            return 0.0;
        }
    }
}
