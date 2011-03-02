﻿using System;
using System.Collections.Generic;
using PanelAlignM;

namespace SIMCalibrator
{
    
    /// <summary>
    /// This class is used to capture details about a set of fiducials needed for calibration...
    /// </summary>
    public class FiducialList
    {
        private List<ManagedFidInfo> _fidList = new List<ManagedFidInfo>();
        public const double cLowestAcceptibleCorrelationScore = .85;
        public const double cMinimumAcceptibleDistanceBetweenFidsForSpeedCalc = .009;
        public const double cMaximumVelocityRatioStillInTolerance = .01;
        private const double cYInTolerance = .0005;
        private const double cXInTolerance = .0005;

        /// <summary>
        /// Clear all fids in the list
        /// </summary>
        public void Clear()
        {
            _fidList.Clear();
        }

        /// <summary>
        /// Add a fid if it meets the criterea (return true if it is added).
        /// </summary>
        /// <param name="fid"></param>
        /// <returns></returns>
        public bool Add(ManagedFidInfo fid)
        {
            if (fid.CorrelationScore() >= cLowestAcceptibleCorrelationScore)
            {
                _fidList.Add(fid);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Number of fids in the list
        /// </summary>
        public int Count
        {
            get { return _fidList.Count; }
        }

        /// <summary>
        /// Get the fid that is closest to the leading edge of the board.
        /// </summary>
        /// <returns></returns>
        public ManagedFidInfo GetFidClosestToLeadingEdge()
        {
            ManagedFidInfo fid = null;
            foreach (ManagedFidInfo curFid in _fidList)
            {
                if (fid == null)
                    fid = curFid;
                else if (fid.GetNominalXPosition() > curFid.GetNominalXPosition())
                    fid = curFid;
            }
            
            return fid;
        }

        /// <summary>
        /// Get the fid that is farthest away from the leading edge of the board
        /// </summary>
        /// <returns></returns>
        public ManagedFidInfo GetFidFarthestFromLeadingEdge()
        {
            ManagedFidInfo fid = null;
            foreach (ManagedFidInfo curFid in _fidList)
            {
                if (fid == null)
                    fid = curFid;
                else if (fid.GetNominalXPosition() < curFid.GetNominalXPosition())
                    fid = curFid;
            }
            return fid;
        }
    
        /// <summary>
        /// Gets the average X Offset from the list...
        /// NOTE:  This should only be used if the client decides that velocity is in tolerance
        /// </summary>
        /// <param name="pixelSize"></param>
        /// <returns></returns>
        public double GetAverageXOffset(double pixelSize)
        {
            if(_fidList.Count == 0)
                return 0.0;

            double xOffset = 0.0;
            foreach (ManagedFidInfo curFid in _fidList)
            {
                xOffset += curFid.ColumnDifference() * pixelSize;
            }
            xOffset /= _fidList.Count;
            return xOffset;
        }

        /// <summary>
        /// Gets the average Y Offset from the list...
        /// </summary>
        /// <param name="pixelSize"></param>
        /// <returns></returns>
        public double GetAverageYOffset(double pixelSize)
        {
            if (_fidList.Count == 0)
                return 0.0;

            double yOffset = 0.0;
            foreach (ManagedFidInfo curFid in _fidList)
            {
                yOffset += curFid.RowDifference() * pixelSize;
            }
            yOffset /= _fidList.Count;
            return yOffset;
        }

        /// <summary>
        /// Gets the ratio of the nominal fid positions to actual fid positions
        /// for the 2 fids that are farthest apart.
        /// </summary>
        /// <param name="pixelSize"></param>
        /// <returns></returns>
        public double GetNominalToActualVelocityRatio(double pixelSize)
        {
            ManagedFidInfo closestFid = GetFidClosestToLeadingEdge();
            ManagedFidInfo farthestFid = GetFidFarthestFromLeadingEdge();

            if (closestFid == null || farthestFid == null)
                return 1.0;

            // If Fids are not far apart, we can't adjust speed...
            if (Math.Abs(farthestFid.GetNominalXPosition() - closestFid.GetNominalXPosition()) < cMinimumAcceptibleDistanceBetweenFidsForSpeedCalc)
                return 1.0;

            // We can try to calculate an offset for speed...
            double nominalDistance = farthestFid.GetNominalXPosition() -
                                     closestFid.GetNominalXPosition();

            double actualDistance =
                (farthestFid.GetNominalXPosition() + farthestFid.ColumnDifference() * pixelSize) -
                (closestFid.GetNominalXPosition() + closestFid.ColumnDifference() * pixelSize);

            return nominalDistance / actualDistance;
        }

        /// <summary>
        /// Is the velocity ratio in tolerance?
        /// </summary>
        /// <param name="ratio"></param>
        /// <returns></returns>
        public bool IsVelocityRatioInTolerance(double ratio)
        {
            if (Math.Abs(1.0 - ratio) < cMaximumVelocityRatioStillInTolerance)
                return true;

            return false;
        }

        public bool IsXInTolerance(double XOffset)
        {
            if (Math.Abs(XOffset) < cXInTolerance)
                return true;

            return false;
        }

        public bool IsYInTolerance(double YOffset)
        {
            if (Math.Abs(YOffset) < cYInTolerance)
                return true;

            return false;
        }

    }
}
