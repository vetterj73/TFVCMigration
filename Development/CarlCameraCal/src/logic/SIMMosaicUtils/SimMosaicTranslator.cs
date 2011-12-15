using System;
using MCoreAPI;
using MMosaicDM;
using SIMAPI;

namespace SIMMosaicUtils
{
    public class SimMosaicTranslator
    {
        /// <summary>
        /// Uses the static SIM CoreAPI to initialize the MosaicSet Layers as needed...
        /// </summary>
        /// <param name="set"></param>
        public static void InitializeMosaicFromCurrentSimConfig(ManagedMosaicSet set, bool bMaskForDiffDevices)
        {
            for (int i = 0; i < ManagedCoreAPI.NumberOfDevices(); i++)
                AddDeviceToMosaic(ManagedCoreAPI.GetDevice(i), (uint)i, set);

            SetDefaultCorrelationFlags(set, bMaskForDiffDevices);
        }

        /// <summary>
        /// @todo - I think this could be optimized...
        /// </summary>
        /// <param name="device"></param>
        /// <param name="set"></param>
        public static void AddDeviceToMosaic(ManagedSIMDevice device, uint deviceIndex, ManagedMosaicSet set)
        {
            if (device.NumberOfCaptureSpecs <= 0)
                throw new ApplicationException("AddDeviceToMosaic - There are not CaptureSpecs defined for teh device!");

            /// @todo - this should be made part of the SIM Device....
            uint numCameras = 0;
            for (int i = 0; i < device.NumberOfCameras; i++)
                if (device.GetSIMCamera(i).Status() == (CameraStatus)1)
                    numCameras++;

            for (uint i = 0; i < device.NumberOfCaptureSpecs; i++)
            {
                ManagedSIMCaptureSpec pSpec = device.GetSIMCaptureSpec((int)i);
                bool bFiducialAllowNegativeMatch = true;   // Dark field allow negative match
                if (i % 2 == 0) bFiducialAllowNegativeMatch = false; // Bright field not allow negavie match
                bool bAlignWithCAD = false;
                bool bAlignWithFiducial = true;
                bool bFiducialBrighterThanBackground = true;
            /* For debug
                bAlignWithFiducial = false;
                if (iDeviceIndex == 1 && i == 0)
                    bAlignWithFiducial = true;
            //*/
                ManagedMosaicLayer layer = set.AddLayer(numCameras, (uint)pSpec.NumberOfTriggers, bAlignWithCAD, bAlignWithFiducial, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch, deviceIndex);

                if (layer == null)
                    throw new ApplicationException("AddDeviceToMosaic - Layer was null - this should never happen!");

                // Use camera zero as reference
                ManagedSIMCamera camera0 = device.GetSIMCamera(0);
                // Set up the transform parameters...
                for (uint j = 0; j < numCameras; j++)
                {
                    ManagedSIMCamera camera = device.GetSIMCamera((int)j);
                    for (uint k = 0; k < pSpec.NumberOfTriggers; k++)
                    {
                        ManagedMosaicTile mmt = layer.GetTile(j, k);

                        if (mmt == null)
                            throw new ApplicationException("AddDeviceToMosaic - Tile was null - this should never happen");
                        
                        // First camera center in X
                        double dTrigOffset = pSpec.GetTriggerAtIndex((int)k) + pSpec.XOffset();
                        double xOffset = set.GetObjectWidthInMeters() - dTrigOffset - camera0.Pixelsize.X * camera0.Rows() / 2;
                        // The camera center in X
                        xOffset += (camera.CenterOffset.X - camera0.CenterOffset.X);
                        // The camera's origin in X
                        xOffset -= (camera.Pixelsize.X * camera.Rows() / 2);

                        // First camera center in Y
                        double yOffset = (-device.YOffset + camera0.Pixelsize.Y * camera0.Columns() / 2); 
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

        private static void SetDefaultCorrelationFlags(ManagedMosaicSet set, bool bMaskForDiffDevices)
        {
            for (uint i = 0; i < set.GetNumMosaicLayers(); i++)
            {
                for (uint j = 0; j < set.GetNumMosaicLayers(); j++)
                {
                    ManagedCorrelationFlags flag = set.GetCorrelationSet(i, j);
                    if (i == j)
                    {
                        flag.SetCameraToCamera(true);

                        if ((set.GetNumMosaicLayers() == 1) ||
                            (set.GetNumMosaicLayers() == 2 && ManagedCoreAPI.NumberOfDevices() == 2))
                            flag.SetTriggerToTrigger(true); // For one illumination for a SIM
                        else
                            flag.SetTriggerToTrigger(false);
                    }
                    else
                    {
                        flag.SetCameraToCamera(false);

                        if ((i == 0 && j == 3) || (i == 3 && j == 0) ||
                             (i == 1 && j == 2) || (i == 2 && j == 1))
                            flag.SetTriggerToTrigger(false); // For four illuminaitons
                        else
                            flag.SetTriggerToTrigger(true);

                        if (Math.Abs((int)i-(int)j)==2) // For four illuminations, (0, 2) and (1,3) are the same illumination type
                            flag.SetApplyCorrelationAreaSizeUpLimit(true);
                    }

                    flag.SetMaskNeeded(false);
                    if (bMaskForDiffDevices)
                    {
                        if(Math.Abs(i-j)>=2)    // For layrer in difference device
                            flag.SetMaskNeeded(true);
                    }   
                }
            }
        }

        public static void SetCorrelationFlagsFIDOnly(ManagedMosaicSet set)
        {
            for (uint i = 0; i < set.GetNumMosaicLayers(); i++)
            {
                for (uint j = 0; j < set.GetNumMosaicLayers(); j++)
                {
                    ManagedCorrelationFlags flag = set.GetCorrelationSet(i, j);
                    flag.SetCameraToCamera(false);
                    flag.SetTriggerToTrigger(false);
                    flag.SetMaskNeeded(false);
                }
            }
        }
    }
}
