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
            for (int i = 0; i < device.NumberOfCamerasEnabled; i++)
                if (device.GetSIMCamera(i+device.FirstCameraEnabled).Status() == (CameraStatus)1)
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
                ManagedSIMCamera camera0 = device.GetSIMCamera(0 + device.FirstCameraEnabled);
                ManagedSIMCamera lastcamera = device.GetSIMCamera((int)numCameras + device.FirstCameraEnabled - 1);
                // Set up the transform parameters...
                for (uint j = 0; j < numCameras; j++)
                {
                    ManagedSIMCamera camera = device.GetSIMCamera((int)j + device.FirstCameraEnabled);

                    for (uint k = 0; k < pSpec.NumberOfTriggers; k++)
                    {
                        uint trigger = (device.ConveyorRtoL) ?
                            (uint)(pSpec.NumberOfTriggers - k - 1) : k;
                        ManagedMosaicTile mmt = layer.GetTile(j, trigger);

                        if (mmt == null)
                            throw new ApplicationException("AddDeviceToMosaic - Tile was null - this should never happen");
                        
                        // First camera center in X
                        double dTrigOffset = pSpec.GetTriggerAtIndex((int)k) + pSpec.XOffset();
                        double xOffset = set.GetObjectWidthInMeters() - dTrigOffset - camera0.Pixelsize.X * camera0.Rows() / 2;
                        // The camera center in X
                        xOffset += (camera.CenterOffset.X - camera0.CenterOffset.X);
                        // The camera's origin in X
                        xOffset -= (camera.Pixelsize.X * camera.Rows() / 2);

                        if (device.ConveyorRtoL)
                        {
                            xOffset = dTrigOffset + (camera.CenterOffset.X - camera0.CenterOffset.X);
                            xOffset += (camera0.Pixelsize.X - camera.Pixelsize.X) * camera0.Rows() / 2;
                        }

                        // First camera center in Y
                        double yOffset = (-device.YOffset + camera0.Pixelsize.Y * camera0.Columns() / 2); 
                        // The camera center in Y
                        yOffset += (camera.CenterOffset.Y - camera0.CenterOffset.Y);
                        // The camera orign in Y
                        yOffset -= (camera.Pixelsize.Y * camera.Columns() / 2);

                        if (device.FixedRearRail)
                        {
                            double boardedge = lastcamera.CenterOffset.Y + lastcamera.Pixelsize.Y * lastcamera.Columns() / 2;
                            boardedge -= device.YOffset + set.GetObjectLengthInMeters();
                            yOffset = camera.CenterOffset.Y - boardedge - (camera.Pixelsize.Y * camera.Columns() / 2);
                        }
                        
                        // Trigger offset is initial offset + triggerIndex * overlap...
                        mmt.SetTransformParameters(camera.Pixelsize.X, camera.Pixelsize.Y,
                            camera.Rotation,
                            xOffset, yOffset);
                        // TODO 
                        // modify to load u,v limits, m, dmdz, calc inverse
                        // Load the Camera Model calibration into the mosaic tile's _tCamCalibration object
                        mmt.ResetTransformCamCalibration();
                        mmt.ResetTransformCamModel();
                        mmt.SetTransformCamCalibrationUMax( camera.Columns());
                        mmt.SetTransformCamCalibrationVMax( camera.Rows());
                        for (uint m = 0; m < 16; m++)
                        {
                            mmt.SetTransformCamCalibrationS(m,      (float)camera.get_HorizontalDistortion(m));
                            mmt.SetTransformCamCalibrationS(m + 16, (float)camera.get_VerticalDistortion(m)  );
                            mmt.SetTransformCamCalibrationdSdz(m,      (float)camera.get_HorizontalSensitivity(m));
                            mmt.SetTransformCamCalibrationdSdz(m + 16, (float)camera.get_VerticalSensitivity(m)  );
                        }
                        // TODO  *** inverse not yet used, is it really needed?
                        // calc Inverse // make sure that this works...
                    }
                }
            }
        }

        //private void 

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

        // calculates new mosaic trigger value based on frame trigger and conveyor direction
        public static int TranslateTrigger(ManagedSIMFrame pframe)
        {
            int device = pframe.DeviceIndex();

            int triggers = ManagedCoreAPI.GetDevice(device).GetSIMCaptureSpec(pframe.CaptureSpecIndex()).NumberOfTriggers;

            int trigger = (ManagedCoreAPI.GetDevice(device).ConveyorRtoL) ?
                triggers - pframe.TriggerIndex() - 1 : pframe.TriggerIndex();

            return trigger;
        }
    }
}
