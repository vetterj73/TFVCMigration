using System;
using MCoreAPI;
using MMosaicDM;
using SIMAPI;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

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
                        ManagedMosaicTile mmt = layer.GetTile(trigger, j);

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
                            //if(m==3 || m==9)
                            //    mmt.SetTransformCamCalibrationS(m, 0);
                            //else
                                mmt.SetTransformCamCalibrationS(m,      (float)camera.get_HorizontalDistortion(m));

                            //if (m == 6 || m == 12)
                            //    mmt.SetTransformCamCalibrationS(m + 16, 0);
                            //else
                                mmt.SetTransformCamCalibrationS(m + 16, (float)camera.get_VerticalDistortion(m));

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
                    if (i == j) // For same layer
                    {
                        flag.SetCameraToCamera(true);

                        if ((set.GetNumMosaicLayers() == 1) ||
                            (set.GetNumMosaicLayers() == 2 && ManagedCoreAPI.NumberOfDevices() == 2))
                            flag.SetTriggerToTrigger(true); // For one Layer for a SIM
                        else
                            flag.SetTriggerToTrigger(false);

                        // If only one camera is used 
                        if(set.GetLayer(0).GetNumberOfCameras() == 1)
                            flag.SetTriggerToTrigger(true);
                    }
                    else // For different layers
                    {
                        flag.SetCameraToCamera(false);

                        // If only one or 2 trigger
                        if (set.GetLayer(0).GetNumberOfTriggers() <= 2)
                            flag.SetCameraToCamera(true);

                        if ((i == 0 && j == 3) || (i == 3 && j == 0) ||
                             (i == 1 && j == 2) || (i == 2 && j == 1))
                            flag.SetTriggerToTrigger(false); // For four layers
                        else
                            flag.SetTriggerToTrigger(true);

                        if (Math.Abs((int)i-(int)j)==2) // For four Layers, (0, 2) and (1,3) are the same Layer type
                            flag.SetApplyCorrelationAreaSizeUpLimit(true);
                    }

                    if (bMaskForDiffDevices)
                    {
                        int iValue = (int)Math.Abs((int)i - (int)j);
                        if (Math.Abs((int)i - (int)j) >= 2 // For layrer in difference device
                            || (i>=2 && j>=2)
                            || set.GetNumMosaicLayers() == 1)   // For single layer case
                        {
                            bool bMask = true; 
		                    double dMinHeight = 0;
                            ManagedMaskInfo maskInfo = new ManagedMaskInfo(
                                bMask, dMinHeight);
                            flag.SetMaskInfo(maskInfo);   
                        }
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

        public static int InitializeMosaicFromNominalTrans(
            ManagedMosaicSet set, string sFilename, 
            double dPanelHeight)
        {
            // Read calibration and trig infomation
            double dOx, dOy;
            List<double []> transList = new List<double[]>();
            List<double> trigList = new List<double>();
            using (System.IO.StreamReader reader = new System.IO.StreamReader(sFilename))
            {
                // Read X, Y offset
                string sLine = reader.ReadLine();
                string [] sSeg = sLine.Split(new char[] {','});
                if (!sSeg[0].Contains("Offset(x y)")) return (-1);
                dOx = -Convert.ToDouble(sSeg[1]);
                dOy = -Convert.ToDouble(sSeg[2]);

                // Read transform based on camera calibration       
                sLine = reader.ReadLine(); // Skip one line
                while (reader.Peek() > 0)
                {
                    // Validation check 
                    sLine = reader.ReadLine();
                    sSeg = sLine.Split(new char[] { ',' });
                    if (!sSeg[0].Contains("Camera"))
                        break;

                    // Read a transform
                    double [] trans = new double[8];
                    for(int i =0; i<8; i++)
                        trans[i] = Convert.ToDouble(sSeg[i+1]);
                    transList.Add(trans);
                }

                // Read trigger list
                    // first one
                if (!sSeg[0].Contains("Trig"))
                    return (-2);
                double dValue = Convert.ToDouble(sSeg[1]);
                trigList.Add(dValue);

                while(reader.Peek() > -1)
                {
                     // Validation check 
                    sLine = reader.ReadLine();
                    sSeg = sLine.Split(new char[] { ',' });
                    if (!sSeg[0].Contains("Trig"))
                        break;

                    dValue = Convert.ToDouble(sSeg[1]);
                    trigList.Add(dValue);
                }
            }

            // Add a mosaic layer
            uint iNumCam = (uint)transList.Count;
            uint iNumTrig = (uint)trigList.Count + 1;

            bool bFiducialAllowNegativeMatch = false; // Bright field not allow negavie match
            bool bAlignWithCAD = false;
            bool bAlignWithFiducial = true;
            bool bFiducialBrighterThanBackground = true;
            uint deviceIndex = 0;
            ManagedMosaicLayer layer = set.AddLayer(iNumCam, iNumTrig, bAlignWithCAD, bAlignWithFiducial, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch, deviceIndex);

            // Set nominal transforms
            uint iImageRows = set.GetImageLengthInPixels();
            double[] leftM = new double[]{ 
                0, -1, dPanelHeight-dOy, 
                1, 0, dOx,
                0, 0};

            double[] rightM = new double[]{
                0, 1, 0,
                -1, 0, iImageRows-1,
                0, 0, 1};

            double[] tempM = new double[8];
            double[] camM = new double[8];
            double[] fovM = new double[8];

            for (uint iCam = 0; iCam < iNumCam; iCam++)
            {
                // Calculate camera transform for first trigger
                MultiProjective2D(leftM, transList[(int)iCam], tempM);
                MultiProjective2D(tempM, rightM, camM);
                for (int i = 0; i < 8; i++)
                    fovM[i] = camM[i];

                for (uint iTrig = 0; iTrig < iNumTrig; iTrig++)
                {
                    // Set transform for each trigger
                    if (iTrig > 0)
                        fovM[2] -= trigList[(int)iTrig - 1]; // This calcualtion is not very accurate
                    ManagedMosaicTile mmt = layer.GetTile(iTrig, iCam);
                    mmt.SetNominalTransform(fovM);
                }
            }

            // Set correlation flags
            SetDefaultCorrelationFlags(set, false);

            return (1);
        }

        // Raw buffers need to be hold until the demosaic or copy is done (maybe in multi-thread)
        // LoadAllRawImages() and ReleaseRawBufs() need to be called in pair 
        private static Bitmap[,] fovs = null;
        private static System.Drawing.Imaging.BitmapData[,] bufs = null; 
        public static bool LoadAllRawImages(ManagedMosaicSet set, string sFolder)
        {
            // Validation check
            if (set.GetNumMosaicLayers() != 1)
                return (false);

            ManagedMosaicLayer layer = set.GetLayer(0);
            uint iTrigNum = (uint)layer.GetNumberOfTriggers();
            uint iCamNum = (uint)layer.GetNumberOfCameras();

            if (fovs == null)
                fovs = new Bitmap[iTrigNum, iCamNum];
            if (bufs == null)
                bufs = new System.Drawing.Imaging.BitmapData[iTrigNum, iCamNum];

            for (uint iTrig = 0; iTrig < iTrigNum; iTrig++)
            {
                for (uint iCam = 0; iCam < iCamNum; iCam++)
                {
                    string sFile = sFolder + "\\Cam" + iCam + "_Trig" + iTrig + ".bmp";
                    if (!File.Exists(sFile))
                        return (false);

                    // Load image  
                    fovs[iTrig, iCam] = new Bitmap(sFile);
                    Bitmap fov = fovs[iTrig, iCam];
                    if (fov.PixelFormat != System.Drawing.Imaging.PixelFormat.Format8bppIndexed ||
                        fov.Width != set.GetImageWidthInPixels() ||
                        fov.Height != set.GetImageLengthInPixels())
                        return false;

                    // Add image to mosaic set
                    System.Drawing.Imaging.BitmapData buf = fov.LockBits(
                        new Rectangle(0, 0, fov.Width, fov.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, fov.PixelFormat);
                    bufs[iTrig, iCam] = buf;

                    set.AddRawImage(buf.Scan0, 0, iCam, iTrig);
                }
            }

            return (true);
        }

        public static void ReleaseRawBufs(ManagedMosaicSet set)
        {
            // Validation check
            if (set.GetNumMosaicLayers() != 1 || fovs == null || bufs == null)
                return;

            ManagedMosaicLayer layer = set.GetLayer(0);
            uint iTrigNum = (uint)layer.GetNumberOfTriggers();
            uint iCamNum = (uint)layer.GetNumberOfCameras();
            for (uint iTrig = 0; iTrig < iTrigNum; iTrig++)
            {
                for (uint iCam = 0; iCam < iCamNum; iCam++)
                {
                    Bitmap fov = fovs[iTrig, iCam];

                    System.Drawing.Imaging.BitmapData buf = bufs[iTrig, iCam];

                    fov.UnlockBits(buf);

                    // Explicitly release bmp memory (may have memory leakage if doesn't do so)
                    fov.Dispose();
                }
            }
        }

        private static void MultiProjective2D(double [] leftM, double [] rightM, double [] outM)
        {
            outM[0] = leftM[0]*rightM[0]+leftM[1]*rightM[3]+leftM[2]*rightM[6];
	        outM[1] = leftM[0]*rightM[1]+leftM[1]*rightM[4]+leftM[2]*rightM[7];
	        outM[2] = leftM[0]*rightM[2]+leftM[1]*rightM[5]+leftM[2]*1;
											 
	        outM[3] = leftM[3]*rightM[0]+leftM[4]*rightM[3]+leftM[5]*rightM[6];
	        outM[4] = leftM[3]*rightM[1]+leftM[4]*rightM[4]+leftM[5]*rightM[7];
	        outM[5] = leftM[3]*rightM[2]+leftM[4]*rightM[5]+leftM[5]*1;
											 
	        outM[6] = leftM[6]*rightM[0]+leftM[7]*rightM[3]+1*rightM[6];
	        outM[7] = leftM[6]*rightM[1]+leftM[7]*rightM[4]+1*rightM[7];
	        double dScale = leftM[6]*rightM[2]+leftM[7]*rightM[5]+1*1;

	        if(dScale<0.01 && dScale>-0.01)
	        {
		        dScale = 0.01;
	        }

	        for(int i=0; i<8; i++)
		        outM[i] = outM[i]/dScale;
        }
        
    }
}
