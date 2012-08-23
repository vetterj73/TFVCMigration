#include "StdAfx.h"
#include "stdlib.h"

#include "ConfigMosaicSet.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"
#include "CorrelationFlags.h"

using namespace SIMAPI;

void ConfigMosaicSet::MosaicSetDefaultConfiguration(MosaicSet* pSet, bool bMaskForDiffDevices)
{
	for (int i = 0; i < SIMCore::NumberOfDevices(); i++)
	    AddDeviceToMosaic(pSet, SIMCore::GetSIMDevice(i), i);
	
	SetDefaultCorrelationFlags(pSet, bMaskForDiffDevices);
}

int ConfigMosaicSet::AddDeviceToMosaic(MosaicSet* pSet, ISIMDevice *pDevice, int iDeviceIndex)
{
	// validation check
	if (pDevice->NumberOfCaptureSpecs() <= 0)
		return(-1);

	// Number of cameras
	unsigned int iNumCam = 0;
	for (unsigned int i = 0; i < pDevice->NumberOfCamerasEnabled(); i++)
		if (pDevice->GetSIMCamera(i+pDevice->FirstCameraEnabled())->Status() == (CameraStatus)1)
			iNumCam++;
	// for each capture spec
	for (unsigned int i = 0; i < pDevice->NumberOfCaptureSpecs(); i++)
    {
		CSIMCaptureSpec* pSpec = pDevice->GetSIMCaptureSpec((int)i);
		bool bFiducialAllowNegativeMatch = true;   // Dark field allow negative match
        if (i % 2 == 0) bFiducialAllowNegativeMatch = false; // Bright field not allow negavie match
        bool bAlignWithCAD = false;
        bool bAlignWithFiducial = true;
        bool bFiducialBrighterThanBackground = true;

		MosaicLayer* pLayer = pSet->AddLayer(iNumCam, pSpec->GetNumberOfTriggers(), bAlignWithCAD, bAlignWithFiducial, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch, iDeviceIndex);

		if (pLayer == NULL)
			return(-2);

        // Use camera zero as reference
        ISIMCamera* pCamera0 = pDevice->GetSIMCamera(0 + pDevice->FirstCameraEnabled());
		ISIMCamera* pLastCamera = pDevice->GetSIMCamera((int)iNumCam + pDevice->FirstCameraEnabled() - 1);
        // Set up the transform parameters...
		for (unsigned int j = 0; j < iNumCam; j++)
        {
			ISIMCamera* pCamera = pDevice->GetSIMCamera((int)j + pDevice->FirstCameraEnabled());

			for (unsigned int k = 0; k < pSpec->GetNumberOfTriggers(); k++)
            {
				unsigned int trigger = (pDevice->ConveyorRtoL()) ?(unsigned int)(pSpec->GetNumberOfTriggers() - k - 1) : k;
               
				MosaicTile* pTile = pLayer->GetTile(trigger, j);

				if (pTile == NULL)
					return(-3);
                        
				// First camera center in X
				double dTrigOffset = pSpec->GetTriggerAtIndex((int)k) + pSpec->XOffset();
				double xOffset = pSet->GetObjectWidthInMeters() - dTrigOffset - pCamera0->Pixelsize().X() * pCamera0->Rows() / 2;
                // The camera center in X
                xOffset += (pCamera->CenterOffset().X() - pCamera0->CenterOffset().X());
                // The camera's origin in X
                xOffset -= (pCamera->Pixelsize().X() * pCamera->Rows() / 2);

				if (pDevice->ConveyorRtoL())
                {
					xOffset = dTrigOffset + (pCamera->CenterOffset().X() - pCamera0->CenterOffset().X());
                    xOffset += (pCamera0->Pixelsize().X() - pCamera->Pixelsize().X()) * pCamera0->Rows() / 2;
				}

                // First camera center in Y
                double yOffset = (-pDevice->YOffset() + pCamera0->Pixelsize().Y() * pCamera0->Columns() / 2); 
                // The camera center in Y
                yOffset += (pCamera->CenterOffset().Y() - pCamera0->CenterOffset().Y());
                // The camera orign in Y
                yOffset -= (pCamera->Pixelsize().Y() * pCamera->Columns() / 2);

                if (pDevice->FixedRearRail())
                {
					double boardedge = pLastCamera->CenterOffset().Y() + pLastCamera->Pixelsize().Y() * pLastCamera->Columns() / 2;
                    boardedge -= pDevice->YOffset() + pSet->GetObjectLengthInMeters();
                    yOffset = pCamera->CenterOffset().Y() - boardedge - (pCamera->Pixelsize().Y() * pCamera->Columns() / 2);
                }
                        
				// Trigger offset is initial offset + triggerIndex * overlap...
                pTile->SetTransformParameters(
					pCamera->Pixelsize().X(), pCamera->Pixelsize().Y(),
                    pCamera->Rotation(),
                    xOffset, yOffset);
				// TODO 
                // modify to load u,v limits, m, dmdz, calc inverse
                // Load the Camera Model calibration into the mosaic tile's _tCamCalibration object
                pTile->ResetTransformCamCalibration();
                pTile->ResetTransformCamModel();
                pTile->SetTransformCamCalibrationUMax( pCamera->Columns());
                pTile->SetTransformCamCalibrationVMax( pCamera->Rows());
                for (unsigned int m = 0; m < 16; m++)
                {
					pTile->SetTransformCamCalibrationS(m,      (float)pCamera->HorizontalDistortion(m));
                    pTile->SetTransformCamCalibrationS(m + 16, (float)pCamera->VerticalDistortion(m)  );
                    pTile->SetTransformCamCalibrationdSdz(m,      (float)pCamera->HorizontalSensitivity(m));
                    pTile->SetTransformCamCalibrationdSdz(m + 16, (float)pCamera->VerticalSensitivity(m)  );
                }
                // TODO  *** inverse not yet used, is it really needed?
                // calc Inverse // make sure that this works..
            }
        }
	}

	return(1);
}


void ConfigMosaicSet::SetDefaultCorrelationFlags(MosaicSet* pSet, bool bMaskForDiffDevices)
{
	for (unsigned int i = 0; i < pSet->GetNumMosaicLayers(); i++)
    {
		for (unsigned int j = 0; j < pSet->GetNumMosaicLayers(); j++)
        {
			CorrelationFlags* pFlag = pSet->GetCorrelationFlags(i, j);
            if (i == j) // For same layer
            {
				pFlag->SetCameraToCamera(true);

				if ((pSet->GetNumMosaicLayers() == 1) ||
					(pSet->GetNumMosaicLayers() == 2 && SIMCore::NumberOfDevices() == 2))
                    pFlag->SetTriggerToTrigger(true); // For one Layer for a SIM
				else
					pFlag->SetTriggerToTrigger(false);

				// If only one camera is used 
                if(pSet->GetLayer(0)->GetNumberOfCameras() == 1)
					pFlag->SetTriggerToTrigger(true);
			}
            else // For different layers
            {
				pFlag->SetCameraToCamera(false);
				
				// If only one or 2 trigger
                if (pSet->GetLayer(0)->GetNumberOfTriggers() <= 2)
					pFlag->SetCameraToCamera(true);

				if ((i == 0 && j == 3) || (i == 3 && j == 0) ||
					(i == 1 && j == 2) || (i == 2 && j == 1))
                    pFlag->SetTriggerToTrigger(false); // For four layers
				else
                    pFlag->SetTriggerToTrigger(true);

                if (abs((int)i-(int)j)==2) // For four Layers, (0, 2) and (1,3) are the same Layer type
					pFlag->SetApplyCorrelationAreaSizeUpLimit(true);
			}
			if (bMaskForDiffDevices)
            {
				int iValue = abs((int)i - (int)j);
                if (abs((int)i - (int)j) >= 2 // For layrer in difference device
					|| (i>=2 && j>=2)
                    || pSet->GetNumMosaicLayers() == 1)   // For single layer case
				{
					bool bMask = true; 
		            double dMinHeight = 0;
                    MaskInfo maskInfo(bMask, dMinHeight);
                    pFlag->SetMaskInfo(maskInfo);   
                }
            }
        }
	}
}


// calculates new mosaic trigger value based on frame trigger and conveyor direction
int ConfigMosaicSet::TranslateTrigger(CSIMFrame* pFrame)
{
	int device = pFrame->DeviceNumber();

    int triggers = SIMCore::GetSIMDevice(device)->GetSIMCaptureSpec(pFrame->CaptureSpecNumber())->GetNumberOfTriggers();

    int trigger = (SIMCore::GetSIMDevice(device)->ConveyorRtoL()) ?
		triggers - pFrame->TriggerIndex() - 1 : pFrame->TriggerIndex();

    return trigger;
}

