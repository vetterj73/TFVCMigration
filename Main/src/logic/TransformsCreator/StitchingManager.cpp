#include "StitchingManager.h"
#include "Logger.h"
#include "MosaicLayer.h"
#include "Bitmap.h"

StitchingManager::StitchingManager(MosaicSet *pMosaicSet, Panel *pPanel)
{
	_pMosaicSet = pMosaicSet;
	_pPanel = pPanel;
}

StitchingManager::~StitchingManager(void)
{
}

// Create a panel image based on a mosaic image 
// iIllumIndex: mosaic image index
// pPanelImage: output, the stitched image
void StitchingManager::CreateStitchingImage(unsigned int iIllumIndex, Image* pPanelImage) const
{
	MosaicLayer* pMosaic = _pMosaicSet->GetLayer(iIllumIndex);
	CreateStitchingImage(pMosaic, pPanelImage);
}

// This is a utility function
// Create a panel image based on a mosaic image
// pMosaic: input, mosaic image
// pPanelImage: output, the stitched image
void StitchingManager::CreateStitchingImage(MosaicLayer* pMosaic, Image* pPanelImage)
{
	// Trigger and camera centers in world space
	unsigned int iNumTrigs = pMosaic->GetNumberOfTriggers();
	unsigned int iNumCams = pMosaic->GetNumberOfCameras();
	double* pdCenX = new double[iNumTrigs];
	double* pdCenY = new double[iNumCams];
	pMosaic->TriggerCentersInX(pdCenX);
	pMosaic->CameraCentersInY(pdCenY);

	// Panel image Row bounds for Roi (decreasing order)
	int* piRectRows = new int[iNumTrigs+1];
	piRectRows[0] = pPanelImage->Rows();
	for(unsigned int i=1; i<iNumTrigs; i++)
	{
		double dX = (pdCenX[i-1] +pdCenX[i])/2;
		double dTempRow, dTempCol;
		pPanelImage->WorldToImage(dX, 0, &dTempRow, &dTempCol);
		piRectRows[i] = (int)dTempRow;
		if(piRectRows[i]>=(int)pPanelImage->Rows()) piRectRows[i] = pPanelImage->Rows();
		if(piRectRows[i]<0) piRectRows[i] = 0;
	}
	piRectRows[iNumTrigs] = 0; 

	// Panel image Column bounds for Roi (increasing order)
	int* piRectCols = new int[iNumCams+1];
	piRectCols[0] = 0;
	for(unsigned int i=1; i<iNumCams; i++)
	{
		double dY = (pdCenY[i-1] +pdCenY[i])/2;
		double dTempRow, dTempCol;
		pPanelImage->WorldToImage(0, dY, &dTempRow, &dTempCol);
		piRectCols[i] = (int)dTempCol;
		if(piRectCols[i]<0) piRectCols[i] = 0;
		if(piRectCols[i]>(int)pPanelImage->Columns()) piRectCols[i] = pPanelImage->Columns();;
	}
	piRectCols[iNumCams] = pPanelImage->Columns();

	// Morph each Fov to create stitched panel image
	for(unsigned int iTrig=0; iTrig<iNumTrigs; iTrig++)
	{
		for(unsigned int iCam=0; iCam<iNumCams; iCam++)
		{
			Image* pFov = pMosaic->GetImage(iCam, iTrig);
			
			UIRect rect((unsigned int)piRectCols[iCam], (unsigned int)piRectRows[iTrig+1], 
				(unsigned int)(piRectCols[iCam+1]-1), (unsigned int)(piRectRows[iTrig]-1));
			// Validation check
			if(!rect.IsValid()) continue;

			pPanelImage->MorphFrom(pFov, rect);
		}
	}

	delete [] pdCenX;
	delete [] pdCenY;
	delete [] piRectRows;
	delete [] piRectCols;
}


//** for debug
void StitchingManager::SaveStitchingImages(string sName, unsigned int iNum, bool bCreateColorImg)
{

	// Create and stitched images for each illumination
	char cTemp[100];
	string s;
	for(unsigned int i=0; i<iNum; i++)
	{
		MosaicLayer *pLayer = _pMosaicSet->GetLayer(i);
		Image *pTempImg = pLayer->GetStitchedImage();
		pTempImg->ZeroBuffer();

		CreateStitchingImage(i, pTempImg);
	
		sprintf_s(cTemp, 100, "%s_%d.bmp", sName.c_str(), i); 
		s.clear();
		s.assign(cTemp);
		pTempImg->Save(s);
	}
	/*
	// Creat color images
	if(bCreateColorImg)
	{
		Image* pCadImage = _pOverlapManager->GetCadImage();
		Bitmap* rbg;

		switch(iNum)
		{
			case 1:
			{
				sprintf_s(cTemp, 100, "%s_color.bmp", sName.c_str()); 
				s.clear();
				s.assign(cTemp);
				
				if(pCadImage != NULL)	// With Cad image
				{
					rbg = Bitmap::New3ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[0].GetBuffer(),
						pPanelImages[0].GetBuffer(),
						pCadImage->GetBuffer(),
						iNumCols,
						iNumCols,
						iNumCols);
				}

				rbg->write(s);
				delete rbg;
			}
			break;

			case 2:
			{
				sprintf_s(cTemp, 100, "%s_color.bmp", sName.c_str()); 
				s.clear();
				s.assign(cTemp);

				if(pCadImage != NULL) // With Cad image
				{
					rbg = Bitmap::New3ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[0].GetBuffer(),
						pPanelImages[1].GetBuffer(),
						pCadImage->GetBuffer(),
						iNumCols,
						iNumCols,
						iNumCols);
				}
				else	// Without Cad image
				{
					rbg = Bitmap::New2ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[0].GetBuffer(),
						pPanelImages[1].GetBuffer(),
						iNumCols,
						iNumCols);
				}

				rbg->write(s);
				delete rbg;
			}
			break;

			case 4:
			{
				// Bright field before and after
				sprintf_s(cTemp, 100, "%s_Bright.bmp", sName.c_str()); 
				s.clear();
				s.assign(cTemp);

				if(pCadImage != NULL)	
				{
					rbg = Bitmap::New3ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[0].GetBuffer(),
						pPanelImages[2].GetBuffer(),
						pCadImage->GetBuffer(),
						iNumCols,
						iNumCols,
						iNumCols);
				}
				else
				{
					rbg = Bitmap::New2ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[0].GetBuffer(),
						pPanelImages[2].GetBuffer(),
						iNumCols,
						iNumCols);
				}

				rbg->write(s);
				delete rbg;

				// Dark field before and after
				sprintf_s(cTemp, 100, "%s_Dark.bmp", sName.c_str()); 
				s.clear();
				s.assign(cTemp);

				if(pCadImage != NULL)
				{
					rbg = Bitmap::New3ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[1].GetBuffer(),
						pPanelImages[3].GetBuffer(),
						pCadImage->GetBuffer(),
						iNumCols,
						iNumCols,
						iNumCols);
				}
				else
				{
					rbg = Bitmap::New2ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[1].GetBuffer(),
						pPanelImages[3].GetBuffer(),
						iNumCols,
						iNumCols);
				}

				rbg->write(s);
				delete rbg;

				// Before bright and dark fields
				sprintf_s(cTemp, 100, "%s_Before.bmp", sName.c_str()); 
				s.clear();
				s.assign(cTemp);

				if(pCadImage != NULL)
				{
					rbg = Bitmap::New3ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[0].GetBuffer(),
						pPanelImages[1].GetBuffer(),
						pCadImage->GetBuffer(),
						iNumCols,
						iNumCols,
						iNumCols);
				}
				else
				{
					rbg = Bitmap::New2ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[0].GetBuffer(),
						pPanelImages[1].GetBuffer(),
						iNumCols,
						iNumCols);
				}

				rbg->write(s);
				delete rbg;

				// After bright and dark fields
				sprintf_s(cTemp, 100, "%s_After.bmp", sName.c_str()); 
				s.clear();
				s.assign(cTemp);

				if(pCadImage != NULL)
				{
					rbg = Bitmap::New3ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[2].GetBuffer(),
						pPanelImages[3].GetBuffer(),
						pCadImage->GetBuffer(),
						iNumCols,
						iNumCols,
						iNumCols);
				}
				else
				{
					rbg = Bitmap::New2ChannelBitmap( 
						iNumRows, 
						iNumCols, 
						pPanelImages[2].GetBuffer(),
						pPanelImages[3].GetBuffer(),
						iNumCols,
						iNumCols);
				}

				rbg->write(s);
				delete rbg;
			}
			break;

			default:
				LOG.FireLogEntry(LogTypeSystem, "StitchingManager::SaveStitchingImages(): Illumination number is not supported");
		}
		*/
	//	delete [] pPanelImages;
	//}
}