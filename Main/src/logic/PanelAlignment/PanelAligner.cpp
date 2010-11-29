#include "PanelAligner.h"

#include "StitchingManager.h"

#include "MosaicLayer.h"
#include "MosaicTile.h"

#include "Panel.h"

PanelAligner::PanelAligner(void)
{
	_pMosaics = NULL;
	_pCorrelationFlags = NULL;
	_pOverlapManager = NULL;
	_pStitchingManager = NULL;
}

PanelAligner::~PanelAligner(void)
{
	CleanUp();
}

// CleanUp internal stuff for new production or desctructor
void PanelAligner::CleanUp()
{
	if(_pMosaics != NULL)
		delete [] _pMosaics;

	if(_pCorrelationFlags != NULL)
	{
		for(unsigned int i=0; i<_iNumIlluminations; i++)
			delete [] _pCorrelationFlags[i];

		delete [] _pCorrelationFlags;
	}

	if(_pOverlapManager != NULL)
		delete _pOverlapManager;

	if(_pStitchingManager != NULL)
		delete _pStitchingManager;

	_pMosaics = NULL;
	_pCorrelationFlags = NULL;
	_pOverlapManager = NULL;
	_pStitchingManager = NULL;
}

// Set panel
bool PanelAligner::SetPanel(MosaicSet* pSet, Panel* pPanel)
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::SetPanel():Begin panel change over");
	// CleanUp internal stuff for new production
	CleanUp();

	_pSet = pSet;
	_pPanel = pPanel;

	// Create moasic images
	_iNumIlluminations = _pSet->GetNumMosaicLayers();
	_pMosaics = new MosaicImage[_iNumIlluminations];

	unsigned int iImWidth  = _pSet->GetImageWidthInPixels();
	unsigned int iImHeight = _pSet->GetImageHeightInPixels();
	unsigned int iImStride = _pSet->GetImageStrideInPixels();
	
	unsigned int i, j, iCam, iTrig;
	for(i=0; i<_iNumIlluminations; i++)	// for each mosaic image
	{
		MosaicLayer* pLayer = _pSet->GetLayer(i);
		unsigned int iNumTrigs = pLayer->GetNumberOfTriggers();
		unsigned int iNumCams = pLayer->GetNumberOfCameras();
		bool bUseCad = pLayer->IsUseCad();

		_pMosaics[i].Config(i, iNumCams, iNumTrigs, iImWidth, iImHeight, iImStride, bUseCad);

		for(iTrig=0; iTrig<iNumTrigs; iTrig++)
		{
			for(iCam=0; iCam<iNumCams; iCam++)
			{
				MosaicTile* pTile = pLayer->GetTile(iCam, iTrig);
				ImgTransform t = pTile->GetNominalTransform();
				// Set both nominal and regular transform
				_pMosaics[i].SetImageTransforms(t, iCam, iTrig);
			}
		}
	}

	// New and set correlation flag
	_pCorrelationFlags = new CorrelationFlags*[_iNumIlluminations];
	for(i=0; i<_iNumIlluminations; i++)
	{
		_pCorrelationFlags[i] = new CorrelationFlags[_iNumIlluminations];
	}

	for(i=0; i<_iNumIlluminations; i++)
		for(j=0; j<_iNumIlluminations; j++)
			_pCorrelationFlags[i][j] = *_pSet->GetCorrelationFlags(i, j);

	// Create Overlap manager
	DRect rect;
	rect.xMin = 0;
	rect.xMax = _pSet->GetObjectWidthInMeters();
	rect.yMin = 0;
	rect.yMax = _pSet->GetObjectLengthInMeters();
	_pOverlapManager = new OverlapManager(_pMosaics, _pCorrelationFlags, _iNumIlluminations, NULL, _pSet->GetNominalPixelSizeX(), _pPanel); // nee work

	// Create stitching manager 
	// (Mask image is NULL at this time)
	_pStitchingManager = new StitchingManager(_pOverlapManager, NULL);

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::SetPanel():Panel change over is done");

	return(true);
}

bool PanelAligner::AddImage(
	unsigned int iLayerIndex, 
	unsigned int iTrigIndex, 
	unsigned int iCamIndex)
{
	// If this is the first image of the cycle, reset
	if(iLayerIndex==0 && iTrigIndex==0 && iCamIndex==0)
	{
		_pStitchingManager->Reset();
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::AddImage():New Panel Begins!");
	}

	// Get image buffer
	unsigned char* pcBuf = _pSet->GetLayer(iLayerIndex)->GetTile(iCamIndex, iTrigIndex)->GetImageBuffer();
	if(pcBuf==NULL)
	{
		LOG.FireLogEntry(LogTypeError, "PanelAligner::AddImage():Fov Layer=%d Trig=%d Cam=%d Buffer is invalid!", iLayerIndex, iTrigIndex, iCamIndex);
		return(false);
	}

	// Add buffer to stitching manager
	_pStitchingManager->AddOneImageBuffer(pcBuf, iLayerIndex, iTrigIndex, iCamIndex);

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::AddImage():Fov Layer=%d Trig=%d Cam=%d added!", iLayerIndex, iTrigIndex, iCamIndex);

	return(true);
}