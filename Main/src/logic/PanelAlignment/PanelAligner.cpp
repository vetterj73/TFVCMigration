#include "PanelAligner.h"

#include "StitchingManager.h"

#include "MosaicLayer.h"
#include "MosaicTile.h"

#include "Panel.h"

void ImageAdded(int layerIndex, int cameraIndex, int triggerIndex, void* context)
{
	PanelAligner *pPanelAlign = (PanelAligner *)context;
	pPanelAlign->AddImage(layerIndex, triggerIndex, cameraIndex);
}

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

// Change production
bool PanelAligner::ChangeProduction(MosaicSet* pSet, Panel* pPanel)
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Begin panel change over");
	// CleanUp internal stuff for new production
	CleanUp();

	_pSet = pSet;
	_pPanel = pPanel;

	_pSet->RegisterImageAddedCallback(ImageAdded, this);

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
		bool bAlignWithCad = pLayer->IsAlignWithCad();
		bool bAlignWithFiducial = pLayer->IsAlignWithFiducial();

		_pMosaics[i].Config(i, iNumCams, iNumTrigs, iImWidth, iImHeight, iImStride, bAlignWithCad, bAlignWithFiducial);

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

	unsigned char* pCadBuf = _pPanel->GetCadBuffer();
	unsigned char* pPanelMaskBuf = _pPanel->GetMaskBuffer(); 

	_pOverlapManager = new OverlapManager(
		_pMosaics, _pCorrelationFlags, _iNumIlluminations, 
		_pPanel, _pSet->GetNominalPixelSizeX(),
		pCadBuf, pPanelMaskBuf);
		
	// Create stitching manager 
	_pStitchingManager = new StitchingManager(_pOverlapManager);

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Panel change over is done");

	return(true);
}

//Reset for next panel
void PanelAligner::ResetForNextPanel()
{
	_pStitchingManager->Reset();
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ResetForNextPanel()");
}

// Add single image
bool PanelAligner::AddImage(
	unsigned int iLayerIndex, 
	unsigned int iTrigIndex, 
	unsigned int iCamIndex)
{
	// If this is the first image of the cycle, reset
	if(iLayerIndex==0 && iTrigIndex==0 && iCamIndex==0)
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::AddImage():New Panel Begins!");
	}

	// Get image buffer
	unsigned char* pcBuf = _pSet->GetLayer(iLayerIndex)->GetTile(iCamIndex, iTrigIndex)->GetBuffer();
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

bool PanelAligner::SaveStitchedImage(int layer, string imagePath)
{
	if(!_pStitchingManager->ResultsReady())
		return false;

	return false;
}

bool PanelAligner::Save3ChannelImage(int layerInChannel1, int layerInChannel2, bool panelCadInLayer3, string imagePath)
{
	if(!_pStitchingManager->ResultsReady())
		return false;

	return false;
}