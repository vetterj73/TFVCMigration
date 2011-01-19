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

	if(_pOverlapManager != NULL)
		delete _pOverlapManager;

	if(_pStitchingManager != NULL)
		delete _pStitchingManager;

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

	unsigned char* pCadBuf = _pPanel->GetCadBuffer();
	unsigned char* pPanelMaskBuf = _pPanel->GetMaskBuffer(); 

	_pOverlapManager = new OverlapManager(_pSet, _pPanel);
		
	// Create stitching manager @todo - this does not belong here...
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
	_pStitchingManager->ImageAdded(iLayerIndex, iTrigIndex, iCamIndex);

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::AddImage():Fov Layer=%d Trig=%d Cam=%d added!", iLayerIndex, iTrigIndex, iCamIndex);

	return(true);
}