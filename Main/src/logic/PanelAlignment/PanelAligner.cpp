#include "PanelAligner.h"

#include "StitchingManager.h"

#include "MosaicLayer.h"
#include "MosaicTile.h"

#include "Panel.h"

void MyWriteCallback(LOGTYPE LogType, const char* message, void* context)
{
	PanelAligner* pAligner = static_cast<PanelAligner*>(context);
	pAligner->Write(LogType, message);
}

PanelAligner::PanelAligner(void)
{
	_pOverlapManager = NULL;
	_pMosaics = NULL;
	_pCorrelationFlags = NULL;

	fopen_s(&m_logFile, "c:\\Temp\\StitchLog", "wt");
	LOG.SetAllLogTypes(true);
	//LOG.RegisterLoggingCallback();
}

PanelAligner::~PanelAligner(void)
{
	if(_pOverlapManager != NULL)
		delete _pOverlapManager;

	if(_pMosaics != NULL)
		delete [] _pMosaics;

	if(_pCorrelationFlags != NULL)
	{
		for(int i=0; i<_iNumIlluminations; i++)
			delete [] _pCorrelationFlags[i];

		delete [] _pCorrelationFlags;
	}

	fclose(m_logFile);
}

void PanelAligner::Write(LOGTYPE LogType, const char* message)
{
	fprintf(m_logFile, message);
}

// Set panel
bool PanelAligner::SetPanel(MosaicSet* pSet, Panel* _pPanel)
{
	if(_pOverlapManager != NULL)
		delete _pOverlapManager;

	if(_pMosaics != NULL)
		delete [] _pMosaics;

	_pSet = pSet;

	_iNumIlluminations = _pSet->GetNumMosaicLayers();
	_pMosaics = new MosaicImage[_iNumIlluminations];

	unsigned int iImWidth  = _pSet->GetImageWidthInPixels();
	unsigned int iImHeight = _pSet->GetImageHeightInPixels();
	unsigned int iImStride = _pSet->GetImageStrideInPixels();
	
	// Create moasic images
	unsigned int i, j, iCam, iTrig;
	for(i=0; i<_iNumIlluminations; i++)
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
	rect.yMax = _pSet->GetObjectWidthInMeters();
	_pOverlapManager = new OverlapManager(_pMosaics, _pCorrelationFlags, _iNumIlluminations, NULL, _pSet->GetNominalPixelSizeX(), _pPanel); // nee work

	// Create stitching manager
	_pStitchingManager = new StitchingManager(_pOverlapManager, NULL);

	LOG.FireLogEntry(LogTypeSystem, "Panel change over is done!");

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
		LOG.FireLogEntry(LogTypeSystem, "New Panel Begins!");
	}

	// Get image buffer
	unsigned char* pcBuf = _pSet->GetLayer(iLayerIndex)->GetTile(iCamIndex, iTrigIndex)->GetImageBuffer();
	if(pcBuf==NULL) return(false);

	// Add buffer to stitching manager
	_pStitchingManager->AddOneImageBuffer(pcBuf, iLayerIndex, iTrigIndex, iCamIndex);

	return(true);
}