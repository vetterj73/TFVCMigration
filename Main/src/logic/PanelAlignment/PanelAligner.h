#pragma once

#include "MosaicImage.h"
#include "OverlapManager.h"
#include "StitchingManager.h"

#include "MosaicSet.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"
#include "Panel.h"

#include "Logger.h"

using namespace MosaicDM;

class PanelAligner
{
public:
	PanelAligner(void);
	~PanelAligner(void);

	bool SetPanel(MosaicSet* pSet, Panel *pPanel);
	bool AddImage(
		unsigned int iLayerIndex, 
		unsigned int iTrigIndex, 
		unsigned int iCamIndex,
		unsigned char* pcBuf);

	LoggableObject* GetLogger() {return &LOG;};

private:
	MosaicSet* _pSet;	
	CorrelationFlags** _pCorrelationFlags;
	
	MosaicImage* _pMosaics;
	OverlapManager* _pOverlapManager;
	StitchingManager* _pStitchingManager;

	unsigned int _iNumIlluminations;
};

