#pragma once

#include "MosaicSet.h"
#include "Logger.h"

using namespace MosaicDM;

class MosaicImage;
class OverlapManager;
class StitchingManager;
class Panel;

class PanelAligner
{
public:
	PanelAligner(void);
	~PanelAligner(void);

	bool SetPanel(MosaicSet* pSet, Panel *pPanel);

	bool AddImage(
		unsigned int iLayerIndex, 
		unsigned int iTrigIndex, 
		unsigned int iCamIndex);

	LoggableObject* GetLogger() {return &LOG;};

private:
	MosaicSet* _pSet;	
	CorrelationFlags** _pCorrelationFlags;
	
	MosaicImage* _pMosaics;
	OverlapManager* _pOverlapManager;
	StitchingManager* _pStitchingManager;

	unsigned int _iNumIlluminations;
};

