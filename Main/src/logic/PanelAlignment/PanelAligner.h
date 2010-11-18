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

	// for Logger
	void Write(LOGTYPE LogType, const char* message);

private:
	MosaicSet* _pSet;	
	CorrelationFlags** _pCorrelationFlags;
	
	MosaicImage* _pMosaics;
	OverlapManager* _pOverlapManager;
	StitchingManager* _pStitchingManager;

	unsigned int _iNumIlluminations;

	//*** Warning:: should  be removef laer
	// for Logger
	FILE* m_logFile;
};

