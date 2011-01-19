/*
	This is an unmanaged interface class of stitching tool 
*/
#pragma once

#include "MosaicSet.h"
#include "Logger.h"

using namespace MosaicDM;

class OverlapManager;
class StitchingManager;
class Panel;

class PanelAligner
{
public:
	PanelAligner(void);
	~PanelAligner(void);

	bool ChangeProduction(MosaicSet* pSet, Panel *pPanel);

	void ResetForNextPanel();

	bool AddImage(
		unsigned int iLayerIndex, 
		unsigned int iTrigIndex, 
		unsigned int iCamIndex);

	LoggableObject* GetLogger() {return &LOG;};

protected:
	// CleanUp internal stuff for new production or desctructor
	void CleanUp();

private:
	// Inputs
	MosaicSet* _pSet;		
	Panel* _pPanel;
	OverlapManager* _pOverlapManager;
	StitchingManager* _pStitchingManager;
};

