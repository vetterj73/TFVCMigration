// ManagedPanelAlignment.h

#pragma once

#include "PanelAligner.h"

using namespace System;
using namespace MMosaicDM; 
using namespace Cyber::SPIAPI;

namespace PanelAlignM {

	public ref class ManagedPanelAlignment
	{
		// TODO: Add your methods for this class here.

		//bool SetPanel(ManagedMosaicSet^ set, CPanel^ panel);
		//bool AddImage(
		//	unsigned int iLayerIndex, 
		//	unsigned int iTrigIndex, 
		//	unsigned int iCamIndex,
		//	unsigned char* pcBuf);

		//LoggableObject* GetLogger() {return &LOG;};


	private:
		PanelAligner* _pAlinger;
	};
}
