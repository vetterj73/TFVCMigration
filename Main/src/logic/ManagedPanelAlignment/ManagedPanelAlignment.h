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
	public:
		ManagedPanelAlignment();
		!ManagedPanelAlignment();

		bool SetPanel(ManagedMosaicSet^ set, CPanel^ panel);

		bool AddImage(
			int iLayerIndex, 
			int iTrigIndex, 
			int iCamIndex);

		//LoggableObject* GetLogger() {return &LOG;};


	private:
		PanelAligner* _pAligner;
	};
}
