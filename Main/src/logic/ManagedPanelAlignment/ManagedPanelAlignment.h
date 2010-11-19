// ManagedPanelAlignment.h

#pragma once

#include "PanelAligner.h"

using namespace System;
using namespace MMosaicDM; 
using namespace Cyber::SPIAPI;
using namespace MLOGGER;

namespace PanelAlignM {

	public ref class ManagedPanelAlignment : public MLoggableObject
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
