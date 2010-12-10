/*
	This is an managed interface class of stitching tool 
*/

#pragma once

#include "PanelAligner.h"

using namespace System;
using namespace MMosaicDM; 
using namespace Cyber::MPanel;
using namespace MLOGGER;

namespace PanelAlignM {

	public ref class ManagedPanelAlignment : public MLoggableObject
	{
		// TODO: Add your methods for this class here.
	public:
		ManagedPanelAlignment();
		!ManagedPanelAlignment();

		bool ChangeProduction(ManagedMosaicSet^ set, CPanel^ panel);

		void ResetForNextPanel();

		bool AddImage(
			int iLayerIndex, 
			int iTrigIndex, 
			int iCamIndex);

		//LoggableObject* GetLogger() {return &LOG;};


	private:
		PanelAligner* _pAligner;
	};
}
