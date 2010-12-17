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

	bool SaveStitchedImage(int layer, System::String^ imagePath);
	bool Save3ChannelImage(int layerInChannel1, int layerInChannel2, bool panelCadInLayer3, System::String^ imagePath);

	private:
		PanelAligner* _pAligner;
	};
}
