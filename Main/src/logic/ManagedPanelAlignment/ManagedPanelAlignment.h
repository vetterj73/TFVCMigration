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

	///
	///	Simple Managed Wrapper for Panel Alignment...
	///
	public ref class ManagedPanelAlignment : public MLoggableObject
	{
	public:
		ManagedPanelAlignment();
		!ManagedPanelAlignment();
		~ManagedPanelAlignment();

		///
		///	Sets up for a new "Run"
		///
		bool ChangeProduction(ManagedMosaicSet^ set, CPanel^ panel, unsigned int numThreads);

		///
		///	Reset for the next cycle
		///
		void ResetForNextPanel();

		///
		///	A way to save a 3 channel bitmap.  Doesn't necessarily belong here, 
		/// but didn't have another place to put it...
		///
		bool Save3ChannelImage(System::String^ imagePath,
			System::IntPtr pChannel1, System::IntPtr pChannel2,	System::IntPtr pChannel3, 
			int numRows, int numColumns);

	private:
		PanelAligner* _pAligner;
	};
}
