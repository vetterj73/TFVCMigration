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

	/*
	public ref class ManagedFidInfo
	{
		public:
			ManagedFidInfo();

			double GetNominalXPosition(){return _nominalXPosition;};
			double GetNominalYPosition(){return _nominalYPosition;};
			double RowDifference(){return _rowDifference;};
			double ColumnDifference(){return _columnDifference;};
			double CorrelationScore(){return _correlationScore;};

		protected:
			double _nominalXPosition;
			double _nominalYPosition;
			double _rowDifference;
			double _columnDifference;
			double _correlationScore;
	};
	*/

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
		bool ChangeProduction(ManagedMosaicSet^ set, CPanel^ panel);

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

		void NumThreads(unsigned int numThreads);
		void LogFiducialOverlaps(bool bLog);
		void LogOverlaps(bool bLog);
		void LogMaskVectors(bool bLog);	

		int GetNumberOfFidsProcessed();

	private:
		PanelAligner* _pAligner;
	};
}
