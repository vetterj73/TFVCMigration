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

	public ref class ManagedFidInfo
	{
		public:
			ManagedFidInfo(double nominalXPosition, double nominalYPosition,
				double rowDifference, double columnDifference, double correlationScore, double ambiguityScore)
			{
				_nominalXPosition = nominalXPosition;
				_nominalYPosition = nominalYPosition;
				_rowDifference = rowDifference;
				_columnDifference = columnDifference;
				_correlationScore = correlationScore;			
				_ambiguityScore = ambiguityScore;			
			}
			
			double GetNominalXPosition(){return _nominalXPosition;};
			double GetNominalYPosition(){return _nominalYPosition;};
			double GetRowDifference(){return _rowDifference;};
			double GetColumnDifference(){return _columnDifference;};
			double GetCorrelationScore(){return _correlationScore;};
			double GetAmbiguityScore(){return _ambiguityScore;};

		protected:
			double _nominalXPosition;
			double _nominalYPosition;
			double _rowDifference;
			double _columnDifference;
			double _correlationScore;
			double _ambiguityScore;
	};

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
		void FiducialSearchExpansionXInMeters(double fidSearchXInMeters);
		void FiducialSearchExpansionYInMeters(double fidSearchYInMeters);
		int GetNumberOfFidsProcessed();
		ManagedFidInfo^ GetFidAtIndex(unsigned int index);

	private:
		PanelAligner* _pAligner;

	};
}
