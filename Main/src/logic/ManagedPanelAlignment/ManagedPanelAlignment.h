/*
	This is an managed interface class of stitching tool 
*/

#pragma once

#include "PanelAligner.h"

using namespace System;
using namespace MMosaicDM; 
using namespace Cyber::MPanel;
using namespace MLOGGER;
using namespace System::Collections::Generic;

namespace PanelAlignM {

	public delegate void AlignmentDoneDelegate(bool status);

	public ref class ManagedFidInfo
	{
		public:
			ManagedFidInfo(double nominalXPositionInMeters, double nominalYPositionInMeters,
				double xOffsetInMeters, double yOffsetInMeters, double correlationScore, double ambiguityScore)
			{
				_nominalXPosition = nominalXPositionInMeters;
				_nominalYPosition = nominalYPositionInMeters;
				_xOffset = xOffsetInMeters;
				_yOffset = yOffsetInMeters;
				_correlationScore = correlationScore;			
				_ambiguityScore = ambiguityScore;			
			}
			
			double GetNominalXPositionInMeters(){return _nominalXPosition;};
			double GetNominalYPositionInMeters(){return _nominalYPosition;};
			double GetXOffsetInMeters(){return _xOffset;};
			double GetYOffsetInMeters(){return _yOffset;};
			double GetCorrelationScore(){return _correlationScore;};
			double GetAmbiguityScore(){return _ambiguityScore;};

		protected:
			double _nominalXPosition;
			double _nominalYPosition;
			double _xOffset;
			double _yOffset;
			double _correlationScore;
			double _ambiguityScore;
	};

	public ref class ManagedFidResult
	{
		public:
			int iLayerIndex;
			int iTrigIndex;
			int iCamIndex;
			double rowOffset;
			double colOffset;
			double correlationScore;
			double ambiguityScore;
			double weight;
	};

	public ref class ManagedPanelFidResults
	{
	public:
		List<ManagedFidResult^> ^results;
		double dConfidence;
		double dCadX;
		double dCadY;
		int iID;
	};

	public ref class ManagedPanelFidResultsSet
	{
	public:
		List<ManagedPanelFidResults^> ^resultsSet;
		double dConfidence;
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

		event AlignmentDoneDelegate^ OnAlignmentDone;

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

		bool Save3ChannelImage(System::String^ imagePath,
			System::IntPtr pChannel1, int iSpan1, 
			System::IntPtr pChannel2, int iSpan2,
			System::IntPtr pChannel3, int iSpan3,
			int numRows, int numColumns);

		void NumThreads(unsigned int numThreads);
		void LogFiducialOverlaps(bool bLog);
		void LogOverlaps(bool bLog);
		void LogTransformVectors(bool bLog);	
		void FiducialSearchExpansionXInMeters(double fidSearchXInMeters);
		void FiducialSearchExpansionYInMeters(double fidSearchYInMeters);
		void UseCyberNgc4Fiducial();
		void UseProjectiveTransform(bool bValue);
		void UseCameraModelStitch(bool bValue);
		void EnableFiducialAlignmentCheck(bool bValue);
		void SetPanelEdgeDetection(
			bool bDetectPanelEdge, 
			bool bConveyorLeft2Right,
			bool bConveyorFixedFrontRail);
		void SetCalibrationWeight(double dValue);
		
		int GetNumberOfFidsProcessed();
		
		ManagedFidInfo^ GetFidAtIndex(unsigned int index);

		ManagedPanelFidResultsSet^ GetFiducialResultsSet();

	private:
		PanelAligner* _pAligner;
		double _pixelSizeX;
		double _pixelSizeY;

		AlignmentDoneDelegate ^_alignmentDoneDelegate;
		void RaiseAlignmentDone(bool status)
		{
			OnAlignmentDone(status);
		}
	};
}
