#pragma once

#include "FeatureLocationCheck.h"
using namespace Cyber::MPanel;

namespace PanelAlignM {

	public ref class ManagedFeatureLocationCheck
	{
		public:
			ManagedFeatureLocationCheck(CPanel^ panel);

			bool CheckFeatureLocation(System::IntPtr pData, int iSpan, array<double>^ pdResults);

		private:
			FeatureLocationCheck* _pChecker;
			Panel* _pPanel;
	};

	public ref class ManagedImageFidAligner
	{
		public: 
			ManagedImageFidAligner(CPanel^ panel);

			bool CalculateTransform(System::IntPtr pData, int iSpan, array<double>^ trans);

			System::IntPtr MorphImage(System::IntPtr pDataIn, int iSpanIn);

		private:
			ImageFidAligner* _imageFidAligner;
			Panel* _pPanel;
	};
}
