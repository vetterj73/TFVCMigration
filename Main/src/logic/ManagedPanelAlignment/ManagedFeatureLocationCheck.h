#pragma once

#include "FeatureLocationCheck.h"
using namespace Cyber::MPanel;

namespace PanelAlignM {

	public ref class ManagedFeatureLocationCheck
	{
		public:
			ManagedFeatureLocationCheck(CPanel^ panel);

			bool CheckFeatureLocation(System::IntPtr pData, array<double>^ pdResults);

		private:
			FeatureLocationCheck* _pChecker;
			Panel* _pPanel;
	};

}
