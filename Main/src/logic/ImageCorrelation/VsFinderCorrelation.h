#pragma once

#include "Panel.h"

class vswrapper;
class VsFinderCorrelation
{
public:
	VsFinderCorrelation();
	~VsFinderCorrelation();

	int CreateVsTemplate(Feature* pFid);

protected:
	bool CreateVsTemplate(Feature* pFid, int* pTemplateID);
	int GetVsTemplateID(Feature* pFeature);

private:
	class FeatureTemplateID
	{
	public:
		FeatureTemplateID(Feature* pFeature ,unsigned int iTemplateID)
		{
			_pFeature = pFeature;
			_iTemplateID = iTemplateID;
		}
		Feature* _pFeature;
		unsigned int _iTemplateID;
	};

	list<FeatureTemplateID> _vsTemplateIDList;
	vswrapper* _pVsw;
};