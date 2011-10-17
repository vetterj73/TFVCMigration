#pragma once
#include "Panel.h";

class FeatureLocationCheck
{
public:
	FeatureLocationCheck(Panel* pPanel);
	~FeatureLocationCheck(void);

	bool CheckFeatureLocation(Image* pImage, double d[]);

private:
	Panel* _pPanel;
	int* _piTemplateIds;
};

