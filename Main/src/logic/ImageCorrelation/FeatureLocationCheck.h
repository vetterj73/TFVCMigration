#pragma once
#include "Panel.h";

// This class is to find the locations of fiducials on the panel (image)
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

