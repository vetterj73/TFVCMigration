#pragma once

//#include "Image.h"
#include "opencv\cxcore.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "Image.h"
#include "EdgeDetectUtil.h"

#include <list>
using std::list;


class PanelEdgeDetection
{
public:
	PanelEdgeDetection(void);
	~PanelEdgeDetection(void);

	static bool FindLeadingEdge(Image* pImage, StPanelEdgeInImage* ptParam);
};

