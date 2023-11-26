#pragma once

#include "Panel.h"

class vswrapper;
class VsFinderCorrelation
{
public:
	VsFinderCorrelation(double dResolution, unsigned int iColumn, unsigned int iRows);
	~VsFinderCorrelation();

	int CreateVsTemplate(Feature* pFid);

	// load the template by a given name and find the match in the given image.
	void Find(
		int iNodeID,			// map ID of template  and finder		
		unsigned char *image,   // buffer containing the image
		int width,              // width of the image in pixels
		int height,             // height of the image in pixels
		double &x,              // returned x location of the center of the template from the origin
		double &y,              // returned x location of the center of the template from the origin
		double &correlation,    // match score 0-1
		double &ambig,          // ratio of (second best/best match) score 0-1
		double *ngc,			// Normalized Grayscale Correlation Score 0-1
		double search_center_x, // x center of the search region in pixels
		double search_center_y, // y center of the search region in pixels
		double search_width,    // width of the search region in pixels
		double search_height,   // height of the search region in pixels
		double time_out,        // number of seconds to search maximum. If limit is reached before any results found, an error will be generated
		int y_origin_ll=1,      // If !0 origin in start of the image is 0,0 increasing in Y and X as you move forward
		double min_accept=-1,   // If >0 minimum score to persue at min pyramid level to look for peak override
		double max_accept=-1,   // If >0 minumum score to accept at max pyramid level to look for peak override
		int num_finds=-1        // If > 0 number of versions of the template to find
	);

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