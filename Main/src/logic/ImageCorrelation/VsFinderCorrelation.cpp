#include "VsFinderCorrelation.h"
#include "vswrapper.h"
#include "Logger.h"

VsFinderCorrelation::VsFinderCorrelation()
{
	_pVsw = new vswrapper();
}

VsFinderCorrelation::~VsFinderCorrelation()
{
	delete _pVsw;
}

// Create vsfinder template for a fiducial if it doesn't exist
// pFid: input, fiducial feature
// Return template ID for existing or new create vsfinder template
// Return -1 if failed
int VsFinderCorrelation::CreateVsTemplate(Feature* pFid)
{
	// If the template exists
	int iTemplateID = GetVsTemplateID(pFid);
	if(iTemplateID >= 0) return(iTemplateID);
	
	// Create a new template
	bool bFlag = CreateVsTemplate(pFid, &iTemplateID);
	if(!bFlag) return(-1);

	// Add new template into list
	FeatureTemplateID templateID(pFid, iTemplateID);
	_vsTemplateIDList.push_back(templateID);
	
	return(iTemplateID);
}

// Create vsfinder template for a fiducial
// pFid: input, fiducial feature
// pTemplateID: output, ID of vsfinder template
bool VsFinderCorrelation::CreateVsTemplate(
	Feature* pFid, 		
	int* pTemplateID)
{
	double dMinScale[2]={0.95, 0.95};
	double dMaxScale[2]={1.05, 1.05};

	switch(pFid->GetShape())
	{
	case Feature::SHAPE_CROSS:
		_pVsw->create_cross_template(pTemplateID, vswrapper::FIDUCIAL,
			((CrossFeature*)pFid)->GetSizeX(), ((CrossFeature*)pFid)->GetSizeY(),
			((CrossFeature*)pFid)->GetLegSizeX(), ((CrossFeature*)pFid)->GetLegSizeY(),
			0, 1, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_DIAMOND:
		_pVsw->create_diamond_template(pTemplateID, vswrapper::FIDUCIAL,
			((DiamondFeature*)pFid)->GetSizeX(), ((DiamondFeature*)pFid)->GetSizeY(),
			0, 1, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_DISC:
		_pVsw->create_disc_template(pTemplateID, vswrapper::FIDUCIAL,
			((DiscFeature*)pFid)->GetDiameter()/2,
			0, 1, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_DONUT:
		_pVsw->create_donut_template(pTemplateID, vswrapper::FIDUCIAL,
			((DonutFeature*)pFid)->GetDiameterInside()/2,((DonutFeature*)pFid)->GetDiameterOutside()/2,
			0, 1, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_RECTANGLE:
		_pVsw->create_rectangle_template(pTemplateID, vswrapper::FIDUCIAL,
			((RectangularFeature*)pFid)->GetSizeX(), ((RectangularFeature*)pFid)->GetSizeY(),
			0, 1, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_TRIANGLE:
		_pVsw->create_triangle_template(pTemplateID, vswrapper::FIDUCIAL,
			((TriangleFeature*)pFid)->GetSizeX(), ((TriangleFeature*)pFid)->GetSizeY(),
			((TriangleFeature*)pFid)->GetOffset(),
			0, 1, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_CYBER:
		LOG.FireLogEntry(LogTypeError, "CreateVsTemplate():unsupported fiducial type");
		break;

	default:
		LOG.FireLogEntry(LogTypeError, "CreateVsTemplate():unsupported fiducial type");	
	}

	return(true);
}

// Get the
int VsFinderCorrelation::GetVsTemplateID(Feature* pFeature)
{
	list<FeatureTemplateID>::const_iterator i;
	for(i=_vsTemplateIDList.begin(); i!=_vsTemplateIDList.end(); i++)
	{
		if(pFeature->GetShape() == i->_pFeature->GetShape()) // Type check
		{
			switch(pFeature->GetShape())// Fearure content check
			{
			case Feature::SHAPE_CROSS:
				{
					CrossFeature* pFeature1 = (CrossFeature*)pFeature;
					CrossFeature* pFeature2 = (CrossFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY() &&
						pFeature1->GetLegSizeX() == pFeature2->GetLegSizeX() &&
						pFeature1->GetLegSizeY() == pFeature2->GetLegSizeY())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_DIAMOND:
				{					
					DiamondFeature* pFeature1 = (DiamondFeature*)pFeature;
					DiamondFeature* pFeature2 = (DiamondFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_DISC:
				{
					DiscFeature* pFeature1 = (DiscFeature*)pFeature;
					DiscFeature* pFeature2 = (DiscFeature*)i->_pFeature;

					if(pFeature1->GetDiameter() == pFeature2->GetDiameter())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_DONUT:
				{
					DonutFeature* pFeature1 = (DonutFeature*)pFeature;
					DonutFeature* pFeature2 = (DonutFeature*)i->_pFeature;

					if(pFeature1->GetDiameterOutside() == pFeature2->GetDiameterOutside() &&
						pFeature1->GetDiameterInside() == pFeature2->GetDiameterInside())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_RECTANGLE:
				{
					RectangularFeature* pFeature1 = (RectangularFeature*)pFeature;
					RectangularFeature* pFeature2 = (RectangularFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_TRIANGLE:
				{
					TriangleFeature* pFeature1 = (TriangleFeature*)pFeature;
					TriangleFeature* pFeature2 = (TriangleFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY() &&
						pFeature1->GetOffset() == pFeature2->GetOffset())
						return(i->_iTemplateID);
				}
				break;

			default:
				break;
			}
		}
	}

	return(-1);
}