#include "FiducialResult.h"
#include "Logger.h"
#include "EquationWeights.h"

///////////////////////////////////////////////////////////////
//////		PanelFiducialResults Class
///////////////////////////////////////////////////////////////
PanelFiducialResults::PanelFiducialResults(void)
{
	_pFeature = NULL;
	_fidFovOverlapPointList.clear();
}

PanelFiducialResults::~PanelFiducialResults(void)
{
}

void PanelFiducialResults::LogResults()
{
	// Validation check
	if(_pFeature != NULL)
	{
		LOG.FireLogEntry(LogTypeDiagnostic, "Fiducial ID(%d); location(%d, %d)mm", 
			_pFeature->GetId(), (int)(_pFeature->GetCadX()*1000), (int)(_pFeature->GetCadY()*1000));
	}

	for(list<FidFovOverlap*>::iterator i = _fidFovOverlapPointList.begin(); i != _fidFovOverlapPointList.end(); i++)
	{
		if((*i)->GetCoarsePair()->IsProcessed())
		{
			CorrelationResult result = (*i)->GetCoarsePair()->GetCorrelationResult();
			int iWeight = (int)(Weights.CalWeight((*i)->GetCoarsePair()));
			LOG.FireLogEntry(LogTypeDiagnostic, "	Illumintion(%d), Trigger(%d), Camera(%d): CorrScore(%d), Ambig(%d), offset(%d,%d)pixels, weight(%d)",
				(*i)->GetMosaicImage()->Index(), (*i)->GetTriggerIndex(), (*i)->GetCameraIndex(),
				(int)(result.CorrCoeff*100), (int)(result.AmbigScore*100), (int)(result.ColOffset), (int)(result.RowOffset),
				(int)(Weights.CalWeight((*i)->GetCoarsePair())));
		}
	}

}

double PanelFiducialResults::CalConfidence()
{
	double dConfidenceScore = 0;

	// Pick the higheset one for each physical fiducial
	for(list<FidFovOverlap*>::iterator i = _fidFovOverlapPointList.begin(); i != _fidFovOverlapPointList.end(); i++)
	{
		if((*i)->IsProcessed() && (*i)->IsGoodForSolver() && (*i)->GetWeightForSolver()>0)
		{
			CorrelationResult result = (*i)->GetCoarsePair()->GetCorrelationResult();
			double dScore = result.CorrCoeff*(1-result.AmbigScore);

			if(dConfidenceScore < dScore)
				dConfidenceScore = dScore;
		}
	}

	return(dConfidenceScore);
}

///////////////////////////////////////////////////////////////
//////		PanelFiducialResultsSet Class
///////////////////////////////////////////////////////////////
PanelFiducialResultsSet::PanelFiducialResultsSet(unsigned int iSize)
{
	_pResultSet = NULL;

	_iSize = iSize;
	if(iSize=0) 
	{
		LOG.FireLogEntry(LogTypeError, "PanelFiducialResultsSet: The input size should not be 0");
		return;
	}

	_pResultSet = new PanelFiducialResults[_iSize];
}

PanelFiducialResultsSet::~PanelFiducialResultsSet()
{
	if(_pResultSet != NULL)
		delete [] _pResultSet;
}

void PanelFiducialResultsSet::LogResults()
{
	for(int i=0; i<_iSize; i++)
		_pResultSet[i].LogResults();

	double dConfidence = CalConfidence();
	LOG.FireLogEntry(LogTypeDiagnostic, "Confidence = %d/100", (int)(dConfidence*100));
}

// Calculate confidence based on the fid
double PanelFiducialResultsSet::CalConfidence()
{
	// Calculate confidence for each physical fiducial
	list<double> dConfidenceList;
	for(int i=0; i<_iSize; i++)
	{
		double dConfidence = _pResultSet[i].CalConfidence();
		dConfidenceList.push_back(dConfidence);
	}
	dConfidenceList.sort();

	// Decide overall confidence
	switch(_iSize)
	{
	case 0:
	case 1:
		return(0);	// No confidence at all
	break;
	
	case 2:
	case 3:
		return(*dConfidenceList.begin()); // The smallest one
	break;

	default:	// 4 or more
		int iCount = 0;
		for(list<double>::iterator i = dConfidenceList.begin(); i!=dConfidenceList.end(); i++)
		{
			if(iCount == _iSize-3) // The third biggest one
				return(*i);
			
			iCount++;
		}
	}

	return(0); // should never reach here
}

//////////////////////////////////////////////////////
//	FiducialDistance class
// World distance between two fiducials in the fiducial overlap
//////////////////////////////////////////////////////////////////
FiducialDistance::FiducialDistance(
	FidFovOverlap* pFidOverlap1,
	ImgTransform trans1,
	FidFovOverlap* pFidOverlap2,
	ImgTransform trans2)
{	
	_bValid = false;
	_bNormalized = false;
	_bFromOutlier = false;
	_pFidOverlap1 = pFidOverlap1;
	_pFidOverlap2 = pFidOverlap2;

	// Validation check
	if(pFidOverlap1==NULL || pFidOverlap2==NULL)
		return;
	
	// Validation check for correlation pair	
	if(pFidOverlap1->GetWeightForSolver()<=0)
		return;
	if(pFidOverlap2->GetWeightForSolver()<=0)
		return;

	// Distance based on CAD
	double dx = _pFidOverlap1->GetFiducialXPos()-_pFidOverlap2->GetFiducialXPos();
	double dy = _pFidOverlap1->GetFiducialYPos()-_pFidOverlap2->GetFiducialYPos();
	_dCadDis = sqrt(dx*dx+dy*dy);

	// Distance based on transform
	double x1, y1, x2, y2;
	if(!pFidOverlap1->CalFidCenterBasedOnTransform(trans1, &x1, &y1))
		return;
	if(!pFidOverlap2->CalFidCenterBasedOnTransform(trans2, &x2, &y2))
		return;
	_dTransDis = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));

	_bValid = true;
}

// Whether object is for certain overlap
bool FiducialDistance::IsWithOverlap(FidFovOverlap* pFidOverlap)
{
	if((pFidOverlap==_pFidOverlap1)  || (pFidOverlap==_pFidOverlap2))
		return(true);
	else
		return(false);
}

// Scale between the distance based on transform and based on Cad
double FiducialDistance::CalTranScale()
{
	// Validation check
	if(!_bValid) return(-1);
	if(_dCadDis < 0.001) return(-1);
	
	return(_dTransDis/_dCadDis);
}

// Normalize the distance based on the transform
void FiducialDistance::NormalizeTransDis(double dScale)
{
	// Validation check
	if(!_bValid) return;
	
	_dTransDis /= dScale;
	_bNormalized = true;
}

