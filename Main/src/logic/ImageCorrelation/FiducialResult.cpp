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

///////////////////////////////////////////////////////
//	FiducialResultCheck Class
// Check the validation of fiducial alignment results
///////////////////////////////////////////////////////
FiducialResultCheck::FiducialResultCheck(PanelFiducialResultsSet* pFidSet, RobustSolver* pSolver)
{
	_pFidSet = pFidSet;
	_pSolver = pSolver;
}

// Check the alignment of fiducial, mark out any outlier, 
// return	1	: success
//			-1	: marked out Outliers only
//			-2	: not marked out exceptions only
//			-3	: Both Oulier and exception
//			-4	: All fiducial distance are out of scale range
int FiducialResultCheck::CheckFiducialResults()
{
	list<FiducialDistance> fidDisList;

	// Calculate all valid distances of all alignment pairs for different physical fiducials 
	int iNumPhyFid = _pFidSet->Size();
	for(int i=0; i<iNumPhyFid; i++)
	{
		for(int j=i+1; j<iNumPhyFid; j++) // j should be bigger than i
		{
			// Alignment results for two different physical fiducials 
			list<FidFovOverlap*>* pResults1 = _pFidSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
			list<FidFovOverlap*>* pResults2 = _pFidSet->GetPanelFiducialResultsPtr(j)->GetFidOverlapListPtr();

			// Calculate distance of two alignments for different physical fiducial based on transforms
			for(list<FidFovOverlap*>::iterator m = pResults1->begin(); m != pResults1->end(); m++)
			{
				for(list<FidFovOverlap*>::iterator n = pResults2->begin(); n != pResults2->end(); n++)
				{
					ImgTransform trans1 = _pSolver->GetResultTransform(
						(*m)->GetMosaicImage()->Index(), (*m)->GetTriggerIndex(), (*m)->GetCameraIndex());
					ImgTransform trans2 = _pSolver->GetResultTransform(
						(*n)->GetMosaicImage()->Index(), (*n)->GetTriggerIndex(), (*n)->GetCameraIndex());
					FiducialDistance fidDis(*m, trans1, *n, trans2);
					if(fidDis._bValid)
						fidDisList.push_back(fidDis);
				}
			}
		}
	}

	// Calculate normal Scale = stitched panle image/CAD image
		// Calcualte mean and variance
	int iCount = 0;
	double dSum = 0;
	double dSumSquare = 0;
	for(list<FiducialDistance>::iterator i = fidDisList.begin(); i != fidDisList.end(); i++)
	{
		if(i->_bValid)
		{
			double dScale = i->CalTranScale();
			if(fabs(dScale-1) < CorrelationParametersInst.dMaxPanelCadScaleDiff) // Ignore outliers
			{
				dSum += dScale;
				dSumSquare += dScale*dScale;
				iCount++;
			}
		}
	}

	if(fidDisList.size()-iCount > 0)
		LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): %d out of %d fiducial distance(s) on panel/CAD Scale is out of range", 
			fidDisList.size()-iCount, fidDisList.size()); 
	
		// Calculate normal scale
	double dNormScale = 1;
	if(iCount==0)					// Failed
		return(-4);
	else if(iCount==1 || iCount==2)	// 1 or 2 values available
		 dNormScale = dSum/iCount;
	else							// 3 or more values available
	{	// refine
		dSum /= iCount;
		dSumSquare /= iCount;
		double dVar = sqrt(dSumSquare - dSum*dSum);
		iCount = 0;
		double dSum2 = 0;
		for(list<FiducialDistance>::iterator i = fidDisList.begin(); i != fidDisList.end(); i++)
		{
			if(i->_bValid)
			{
				double dScale = i->CalTranScale();
				if(fabs(dScale-dSum) <= dVar)
				{
					dSum2 += dScale;
					iCount++;
				}
			}
		}
		dNormScale = dSum2/iCount;
	}

		// Adjust distance base on transform
	for(list<FiducialDistance>::iterator i = fidDisList.begin(); i != fidDisList.end(); i++)
	{
		if(i->_bValid)
		{
			i->NormalizeTransDis(dNormScale);
		}
	}

	// Mark alignment outlier out based on distance/scale check
	int iOutlierCount = 0;
	for(int i=0; i<iNumPhyFid; i++) // for each physical fiducial
	{
		list<FidFovOverlap*>* pResults = _pFidSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++) // For each alignment
		{
			// Outlier check
			int iCount1=0, iCount2=0;
			for(list<FiducialDistance>::iterator m = fidDisList.begin(); m != fidDisList.end(); m++)
			{
				if(m->_bValid && !m->_bFromOutlier && m->IsWithOverlap(*j))
				{
					iCount1++;
					double dScale = m->CalTranScale();
					if(fabs(1-dScale) > CorrelationParametersInst.dMaxFidDisScaleDiff)
					{ 
						iCount2++;
					}
				}	
			}

			// If it is an alignment outlier
			if(iCount2 >=2 && (double)iCount2/(double)iCount1>0.5)
			{
				LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): FidOverlap (Layer=%d, Trig=%d, Cam=%d) is outlier base on consistent check, %d out %d scale are out of rang", 
					(*j)->GetMosaicImage()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex(),
					iCount2, iCount1);

				// Mark all distances related to the aligmment outlier out
				for(list<FiducialDistance>::iterator m = fidDisList.begin(); m != fidDisList.end(); m++)
				{
					if(m->_bValid && !m->_bFromOutlier &&m->IsWithOverlap(*j))
					{
						m->_bFromOutlier = true;
					}
				}

				// The alignment outlier should not be used for solver
				(*j)->SetIsGoodForSolver(false);

				iOutlierCount++;
			}
		}
	}

	// Exception that is not marked as outlier
	int iExceptCount = 0;
	for(list<FiducialDistance>::iterator m = fidDisList.begin(); m != fidDisList.end(); m++)
	{
		if(m->_bValid && !m->_bFromOutlier && fabs(1-m->CalTranScale())>CorrelationParametersInst.dMaxFidDisScaleDiff)
			iExceptCount++;
	}
	if(iExceptCount>0)
		LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): There are %d exception distances not from marked outlier(s)", iExceptCount);

	// Check consistent of alignments for each physical fiducial
	// Assumption: at most one outlier exists for each physical fiducial
	for(int i=0; i<iNumPhyFid; i++)
	{
		list<FidFovOverlap*>* pResults = _pFidSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		if(pResults->size() == 1) // No consistent check can be done
			continue;

		iCount = 0;
		double dSumX=0, dSumY=0;
		double dSumXSq=0, dSumYSq=0;
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
		{
			if((*j)->IsProcessed() && (*j)->IsGoodForSolver() && (*j)->GetWeightForSolver()>0)
			{
				// Calcualte the fiducail location based on alignment
				ImgTransform trans = _pSolver->GetResultTransform(
					(*j)->GetMosaicImage()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex());
				double x, y;
				(*j)->CalFidCenterBasedOnTransform(trans, &x, &y);

				dSumX += x;
				dSumY += y;
				dSumXSq += x*x;
				dSumYSq += y*y;
				iCount++;
			}
		}
		if(iCount<=1) // No consistent check can be done
			continue;

		dSumX /= iCount;
		dSumY /= iCount;
		dSumXSq /= iCount;
		dSumYSq /= iCount;
		double dVarX= sqrt(dSumXSq-dSumX*dSumX);
		double dVarY= sqrt(dSumYSq-dSumY*dSumY);
		
		// If there is only one outlier and all other alignments are on the same physical location 
		// The outlier's physical position away from all other alignment = dAdjustScale*variance
		double dAdjustScale = (double)iCount/sqrt((double)iCount-1);
		double dAdjustDisX = dVarX*dAdjustScale;
		double dAdjustDisY = dVarY*dAdjustScale;

		// If count==2, only exceptions can be checked, no outlier can be identified
		if(iCount==2)
		{
			if(dAdjustDisX > CorrelationParametersInst.dMaxSameFidInConsist ||
				dAdjustDisY > CorrelationParametersInst.dMaxSameFidInConsist)
			{
				iExceptCount++;
				LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): Two alignments for fiducial #%d are inconsistent ", i);
			}
		}

		// Count >=3, mark outlier out if there is some
		if(iCount>=3)
		{
			// If there is outlier 
			if(dAdjustDisX > CorrelationParametersInst.dMaxSameFidInConsist ||
				dAdjustDisY > CorrelationParametersInst.dMaxSameFidInConsist)
			{
				for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
				{
					if((*j)->IsProcessed() && (*j)->IsGoodForSolver() && (*j)->GetWeightForSolver()>0)
					{
						// Calcualte the fiducail location based on alignment
						ImgTransform trans = _pSolver->GetResultTransform(
							(*j)->GetMosaicImage()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex());
						double x, y;
						(*j)->CalFidCenterBasedOnTransform(trans, &x, &y);

						// If it is outlier based on consistent check
						if((dAdjustDisX>CorrelationParametersInst.dMaxSameFidInConsist && fabs(x-dSumX)>dVarX) || 
							(dAdjustDisY>CorrelationParametersInst.dMaxSameFidInConsist && fabs(y-dSumY)>dVarY))
						{
							(*j)->SetIsGoodForSolver(false);
							iOutlierCount++;
							LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): FidOverlap (Layer=%d, Trig=%d, Cam=%d) is outlier base on consistent check", 
								(*j)->GetMosaicImage()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex()); 
						}
					}
				}
			}
		}
	}

	if(iOutlierCount>0 && iExceptCount==0)
		return(-1);
	else if(iOutlierCount==0 && iExceptCount>0)
		return(-2);
	else if(iOutlierCount>0 && iExceptCount>0)
		return(-3);
	else
		return(1);
}