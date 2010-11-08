#include "SquareRootCorrelation.h"
#include "regoff.h"
#include "CorrelationPair.h"

// Do the square root correlation and report result
// overlap: input, overlap for correlation calculation
// decimation_factor: decimation for correlation calculation
// bEnableNegCorr: All negative correlation
bool SqRtCorrelation(CorrelationPair* pCorrPair, unsigned int decimation_factor, bool bEnableNegCorr)
{
	// Calculate number of rows and columns.  
	// Must adjust for decimation requirements.

	/*
	   Note: Due to the fact that a 2-D FFT is taken of the
	   the decimated image, there are restrictions on the
	   permissible values of nrows and ncols.  They must
	   each be factorizable as decim*2^p * 3^q * 5^r.
	   Furthermore, performance is poor if the image
	   dimensions are less than 2*HOOD*decim.  A practical
	   option for images of unsuitable dimensions is to
	   register based on the largest feasible subsection of
	   the image.  Subroutine RegoffLength() is provided to
	   assist in computing suitable dimensions.
	*/

	Image* pImgA = pCorrPair->GetFirstImg();
	Image* pImgB = pCorrPair->GetSecondImg();

	int   nrows = pCorrPair->Rows();
	int   ncols = pCorrPair->Columns();

	nrows = RegoffLength(nrows, decimation_factor);
	ncols = RegoffLength(ncols, decimation_factor);

	/* Pointer to first image  */
	Byte* first_image_buffer =
		pImgA->GetBuffer( pCorrPair->GetFirstRoi().FirstRow, pCorrPair->GetFirstRoi().FirstColumn);

	/* Pointer to second image */
	Byte* second_image_buffer =
		pImgB->GetBuffer( pCorrPair->GetSecondRoi().FirstRow, pCorrPair->GetSecondRoi().FirstColumn);

	// it is assumed that the row stride is the same for both images
	int RowStrideA(pImgA->PixelRowStride());
	int RowStrideB(pImgB->PixelRowStride());

	/*
		Pointer to space large enough to contain complexf
		array of size at least ncols*nrows/decimx.  The
		array is filled with the correlogram in the .r member
		of each element.  If null pointers are passed for
		this argument, a local array is allocated and freed.
	*/
	complexf *z(0);			

	int      decimx(decimation_factor);     /* Decimation factors.  Currently-allowable values */
	int      decimy(decimation_factor);     /* are 1, 2, or 4. */

	REGLIM  *lims(0);   /* Limits of search range for registration offset.  Use
						   if there is _a priori_ knowledge about the offset.
						   If a null pointer is passed, the search range
						   defaults to the entire range of values

							  x = [-ncols/2, ncols/2>
							  y = [-nrows/2, nrows/2>

						   Excessively large values are clipped to the above
						   range; inconsistent values result in an error return.
						*/

	int      job(0);   /* 1 = histogram equalize images, 0 = no EQ */
	
	// Enable negative corrlation 
	if(bEnableNegCorr) 
		job |= REGOFF_ABS; 

	int      histclip(1);   /* Histogram clipping factor; clips peaks to prevent
						   noise in large flat regions from being excessively
						   amplified.  Use histclip=1 for no clipping;
						   histclip>1 for clipping.  Recommended value = 32 */

	int      dump(0);       /* Dump intermediate images to TGA files
						   (useful for debugging):

						   ZR.TGA and ZI.TGA are decimated (and possibly
							  histogram-equalized images that are input to the
							  correlation routine.

						   PCORR.TGA is the correlogram.

						   HOLE.TGA is the correlogram, excluding the vicinity
							  of the peak. */


      /* Address of pointer to error message string.  Display
						   to obtain verbal description of error return. */

	char myChar[512];
	char** myCharPtr = (char**)(&myChar);

	CorrelationResult result;

	regoff(	ncols, nrows, first_image_buffer, second_image_buffer, 
			RowStrideA, RowStrideB, z, decimx, decimy, lims, job, 
			histclip, dump, &result.ColOffset, &result.RowOffset,
			&result.CorrCoeff, &result.AmbigScore, myCharPtr/*&error_msg*/);

	pCorrPair->SetCorrlelationResult(result);

	return(true);
}