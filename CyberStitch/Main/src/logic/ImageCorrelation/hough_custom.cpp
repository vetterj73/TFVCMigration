/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*////////////////////////////////////////////
// This modified version is to support angle range of search
// It doesn't reduce the memeory usage for a angle range smaller than PI
//*/

#include "hough_custom.h"

#define halfPi ((float)(CV_PI*0.5))
#define Pi     ((float)CV_PI)
#define a0  0 /*-4.172325e-7f*/   /*(-(float)0x7)/((float)0x1000000); */
#define a1 1.000025f        /*((float)0x1922253)/((float)0x1000000)*2/Pi; */
#define a2 -2.652905e-4f    /*(-(float)0x2ae6)/((float)0x1000000)*4/(Pi*Pi); */
#define a3 -0.165624f       /*(-(float)0xa45511)/((float)0x1000000)*8/(Pi*Pi*Pi); */
#define a4 -1.964532e-3f    /*(-(float)0x30fd3)/((float)0x1000000)*16/(Pi*Pi*Pi*Pi); */
#define a5 1.02575e-2f      /*((float)0x191cac)/((float)0x1000000)*32/(Pi*Pi*Pi*Pi*Pi); */
#define a6 -9.580378e-4f    /*(-(float)0x3af27)/((float)0x1000000)*64/(Pi*Pi*Pi*Pi*Pi*Pi); */

#define _sin(x) ((((((a6*(x) + a5)*(x) + a4)*(x) + a3)*(x) + a2)*(x) + a1)*(x) + a0)
#define _cos(x) _sin(halfPi - (x))


/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

static void
icvHoughLinesProbabalistic_custom( 
	CvMat* image, float rho, 
	float fStartAngle, float fEndAngle, float theta, 
	int threshold, int lineLength, int lineGap,
    CvSeq *lines, int linesMax )
{
    cv::Mat accum, mask;
    cv::vector<float> trigtab;
    cv::MemStorage storage(cvCreateMemStorage(0));

    CvSeq* seq;
    CvSeqWriter writer;
    int width, height;
    int numangle, numrho;
    float ang;
    int r, n, count;
    CvPoint pt;
    float irho = 1 / rho;
    CvRNG rng = cvRNG(-1);
    const float* ttab;
    uchar* mdata0;

    CV_Assert( CV_IS_MAT(image) && CV_MAT_TYPE(image->type) == CV_8UC1 );

    width = image->cols;
    height = image->rows;

    numangle = cvRound(CV_PI / theta);
    numrho = cvRound(((width + height) * 2 + 1) / rho);

	// custom
	int iStartIndex = (int)(fStartAngle/theta-0.5);
	int iEndIndex = (int)(fEndAngle/theta+0.5);

    accum.create( numangle, numrho, CV_32SC1 );
    mask.create( height, width, CV_8UC1 );
    trigtab.resize(numangle*2);
    accum = cv::Scalar(0);

    //for( ang = 0, n = 0; n < numangle; ang += theta, n++ )
	for( ang = iStartIndex*theta, n = iStartIndex; n < iEndIndex+1; ang += theta, n++ )
    {
        trigtab[n*2] = (float)(cos(ang) * irho);
        trigtab[n*2+1] = (float)(sin(ang) * irho);
    }
    ttab = &trigtab[0];
    mdata0 = mask.data;

    cvStartWriteSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage, &writer );

    // stage 1. collect non-zero image points
    for( pt.y = 0, count = 0; pt.y < height; pt.y++ )
    {
        const uchar* data = image->data.ptr + pt.y*image->step;
        uchar* mdata = mdata0 + pt.y*width;
        for( pt.x = 0; pt.x < width; pt.x++ )
        {
            if( data[pt.x] )
            {
                mdata[pt.x] = (uchar)1;
                CV_WRITE_SEQ_ELEM( pt, writer );
            }
            else
                mdata[pt.x] = 0;
        }
    }

    seq = cvEndWriteSeq( &writer );
    count = seq->total;

    // stage 2. process all the points in random order
    for( ; count > 0; count-- )
    {
        // choose random point out of the remaining ones
        int idx = cvRandInt(&rng) % count;
        int max_val = threshold-1, max_n = 0;
        CvPoint* pt = (CvPoint*)cvGetSeqElem( seq, idx );
        CvPoint line_end[2] = {{0,0}, {0,0}};
        float a, b;
        int* adata = (int*)accum.data;
        int i, j, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        i = pt->y;
        j = pt->x;

        // "remove" it by overriding it with the last element
        *pt = *(CvPoint*)cvGetSeqElem( seq, count-1 );

        // check if it has been excluded already (i.e. belongs to some other line)
        if( !mdata0[i*width + j] )
            continue;

        // update accumulator, find the most probable line
        //for( n = 0; n < numangle; n++, adata += numrho )
		adata += numrho*iStartIndex;		// custom
		for( n = iStartIndex; n < iEndIndex+1; n++, adata += numrho ) // custom
        {
            r = cvRound( j * ttab[n*2] + i * ttab[n*2+1] );
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if( max_val < val )
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if( max_val < threshold )
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n*2+1];
        b = ttab[max_n*2];
        x0 = j;
        y0 = i;
        if( fabs(a) > fabs(b) )
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound( b*(1 << shift)/fabs(a) );
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound( a*(1 << shift)/fabs(b) );
            x0 = (x0 << shift) + (1 << (shift-1));
        }

        for( k = 0; k < 2; k++ )
        {
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if( ++gap > lineGap )
                    break;
            }
        }

        good_line = abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    abs(line_end[1].y - line_end[0].y) >= lineLength;

        for( k = 0; k < 2; k++ )
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    if( good_line )
                    {
                        adata = (int*)accum.data;
                        for( n = 0; n < numangle; n++, adata += numrho )
                        {
                            r = cvRound( j1 * ttab[n*2] + i1 * ttab[n*2+1] );
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if( i1 == line_end[k].y && j1 == line_end[k].x )
                    break;
            }
        }

        if( good_line )
        {
            CvRect lr = { line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y };
            cvSeqPush( lines, &lr );
            if( lines->total >= linesMax )
                return;
        }
    }
}

/* Wrapper function for standard hough transform */
CV_IMPL CvSeq* cvHoughLines2_P_Custom( 
	CvArr* src_image, void* lineStorage, double rho, 
	double dStartAngle, double dEndAngle, double theta, 
	int threshold, int iMinLength, int iMaxGap)
{
    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image;
    CvMat* mat = 0;
    CvSeq* lines = 0;
    CvSeq lines_header;
    CvSeqBlock lines_block;
    int lineType, elemSize;
    int linesMax = INT_MAX;

    img = cvGetMat( img, &stub );

    if( !CV_IS_MASK_ARR(img))
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if( !lineStorage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if( rho <= 0 || theta <= 0 || threshold <= 0 )
        CV_Error( CV_StsOutOfRange, "rho, theta and threshold must be positive" );

    lineType = CV_32SC4;
    elemSize = sizeof(int)*4;

    if( CV_IS_STORAGE( lineStorage ))
    {
        lines = cvCreateSeq( lineType, sizeof(CvSeq), elemSize, (CvMemStorage*)lineStorage );
    }
    else if( CV_IS_MAT( lineStorage ))
    {
        mat = (CvMat*)lineStorage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        if( CV_MAT_TYPE( mat->type ) != lineType )
            CV_Error( CV_StsBadArg,
            "The destination matrix data type is inappropriate, see the manual" );

        lines = cvMakeSeqHeaderForArray( lineType, sizeof(CvSeq), elemSize, mat->data.ptr,
                                         mat->rows + mat->cols - 1, &lines_header, &lines_block );
        linesMax = lines->total;
        cvClearSeq( lines );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );

    icvHoughLinesProbabalistic_custom( 
		img, (float)rho, 
		(float)dStartAngle, (float)dEndAngle, (float)theta,
        threshold, iMinLength, iMaxGap, lines, linesMax );

    if( mat )
    {
        if( mat->cols > mat->rows )
            mat->cols = lines->total;
        else
            mat->rows = lines->total;
    }
    else
        result = lines;

    return result;
}    

/* End of file. */
