#include "UIRect.h"


UIRect::UIRect() 
{
	FirstColumn	= 0;
	LastColumn	= 0;
	FirstRow	= 0;
	LastRow		= 0;
}

UIRect::UIRect(
	unsigned int firstCol,
	unsigned int firstRow,	
	unsigned int lastCol,
	unsigned int lastRow ) 
{
	FirstColumn	= firstCol;
	LastColumn	= lastCol;
	FirstRow	= firstRow;
	LastRow		= lastRow;
}

UIRect::UIRect(const UIRect& b)
{
	*this = b;
}

void UIRect::operator=(const UIRect& b)
{
	FirstColumn	= b.FirstColumn;
	LastColumn	= b.LastColumn;
	FirstRow	= b.FirstRow;
	LastRow		= b.LastRow;
}

bool UIRect::IsValid() const
{
	return FirstColumn<LastColumn && 
		FirstRow<LastRow;
}

unsigned int UIRect::Rows() const
{
	if( !IsValid() )
		return 0;

	return LastRow-FirstRow+1; 
}

unsigned int UIRect::Columns() const
{
	if( !IsValid() )
		return 0;

	return LastColumn-FirstColumn+1; 
}

unsigned int UIRect::Size() const
{
	return(Rows()*Columns());
}

double UIRect::ColumnCenter() const
{
	if( !IsValid() )
		return 0;

	return (LastColumn+FirstColumn)/2.0;
}

double UIRect::RowCenter() const
{
	if( !IsValid() )
		return 0;

	return (LastRow+FirstRow)/2.0; 
}


DRect::DRect()
{
	Reset();
}

void DRect::Reset()
{
	xMin = 0;
	xMax = 0;
	yMin = 0;
	yMax = 0;
}

double DRect::Width() const
{
	if(!IsValid())
		return 0;
	else
		return(xMax-xMin);
}

double DRect::Height() const
{
	if(!IsValid())
		return 0;
	else
		return(yMax-yMin);
}

double DRect::Area() const
{
	return(Width()*Height());
}

double DRect::CenX() const
{
	if(!IsValid())
		return(0);
	else
		return((xMin+xMax)/2.0);
}

double DRect::CenY() const
{
	if(!IsValid())
		return(0);
	else
		return((yMin+yMax)/2.0);
}

bool DRect::IsValid() const
{
	if(xMax>xMin && yMax>yMin)
		return(true);
	else
		return(false);
}

DRect DRect::OverlapRect(DRect inRect) const
{
	DRect overlapRect;

	if(!inRect.IsValid() || !IsValid())
		return(overlapRect);

	overlapRect.xMin = xMin > inRect.xMin ? xMin : inRect.xMin;
	overlapRect.xMax = xMax < inRect.xMax ? xMax : inRect.xMax;
	overlapRect.yMin = yMin > inRect.yMin ? yMin : inRect.yMin;
	overlapRect.yMax = yMax < inRect.yMax ? yMax : inRect.yMax;

	return(overlapRect);
}
