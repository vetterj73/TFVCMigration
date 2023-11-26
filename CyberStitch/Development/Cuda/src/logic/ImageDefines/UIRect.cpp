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