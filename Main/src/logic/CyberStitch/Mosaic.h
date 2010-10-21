// CyberStitch.h

#pragma once

namespace CyberStitch 
{
	class Mosaic
	{
		public:
			Mosaic(int rows, int columns, int layers);
			bool AddImage(int row, int column, int layer);

		private:
			Mosaic(){};
	};
}
