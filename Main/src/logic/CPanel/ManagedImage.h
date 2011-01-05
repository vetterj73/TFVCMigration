#pragma once

/**
	This class was stolen from Doug Butler's SE500OEM Project.  This would be a good candidate for 
	sharing in a common project... evaluating the usefulness and where to put it...

**/
namespace Cyber
{	
	namespace MPanel
	{
		template<class T>
		public ref class ManagedImage
		{
			public:
				virtual ~ManagedImage() 
				{
					delete[] pImage_;
				}

				property System::IntPtr Buffer
				{
					System::IntPtr get() { return System::IntPtr((void*)pImage_); }
				}

				property unsigned int Columns
				{
					unsigned int get() { return columns_;}
				}

				property unsigned int Rows
				{
					unsigned int get() { return rows_; }
				}

				property unsigned int BytesPerPixel
				{
					unsigned int get() { return (unsigned int)sizeof(T); }
				}

				property unsigned int BufferSizeInBytes
				{
					unsigned int get() { return columns_*rows_*BytesPerPixel; }
				}

				property unsigned int BufferSizeInPixels
				{
					unsigned int get() { return columns_*rows_; }
				}
	
				property unsigned int RowStrideInBytes
				{
					unsigned int get() { return columns_*BytesPerPixel; }
				}

				property unsigned int RowStrideInPixels
				{
					unsigned int get() { return columns_; }
				}

			protected:
				ManagedImage(unsigned int columns, unsigned int rows)
				{
					columns_ = columns;
					rows_ = rows;		
				}
				T* pImage_;

				unsigned int columns_;
				unsigned int rows_;

				ManagedImage() : pImage_(NULL) {};
		};

		template ref class ManagedImage<unsigned char>;
		template ref class ManagedImage<unsigned short>;

		public ref class ManagedImage8 : public ManagedImage<unsigned char>
		{
			public:
				ManagedImage8(unsigned int columns, unsigned int rows):
				  ManagedImage(columns, rows)
				{
					pImage_ = new unsigned char[columns*rows];
					memset(pImage_, 0, columns*rows);
				}
			protected:
				ManagedImage8() {};
		};

		public ref class ManagedImage16 : public ManagedImage<unsigned short>
		{	
			public:
				ManagedImage16(unsigned int columns, unsigned int rows):
				  ManagedImage(columns, rows)
				{
					pImage_ = new unsigned short[columns*rows];
					memset(pImage_, 0, columns*rows*2);				
				}
			private:
				ManagedImage16() {};
		};
	}
}

