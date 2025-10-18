"""
Generic text extractor for various document formats
Supports EPUB and PDF files with unified interface
"""
import os
from pathlib import Path
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

# Import existing EPUB extractor
from epub_extractor import EPUBExtractor


class BaseTextExtractor(ABC):
    """Base class for text extractors"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    @abstractmethod
    def extract_text(self) -> str:
        """Extract text content from the file"""
        pass
    
    @abstractmethod 
    def get_metadata(self) -> Dict[str, str]:
        """Extract metadata from the file"""
        pass


class PDFTextExtractor(BaseTextExtractor):
    """Text extractor for PDF files"""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
        except ImportError:
            try:
                import pypdf
                self.PyPDF2 = pypdf
            except ImportError:
                raise ImportError(
                    "PDF extraction requires PyPDF2 or pypdf. "
                    "Install with: pip install PyPDF2 or pip install pypdf"
                )
    
    def extract_text(self) -> str:
        """Extract text from PDF file"""
        text_content = []
        
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = self.PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        print(f"Warning: Failed to extract text from page {page_num + 1}: {e}")
                        continue
        
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")
        
        return "\n\n".join(text_content)
    
    def get_metadata(self) -> Dict[str, str]:
        """Extract metadata from PDF file"""
        metadata = {
            'title': 'Unknown',
            'author': 'Unknown', 
            'language': 'Unknown',
            'format': 'PDF'
        }
        
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = self.PyPDF2.PdfReader(file)
                
                if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    
                    # Map PDF metadata fields to our standard format
                    if '/Title' in pdf_meta and pdf_meta['/Title']:
                        metadata['title'] = str(pdf_meta['/Title'])
                    
                    if '/Author' in pdf_meta and pdf_meta['/Author']:
                        metadata['author'] = str(pdf_meta['/Author'])
                    
                    if '/Creator' in pdf_meta and pdf_meta['/Creator']:
                        if metadata['author'] == 'Unknown':
                            metadata['author'] = str(pdf_meta['/Creator'])
                    
                    if '/Subject' in pdf_meta and pdf_meta['/Subject']:
                        metadata['subject'] = str(pdf_meta['/Subject'])
                    
                    if '/Producer' in pdf_meta and pdf_meta['/Producer']:
                        metadata['producer'] = str(pdf_meta['/Producer'])
                
                # Add page count
                metadata['pages'] = str(len(pdf_reader.pages))
        
        except Exception as e:
            print(f"Warning: Failed to extract PDF metadata: {e}")
        
        # Use filename as fallback title
        if metadata['title'] == 'Unknown':
            metadata['title'] = Path(self.file_path).stem
        
        return metadata


class EPUBTextExtractorWrapper(BaseTextExtractor):
    """Wrapper for the existing EPUB extractor to match the new interface"""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.extractor = EPUBExtractor(file_path)
    
    def extract_text(self) -> str:
        """Extract text using existing EPUB extractor"""
        return self.extractor.extract_text()
    
    def get_metadata(self) -> Dict[str, str]:
        """Get metadata using existing EPUB extractor"""
        metadata = self.extractor.get_metadata()
        metadata['format'] = 'EPUB'
        return metadata


class TextExtractor:
    """Factory class for creating appropriate text extractors"""
    
    @staticmethod
    def create_extractor(file_path: str) -> BaseTextExtractor:
        """
        Create appropriate text extractor based on file extension
        
        Args:
            file_path (str): Path to the source file
            
        Returns:
            BaseTextExtractor: Appropriate extractor instance
            
        Raises:
            ValueError: If file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.epub':
            return EPUBTextExtractorWrapper(file_path)
        elif file_ext == '.pdf':
            return PDFTextExtractor(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: .epub, .pdf"
            )
    
    @staticmethod
    def get_supported_extensions() -> list:
        """Get list of supported file extensions"""
        return ['.epub', '.pdf']
    
    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        """Check if file format is supported"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in TextExtractor.get_supported_extensions()
    
    @staticmethod
    def create(file_path: str) -> BaseTextExtractor:
        """Alias for create_extractor for backward compatibility"""
        return TextExtractor.create_extractor(file_path)


def main():
    """Command line interface for testing text extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from EPUB or PDF files")
    parser.add_argument("file", help="Path to EPUB or PDF file")
    parser.add_argument("--metadata-only", action="store_true", help="Show only metadata")
    
    args = parser.parse_args()
    
    try:
        extractor = TextExtractor.create_extractor(args.file)
        
        print("Extracting metadata...")
        metadata = extractor.get_metadata()
        
        print("\n" + "="*50)
        print("METADATA")
        print("="*50)
        for key, value in metadata.items():
            print(f"{key.title()}: {value}")
        
        if not args.metadata_only:
            print("\nExtracting text...")
            text = extractor.extract_text()
            
            print("\n" + "="*50)
            print("TEXT CONTENT")
            print("="*50)
            print(f"Characters: {len(text)}")
            print(f"Words (approx): {len(text.split())}")
            print("\nFirst 500 characters:")
            print(text[:500])
            if len(text) > 500:
                print("...")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())