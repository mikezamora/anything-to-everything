"""
Module for extracting text content from EPUB files
"""
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import html
import re


class EPUBExtractor:
    """Extract and clean text from EPUB files"""
    
    def __init__(self, epub_path):
        """
        Initialize the EPUB extractor
        
        Args:
            epub_path (str): Path to the EPUB file
        """
        self.epub_path = epub_path
        self.book = None
        
    def load(self):
        """Load the EPUB file"""
        try:
            self.book = epub.read_epub(self.epub_path)
            return True
        except Exception as e:
            print(f"Error loading EPUB file: {e}")
            return False
    
    def extract_text(self):
        """
        Extract all text content from the EPUB file
        
        Returns:
            str: Extracted text content
        """
        if not self.book:
            if not self.load():
                return ""
        
        text_content = []
        
        # Iterate through all items in the book
        for item in self.book.get_items():
            # Only process document items (chapters, etc.)
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                text = soup.get_text()
                
                # Clean up text
                text = self._clean_text(text)
                
                if text.strip():
                    text_content.append(text)
        
        return "\n\n".join(text_content)
    
    def _clean_text(self, text):
        """
        Clean extracted text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def get_metadata(self):
        """
        Extract metadata from the EPUB file
        
        Returns:
            dict: Metadata dictionary with fields for audiobook conversion
        """
        if not self.book:
            if not self.load():
                return {}
        
        metadata = {}
        
        # Extract common metadata
        try:
            title = self.book.get_metadata('DC', 'title')[0][0] if self.book.get_metadata('DC', 'title') else "Unknown"
            author = self.book.get_metadata('DC', 'creator')[0][0] if self.book.get_metadata('DC', 'creator') else "Unknown"
            language = self.book.get_metadata('DC', 'language')[0][0] if self.book.get_metadata('DC', 'language') else "Unknown"
            
            metadata['title'] = title
            metadata['author'] = author
            metadata['language'] = language
            
            # For M4B format - album field
            # Try to get series information first, otherwise use title
            series = None
            try:
                # Check for calibre series metadata
                series_meta = self.book.get_metadata('OPF', 'meta')
                for meta in series_meta:
                    if isinstance(meta, tuple) and len(meta) > 1:
                        attrs = meta[1]
                        if isinstance(attrs, dict) and attrs.get('name') == 'calibre:series':
                            series = attrs.get('content')
                            break
            except:
                pass
            
            # Set album: use series if available, otherwise use title
            if series:
                metadata['album'] = series
                metadata['series'] = series
            else:
                metadata['album'] = title
            
            # Try to extract publisher for additional metadata
            try:
                publisher = self.book.get_metadata('DC', 'publisher')
                if publisher:
                    metadata['publisher'] = publisher[0][0]
            except:
                pass
            
            # Try to extract publication date
            try:
                date = self.book.get_metadata('DC', 'date')
                if date:
                    metadata['date'] = date[0][0]
            except:
                pass
            
            # Try to extract description/summary
            try:
                description = self.book.get_metadata('DC', 'description')
                if description:
                    metadata['description'] = description[0][0]
            except:
                pass
                
        except Exception as e:
            print(f"Error extracting metadata: {e}")
        
        return metadata


if __name__ == "__main__":
    # Test the extractor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python epub_extractor.py <epub_file>")
        sys.exit(1)
    
    extractor = EPUBExtractor(sys.argv[1])
    metadata = extractor.get_metadata()
    print(f"Title: {metadata.get('title', 'Unknown')}")
    print(f"Author: {metadata.get('author', 'Unknown')}")
    print(f"Language: {metadata.get('language', 'Unknown')}")
    print("\n" + "="*50 + "\n")
    
    text = extractor.extract_text()
    print(f"Extracted {len(text)} characters")
    print("\nFirst 500 characters:")
    print(text[:500])
