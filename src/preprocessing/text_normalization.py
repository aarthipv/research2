import re
import unicodedata
from typing import Optional
import sacremoses
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer

class TextNormalizer:
    def __init__(self):
        self.punct_normalizer = MosesPunctNormalizer()
        self.tokenizer = MosesTokenizer()
        self.detokenizer = MosesDetokenizer()
        
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to their canonical form.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        return unicodedata.normalize('NFKC', text)
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing punctuation.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize punctuation
        text = self.punct_normalizer.normalize(text)
        return text.strip()
        
    def normalize(self, text: str, lang: Optional[str] = None) -> str:
        """
        Apply all normalization steps to the text.
        
        Args:
            text (str): Input text
            lang (Optional[str]): Language code for language-specific normalization
            
        Returns:
            str: Normalized text
        """
        # Apply Unicode normalization
        text = self.normalize_unicode(text)
        
        # Apply language-specific normalization if language is specified
        if lang:
            # Set language for tokenizer and detokenizer
            self.tokenizer.lang = lang
            self.detokenizer.lang = lang
            # Tokenize and detokenize to normalize spacing
            tokens = self.tokenizer.tokenize(text)
            text = self.detokenizer.detokenize(tokens)
        
        # Clean text
        text = self.clean_text(text)
        
        return text
        
    def transliterate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Transliterate text between scripts if needed.
        Currently supports basic Latin to Devanagari and vice versa.
        
        Args:
            text (str): Input text
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            str: Transliterated text
        """
        # Add transliteration logic here
        # This is a placeholder for actual transliteration implementation
        return text 