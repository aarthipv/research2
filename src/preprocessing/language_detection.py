from typing import Tuple, Optional
import langdetect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

class LanguageDetector:
    def __init__(self):
        # Set seed for reproducibility
        DetectorFactory.seed = 0
        
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text.
        
        Args:
            text (str): Input text to detect language
            
        Returns:
            Tuple[str, float]: Language code and confidence score
        """
        try:
            # Get all possible languages with their probabilities
            detector = langdetect.detect_langs(text)
            # Get the most probable language
            most_probable = detector[0]
            return most_probable.lang, most_probable.prob
        except LangDetectException:
            # Default to English if detection fails
            return "en", 0.0
            
    def is_supported(self, lang_code: str) -> bool:
        """
        Check if the language is supported by the system.
        
        Args:
            lang_code (str): Language code to check
            
        Returns:
            bool: True if language is supported, False otherwise
        """
        supported_langs = ["en", "fr", "hi", "es", "de"]
        return lang_code in supported_langs
        
    def get_language_name(self, lang_code: str) -> str:
        """
        Get the full name of the language from its code.
        
        Args:
            lang_code (str): Language code
            
        Returns:
            str: Full language name
        """
        lang_names = {
            "en": "English",
            "fr": "French",
            "hi": "Hindi",
            "es": "Spanish",
            "de": "German"
        }
        return lang_names.get(lang_code, "Unknown") 