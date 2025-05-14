import argparse
import yaml
import os
from typing import Optional, Dict, List, Tuple
from src.models.encoder_decoder import MultilingualTransformer
from src.models.prompt_tuning import PromptTuning
from src.models.rag import RAG
from src.preprocessing.language_detection import LanguageDetector
from src.preprocessing.text_normalization import TextNormalizer
from src.evaluation.metrics import TextEvaluator
import logging

logger = logging.getLogger(__name__)

class StyleTransformerCLI:
    def __init__(
        self,
        model_name: str = "t5-base",
        config_path: str = "config/default.yaml"
    ):
        """
        Initialize the style transformer CLI.
        
        Args:
            model_name: Name of the model to use
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        logger.info(f"Initializing components with model: {model_name}")
        
        # Initialize language detector
        self.language_detector = LanguageDetector()
        
        # Initialize text normalizer
        self.normalizer = TextNormalizer()
        
        # Initialize encoder-decoder model
        self.model = MultilingualTransformer(
            model_name=model_name,
            max_length=self.config.get('max_length', 512),
            num_beams=self.config.get('num_beams', 4),
            temperature=self.config.get('temperature', 0.7)
        )
        
        # Initialize prompt tuning if enabled
        if self.config.get('use_prompt_tuning', False):
            self.prompt_tuning = PromptTuning(
                model_name=model_name,
                num_virtual_tokens=self.config.get('num_virtual_tokens', 10)
            )
        else:
            self.prompt_tuning = None
            
        # Initialize RAG if enabled
        if self.config.get('use_rag', False):
            self.rag = RAG(
                model_name=model_name,
                index_path=self.config.get('rag_index_path')
            )
        else:
            self.rag = None
            
        # Initialize evaluator
        self.evaluator = TextEvaluator()
        
    def transform(
        self,
        text: str,
        few_shot_examples: Optional[List[Tuple[str, str]]] = None,
        style: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> str:
        """
        Transform text using few-shot examples.
        
        Args:
            text: Input text
            few_shot_examples: List of (input, output) pairs for few-shot learning
            style: Target style (ignored)
            source_lang: Source language (optional)
            target_lang: Target language (optional)
            
        Returns:
            str: Transformed text
        """
        # Detect language if not provided
        if not source_lang:
            source_lang = self.language_detector.detect(text)
            logger.info(f"Detected source language: {source_lang}")
            
        # Set target language to source if not provided
        if not target_lang:
            target_lang = source_lang
            
        # Normalize text
        text = self.normalizer.normalize(text)
        logger.info(f"Normalized text: {text}")
        
        # Apply RAG if enabled
        if self.rag:
            text = self.rag.enrich_generation(text, style)
            logger.info(f"Enriched text: {text}")
            
        # Generate transformed text
        transformed = self.model.generate(
            text=text,
            few_shot_examples=few_shot_examples,
            style=style,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        return transformed
        
    def evaluate_transformation(
        self,
        original: str,
        transformed: str,
        style: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate transformation quality.
        
        Args:
            original: Original text
            transformed: Transformed text
            style: Target style (ignored)
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        return self.evaluator.evaluate(
            reference=original,
            candidate=transformed,
            target_style=style
        )
        
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Text Style Transfer with Few-Shot Learning"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text to transform"
    )
    
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        help="Example pairs in format 'input:output'"
    )
    
    parser.add_argument(
        "--source-lang",
        type=str,
        help="Source language code (optional)"
    )
    
    parser.add_argument(
        "--target-lang",
        type=str,
        help="Target language code (optional)"
    )
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG for knowledge enrichment"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate transformation quality"
    )
    
    args = parser.parse_args()
    
    # Parse examples
    few_shot_examples = None
    if args.examples:
        few_shot_examples = []
        for example in args.examples:
            input_text, output_text = example.split(":")
            few_shot_examples.append((input_text.strip(), output_text.strip()))
    
    # Initialize CLI
    cli = StyleTransformerCLI()
    
    # Transform text
    output = cli.transform(
        text=args.input,
        few_shot_examples=few_shot_examples,
        source_lang=args.source_lang,
        target_lang=args.target_lang
    )
    
    # Print output
    print("\nTransformed text:")
    print(output)
    
    # Evaluate if requested
    if args.evaluate:
        metrics = cli.evaluate_transformation(
            original=args.input,
            transformed=output
        )
        print("\nEvaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
if __name__ == "__main__":
    main() 