from typing import Dict, List, Optional
import torch
from transformers import T5Tokenizer
from bert_score import score
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TextEvaluator:
    def __init__(self, model_name: str = "vennify/t5-base-grammar-correction"):
        """
        Initialize text evaluator.
        
        Args:
            model_name: Name of the model for BERTScore
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.smooth = SmoothingFunction().method1
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def compute_bleu(
        self,
        reference: str,
        candidate: str
    ) -> float:
        """
        Compute BLEU score.
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            float: BLEU score
        """
        # Tokenize
        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())
        
        # Compute BLEU
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smooth)
        
    def compute_rouge(
        self,
        reference: str,
        candidate: str
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            Dict of ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
        
    def compute_bert_score(
        self,
        references: List[str],
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            references: List of reference texts
            candidates: List of generated texts
            
        Returns:
            Dict of BERTScore metrics
        """
        P, R, F1 = score(candidates, references, lang='en', verbose=True)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
        
    def compute_style_consistency(self, text: str, target_style: str) -> float:
        """
        Compute style consistency score.
        
        Args:
            text: Input text
            target_style: Target style
            
        Returns:
            float: Style consistency score
        """
        # For now, return a placeholder score
        # TODO: Implement proper style consistency scoring
        return 0.0
        
    def evaluate(
        self,
        reference: str,
        candidate: str,
        target_style: str = None
    ) -> Dict[str, float]:
        """
        Evaluate text transformation quality.
        
        Args:
            reference: Original text
            candidate: Transformed text
            target_style: Target style for consistency check
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Check for trivial output
        if candidate.strip() in ["<extra_id_0>", "<extra_id_0>.", ""]:
            logger.warning("Skipping evaluation for trivial output")
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "style_consistency": 0.0
            }
            
        metrics = {}
        
        try:
            # Compute BLEU score
            metrics['bleu'] = self.compute_bleu(reference, candidate)
            
            # Compute ROUGE scores
            rouge_scores = self.compute_rouge(reference, candidate)
            metrics.update(rouge_scores)
            
            # Compute BERTScore
            bert_scores = self.compute_bert_score(reference, candidate)
            metrics.update(bert_scores)
            
            # Compute style consistency if target style provided
            if target_style:
                metrics['style_consistency'] = self.compute_style_consistency(
                    candidate,
                    target_style
                )
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            # Return zero scores for failed metrics
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'precision', 'recall', 'f1', 'style_consistency']:
                if metric not in metrics:
                    metrics[metric] = 0.0
                    
        return metrics
        
    def evaluate_batch(
        self,
        references: List[str],
        candidates: List[str],
        target_styles: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for a batch of examples.
        
        Args:
            references: List of reference texts
            candidates: List of generated texts
            target_styles: List of target styles (optional)
            
        Returns:
            Dict of average metrics
        """
        all_metrics = []
        
        for i in range(len(references)):
            metrics = self.evaluate(
                references[i],
                candidates[i],
                target_styles[i] if target_styles else None
            )
            all_metrics.append(metrics)
            
        # Compute averages
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            
        return avg_metrics 