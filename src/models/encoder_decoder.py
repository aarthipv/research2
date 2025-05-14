from typing import Dict, List, Optional, Tuple
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import logging
import re
from langdetect import detect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualTransformer:
    def __init__(
        self,
        model_name: str = "t5-base",
        device: str = None,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 0.7
    ):
        """
        Initialize transformer model for style transfer.
        
        Args:
            model_name: Name of the model to load
            device: Device to run on
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer from {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set generation config
        self.generation_config = {
            "max_length": max_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "length_penalty": 1.0,
            "repetition_penalty": 1.2
        }
        
    def _prepare_input(self, text: str, few_shot_examples: Optional[List[Tuple[str, str]]] = None) -> str:
        """Prepare input text with few-shot examples if provided."""
        if few_shot_examples:
            # Detect language
            lang = detect(text)
            prompt = f"Translate informal {lang} text to formal {lang}:\n"
            
            # Add few-shot examples
            for informal, formal in few_shot_examples:
                prompt += f"Informal: {informal}\nFormal: {formal}\n"
            
            # Add target input
            prompt += f"Informal: {text}\nFormal:"
        else:
            prompt = text
            
        return prompt
        
    def generate(
        self,
        text: str,
        few_shot_examples: Optional[List[Tuple[str, str]]] = None,
        style: str = None,
        source_lang: str = None,
        target_lang: str = None
    ) -> str:
        """
        Generate transformed text.
        
        Args:
            text: Input text
            few_shot_examples: List of (input, output) pairs for few-shot learning
            style: Target style (ignored)
            source_lang: Source language (ignored)
            target_lang: Target language (ignored)
            
        Returns:
            str: Generated text
        """
        # Prepare input
        prompt = self._prepare_input(text, few_shot_examples)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Log tokenized input for debugging
        logger.info(f"Tokenized input: {self.tokenizer.decode(inputs['input_ids'][0])}")
        
        # Generate output
        outputs = self.model.generate(
            **inputs,
            **self.generation_config
        )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process: extract only the formalized sentence after the last 'Formal:'
        if 'Formal:' in generated_text:
            # Take the last occurrence of 'Formal:' and everything after
            generated_text = generated_text.split('Formal:')[-1].strip()
            # Stop at the next 'Informal:' if present
            if 'Informal:' in generated_text:
                generated_text = generated_text.split('Informal:')[0].strip()
            # Stop at the first newline if present
            if '\n' in generated_text:
                generated_text = generated_text.split('\n')[0].strip()
            # Stop at the first sentence-ending punctuation if present
            match = re.match(r'(.+?[.!?])\s', generated_text)
            if match:
                generated_text = match.group(1).strip()

        # Check if output is trivial
        if generated_text.strip() in ["<extra_id_0>", "<extra_id_0>.", ""]:
            logger.warning("Model generated trivial output. Returning original text.")
            return text
        
        return generated_text
        
    def save(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load(self, path: str):
        """Load model and tokenizer."""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
    def encode(
        self,
        text: str,
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            torch.Tensor: Text embeddings
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
        
    def decode(
        self,
        embeddings: torch.Tensor,
        max_length: int = 512
    ) -> str:
        """
        Decode embeddings back to text.
        
        Args:
            embeddings: Input embeddings
            max_length: Maximum output length
            
        Returns:
            str: Decoded text
        """
        # Expand embeddings to match expected input shape
        expanded_embeddings = embeddings.unsqueeze(0)
        
        # Generate
        outputs = self.model.generate(
            encoder_outputs=(expanded_embeddings,),
            max_length=max_length,
            num_beams=5,
            temperature=0.7,
            do_sample=True
        )
        
        # Decode output
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_text
        
    def fine_tune(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 8
    ):
        """
        Fine-tune the model on custom data.
        
        Args:
            train_data (List[Dict[str, str]]): Training data
            eval_data (Optional[List[Dict[str, str]]]): Evaluation data
            learning_rate (float): Learning rate
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        # Implementation for fine-tuning
        pass 