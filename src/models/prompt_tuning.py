from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import yaml
import os

class PromptTuning:
    def __init__(
        self,
        model: MT5ForConditionalGeneration,
        tokenizer: MT5Tokenizer,
        prefix_length: int = 20,
        device: str = None
    ):
        """
        Initialize prompt tuning module.
        
        Args:
            model: Base MT5 model
            tokenizer: MT5 tokenizer
            prefix_length: Length of the prompt prefix
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        # Initialize prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prefix_length, self.model.config.hidden_size)
        ).to(self.device)
        
    def _load_prompt_templates(self) -> Dict:
        """Load prompt templates from yaml file."""
        template_path = os.path.join('config', 'prompt_templates.yaml')
        with open(template_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get_prompt_template(self, style: str, lang: str) -> str:
        """
        Get the prompt template for a specific style and language.
        
        Args:
            style: Target style (formal, simple, etc.)
            lang: Language code
            
        Returns:
            str: Prompt template
        """
        return self.prompt_templates.get(style, {}).get(lang, "")
        
    def prepare_input(
        self,
        text: str,
        style: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input with prompt for the model.
        
        Args:
            text: Input text
            style: Target style
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Dict containing model inputs
        """
        # Get prompt template
        template = self.get_prompt_template(style, target_lang)
        prompt = template.format(text=text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Add prompt embeddings
        input_embeddings = self.model.get_input_embeddings()(inputs["input_ids"])
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
            input_embeddings.size(0), -1, -1
        )
        
        # Concatenate prompt embeddings with input embeddings
        combined_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        
        return {
            "inputs_embeds": combined_embeddings,
            "attention_mask": torch.ones_like(combined_embeddings[:, :, 0])
        }
        
    def generate(
        self,
        text: str,
        style: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text with prompt tuning.
        
        Args:
            text: Input text
            style: Target style
            source_lang: Source language
            target_lang: Target language
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
        """
        # Prepare input with prompt
        model_inputs = self.prepare_input(text, style, source_lang, target_lang)
        
        # Generate
        outputs = self.model.generate(
            **model_inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=True
        )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def train(
        self,
        train_data: List[Dict[str, str]],
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 8
    ):
        """
        Train the prompt embeddings.
        
        Args:
            train_data: List of training examples
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
        """
        optimizer = torch.optim.AdamW([self.prompt_embeddings], lr=learning_rate)
        
        for epoch in range(num_epochs):
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                # Prepare batch inputs
                batch_inputs = [
                    self.prepare_input(
                        example["text"],
                        example["style"],
                        example["source_lang"],
                        example["target_lang"]
                    )
                    for example in batch
                ]
                
                # Forward pass
                outputs = self.model(
                    **batch_inputs[0],  # Use first example's inputs for now
                    labels=self.tokenizer(
                        [example["target"] for example in batch],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)["input_ids"]
                )
                
                # Backward pass
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
    def save_prompts(self, path: str):
        """Save prompt embeddings."""
        torch.save(self.prompt_embeddings.state_dict(), path)
        
    def load_prompts(self, path: str):
        """Load prompt embeddings."""
        self.prompt_embeddings.load_state_dict(torch.load(path)) 