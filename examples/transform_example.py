import sys
import os
import logging

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import StyleTransformerCLI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize transformer
    transformer = StyleTransformerCLI()
    
    # Define few-shot examples
    few_shot_examples = [
        ("yo wassup fam", "Hello, how are you doing?"),
        ("bruh this is lit", "This is excellent."),
        ("gonna bounce", "I must leave now."),
        ("can't make it, got stuff to do", "I apologize, but I am unable to attend as I have prior commitments."),
        ("wanna grab food?", "Would you like to join me for a meal?")
    ]
    
    # Example 1: Informal to Formal
    print("\nExample 1: Informal to Formal")
    input_text = "yo wassup fam"
    output = transformer.transform(
        text=input_text,
        few_shot_examples=few_shot_examples
    )
    print(f"Input: {input_text}")
    print(f"Output: {output}")
    
    # Example 2: Informal to Formal
    print("\nExample 2: Informal to Formal")
    input_text = "bruh this is lit"
    output = transformer.transform(
        text=input_text,
        few_shot_examples=few_shot_examples
    )
    print(f"Input: {input_text}")
    print(f"Output: {output}")
    
    # Example 3: Informal to Formal
    print("\nExample 3: Informal to Formal")
    input_text = "gonna bounce, see ya!"
    output = transformer.transform(
        text=input_text,
        few_shot_examples=few_shot_examples
    )
    print(f"Input: {input_text}")
    print(f"Output: {output}")
    
    # Example 4: Informal to Formal
    print("\nExample 4: Informal to Formal")
    input_text = "can't make it, got stuff to do"
    output = transformer.transform(
        text=input_text,
        few_shot_examples=few_shot_examples
    )
    print(f"Input: {input_text}")
    print(f"Output: {output}")
    
    # Example 5: With Evaluation
    print("\nExample 5: With Evaluation")
    input_text = "wanna grab food?"
    output = transformer.transform(
        text=input_text,
        few_shot_examples=few_shot_examples
    )
    print(f"Input: {input_text}")
    print(f"Output: {output}")
    
    # Evaluate transformation
    metrics = transformer.evaluate_transformation(
        original=input_text,
        transformed=output
    )
    print("\nEvaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
if __name__ == "__main__":
    main() 