import sys
import os
import logging

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import StyleTransformerCLI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_separator():
    print("-" * 40)

def print_example_header(example_num, language):
    print("\n" + "=" * 80)
    print(f"Example {example_num}: Informal to Formal ({language})")
    print("-" * 80)

def build_prompt(language, examples, input_text):
    prompt = (
        f"You are a formal language assistant. Convert informal {language.lower()} to formal {language.lower()}.\n\n"
        f"Examples:\n"
    )
    for informal, formal in examples:
        prompt += f"Informal: {informal}\nFormal: {formal}\n\n"
    prompt += f"Now convert:\nInformal: {input_text}\nFormal:"
    return prompt

def main():
    # ✅ Use flan-t5 model
    transformer = StyleTransformerCLI(model_name="google/flan-t5-base")

    # English examples
    few_shot_examples_en = [
        ("yo wassup fam", "Hello, how are you doing?"),
        ("bruh this is lit", "This is excellent."),
        ("gonna bounce", "I must leave now."),
        ("can't make it, got stuff to do", "I apologize, but I am unable to attend as I have prior commitments."),
        ("wanna grab food?", "Would you like to join me for a meal?")
    ]

    test_inputs_en = [
        "yo wassup fam",
        "bruh this is lit",
        "gonna bounce, see ya!",
        "can't make it, got stuff to do",
        "wanna grab food?"
    ]

    print_example_header(1, "English")
    for input_text in test_inputs_en:
        prompt = build_prompt("English", few_shot_examples_en, input_text)
        output = transformer.transform(text=prompt)
        print(f"Input:   {input_text}")
        print(f"Output:  {output}")
        print_separator()

    # French examples
    few_shot_examples_fr = [
        ("ça va mec ?", "Comment allez-vous ?"),
        ("t'es où ?", "Où êtes-vous ?"),
        ("j'vais y aller", "Je vais y aller."),
        ("t'as pas le temps ?", "N'avez-vous pas le temps ?"),
        ("on se voit plus tard ?", "Nous verrons-nous plus tard ?"),
    ]

    test_inputs_fr = [
        "salut, tu fais quoi ?",
        "t'as faim ?",
        "t'as compris ?",
        "on y va ?"
    ]

    print_example_header(2, "French")
    for input_text in test_inputs_fr:
        prompt = build_prompt("French", few_shot_examples_fr, input_text)
        output = transformer.transform(text=prompt)
        print(f"Input:   {input_text}")
        print(f"Output:  {output}")
        print_separator()

if __name__ == "__main__":
    main()