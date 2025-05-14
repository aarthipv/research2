# Multilingual Style-Aware Text Generation

A modular Python-based pipeline for style-aware text transformation across multiple languages, supporting simplification and formalization tasks.

## Features

- Multilingual text processing (English, Hindi, French, etc.)
- Style-aware text transformation (simplification, formalization)
- Prompt-tuning and prefix-tuning support
- Retrieval-Augmented Generation (RAG) for knowledge enrichment
- Comprehensive evaluation metrics
- CLI and API interfaces

## Project Structure

```
multilingual_style/
├── config/
│   ├── model_config.yaml
│   └── prompt_templates.yaml
├── src/
│   ├── preprocessing/
│   │   ├── language_detection.py
│   │   └── text_normalization.py
│   ├── models/
│   │   ├── encoder_decoder.py
│   │   ├── prompt_tuning.py
│   │   └── rag.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── human_eval.py
│   └── utils/
│       ├── data_utils.py
│       └── logging_utils.py
├── tests/
├── examples/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### CLI Interface

```bash
python -m multilingual_style.cli --input "Your text here" --style formal --lang en
```

### Python API

```python
from multilingual_style import StyleTransformer

transformer = StyleTransformer()
result = transformer.transform(
    text="Your text here",
    target_style="formal",
    source_lang="en",
    target_lang="fr"
)
```

## Configuration

Edit `config/model_config.yaml` to customize:
- Model parameters
- Prompt templates
- Evaluation metrics
- RAG settings

## License

MIT License 