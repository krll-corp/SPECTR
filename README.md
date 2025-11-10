# SPECTR

A deep learning model for predicting chemical formulas from mass spectrometry data using a CNN-Transformer architecture.

## Overview

SPECTR is a machine learning system that accepts mass spectral data (m/z peaks and intensities) and predicts the corresponding chemical formula. The model uses a convolutional neural network encoder to process spectral peaks and a transformer decoder to generate chemical formulas token by token.

## Key Features

- CNN-based encoder for processing mass spectrometry peaks
- Transformer decoder for sequential chemical formula generation
- Support for all chemical elements from H to Og
- Handles up to 300 peaks per spectrum
- Multiple decoding strategies: greedy, beam search, top-k, and top-p sampling
- Training with gradient accumulation and learning rate scheduling
- Integration with Weights & Biases for experiment tracking
- Checkpoint saving and resumption support

## Architecture

The model consists of two main components:

### Encoder
- 4-layer CNN with batch normalization and max pooling
- Processes m/z and intensity pairs
- Global adaptive pooling for consistent output size
- Output: encoded representation of spectral data

### Decoder
- Transformer decoder with multi-head attention
- Token embedding with positional encoding
- Causal masking for autoregressive generation
- Vocabulary: special tokens, chemical elements (H-Og), and digits (0-9)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- pandas
- scikit-learn
- wandb (for experiment tracking)
- tqdm (optional, for progress bars)

Additional dependencies for data preparation:
- requests
- beautifulsoup4

## Installation

1. Clone the repository:
```bash
git clone https://github.com/krll-corp/SPECTR.git
cd SPECTR
```

2. Install dependencies:
```bash
pip install torch numpy pandas scikit-learn wandb tqdm requests beautifulsoup4
```

## Data Format

The model expects data in JSONL format with the following structure:

```json
{"formula": "C6H12O6", "peaks": [{"m/z": 180.063, "intensity": 999}, {"m/z": 145.050, "intensity": 450}]}
```

Each line contains:
- `formula`: Chemical formula as a string (e.g., "C6H12O6")
- `peaks`: List of peak objects with `m/z` (mass-to-charge ratio) and `intensity` values

## Usage

### Training

Train the model using the `train_conv.py` script:

```bash
python train_conv.py
```

Training options:
- `--resume`: Resume from previous run if available
- `--checkpoint`: Path to checkpoint file (default: `checkpoint_last.pt`)
- `--save-every`: Save checkpoint every N steps (default: 500)

The script expects a data file at `../mona_massbank_dataset.jsonl`. Modify the `data_file` variable in the script to point to your dataset.

Training hyperparameters (configurable in the script):
- Model dimension: 256
- Attention heads: 8
- Encoder/decoder layers: 8
- Batch size: 64
- Learning rate: 1e-4
- Max training steps: 10,000
- Gradient accumulation: 16 steps
- Warmup steps: 3% of max steps

### Evaluation

Evaluate a trained model using the `eval_conv3.py` script:

```bash
python eval_conv3.py --checkpoint <path_to_checkpoint> --data <path_to_data> --device cuda
```

Options:
- `--checkpoint`: Path to trained model checkpoint (default: `checkpoint_best.pt`)
- `--data`: Path to evaluation data file (default: `massbank_dataset.jsonl`)
- `--device`: Device to use (choices: `cpu`, `cuda`, `mps`)
- `--strategy`: Decoding strategy (choices: `greedy`, `beam`, `top_k`, `top_p`; default: `greedy`)
- `--beam-width`: Beam width for beam search (default: 3)
- `--top-k`: Top-k value for top-k sampling (default: 5)
- `--top-p`: Top-p value for nucleus sampling (default: 0.9)
- `--limit`: Limit number of samples to evaluate (optional)

### Data Preparation

The `data/` directory contains utilities for data collection and preparation:

- `crawler.py`: Extract mass spectrometry data from MassBank web records
- `filtering.py`: Process and filter MoNA and MassBank datasets into training format
- `analyze_massbank.py`: Analyze MassBank dataset statistics

Example usage for data crawling:
```bash
cd data
python crawler.py
```

## Model Specifications

Default configuration:
- Input: Up to 300 mass spectrometry peaks (m/z, intensity pairs)
- Output: Chemical formula up to 50 tokens
- Vocabulary size: 135 tokens (4 special + 118 elements + 10 digits + 3 reserved)
- Model parameters: ~12M (base configuration)

Supported chemical elements: All elements from H (Hydrogen) to Og (Oganesson)

Special tokens:
- `<PAD>`: Padding token
- `<SOS>`: Start of sequence
- `<EOS>`: End of sequence
- `<UNK>`: Unknown token

## Training Features

- Automatic mixed precision training support
- TF32 precision for Ampere GPUs
- Learning rate warmup and cosine decay
- Gradient accumulation for effective larger batch sizes
- Token-level accuracy metric (excluding padding)
- Validation every 1000 steps
- Automatic checkpoint saving
- Resume capability from saved checkpoints
- Weights & Biases integration for tracking

## Performance

The model is evaluated using:
- Cross-entropy loss (ignoring padding tokens)
- Token-level accuracy
- Perplexity
- Exact match accuracy (during evaluation)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Kyryll Kochkin

## Citation

If you use SPECTR in your research, please cite this repository:

```
@software{spectr2025,
  author = {Kochkin, Kyryll},
  title = {SPECTR: Mass Spectrometry to Chemical Formula Prediction},
  year = {2025},
  url = {https://github.com/krll-corp/SPECTR}
}
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments

This project uses data from:
- MassBank: A public repository of mass spectra
- MoNA (MassBank of North America): Mass spectral database
