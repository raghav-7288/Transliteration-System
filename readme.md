---

# RNN BASED Seq2Seq MODEL

This repository contains a Python implementation of a sequence-to-sequence (Seq2Seq) model for sequence prediction tasks. The Seq2Seq model is implemented using PyTorch and includes different recurrent neural network (RNN) cell types such as LSTM, RNN, and GRU for both the encoder and decoder.


## Classes

### Encoder
- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  
- **Methods**:
  - `__init__()`: Initializes the encoder.
  - `forward()`: Performs forward pass through the encoder.

### Decoder
- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `output_size`: Size of the output vocabulary.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  
- **Methods**:
  - `__init__()`: Initializes the decoder.
  - `forward()`: Performs forward pass through the decoder.

### Seq2Seq
- **Data Members**:
  - `encoder`: Instance of the Encoder class.
  - `decoder`: Instance of the Decoder class.
  - `output_index_size`: Size of the target vocabulary.
  - `teacher_force_ratio`: Ratio of teacher forcing during training.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

- **Methods**:
  - `__init__()`: Initializes the Seq2Seq model.
  - `forward()`: Performs forward pass through the model.

## Training Functions

### train()
- **Arguments**:
  - `model`: The Seq2Seq model to be trained.
  - `num_epochs`: Number of training epochs.
  - `criterion`: Loss criterion for training.
  - `optimizer`: Optimizer for training.
  - `input_data`: Training input data batch.
  - `output_data`: Training target data batch.
  - `val_input_data`: Validation input data batch.
  - `val_output_data`: Validation target data batch.
  - `df_val`: DataFrame for validation data.
  - `input_index`: Mapping from characters to integers for the input vocabulary.
  - `output_index`: Mapping from characters to integers for the output vocabulary.
  - `output_index_reversed`: Reverse mapping from integers to characters for the output vocabulary.
  - `beam_width`: Beam width for beam search.
  - `length_penalty`: Length penalty for beam search.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
  - `max_len`: Maximum length of sequences.
  - `wandb_log`: Whether to log to Weights & Biases (1 for yes, 0 for no).

- **Returns**:
  - `model`: The trained Seq2Seq model.
  - `beam_val`: Validation accuracy using beam search.

### beam_search()
- **Arguments**:
  - `model`: The Seq2Seq model for inference.
  - `input_seq`: Input sequence for translation.
  - `max_length`: Maximum length of the input sequence.
  - `input_index`: Mapping from characters to integers for the input vocabulary.
  - `output_index`: Mapping from characters to integers for the output vocabulary.
  - `output_index_reversed`: Reverse mapping from integers to characters for the output vocabulary.
  - `beam_width`: Beam width for beam search.
  - `length_penalty`: Length penalty for beam search.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

- **Returns**:
  - `str`: The generated output sequence.

---

## Installation

To run the training script, ensure you have Python 3 installed along with the following dependencies:

- torch
- numpy
- pandas
- tqdm
- wandb
- argparse

You can install these dependencies using pip:

```bash
pip install torch numpy pandas tqdm wandb argparse
```

## Usage

To train the Seq2Seq model with different RNN cell types, use the `train.py` script with the following command-line arguments:

| Argument               | Description                                                              | Default Value                               |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------------------------- |
| -dp, --dataset_path    | Path to the data folder                                                  | '/kaggle/input/dl-ass3/aksharantar_sampled' |
| -lg, --language        | Language for which training is to be done                                | 'hin'                                       |
| -es, --embedding_size  | Embedding size                                                           | 256                                         |
| -hs, --hidden_size     | Hidden size                                                              | 512                                         |
| -nl, --num_layers      | Number of layers                                                         | 2                                           |
| -ct, --cell_type       | Cell type (RNN, LSTM, GRU)                                               | 'LSTM'                                      |
| -do, --dropout         | Dropout rate                                                             | 0.3                                         |
| -lr, --learning_rate   | Learning rate                                                            | 0.01                                        |
| -bs, --batch_size      | Batch size                                                               | 32                                          |
| -ep, --num_epochs      | Number of epochs                                                         | 10                                          |
| -op, --optimizer       | Optimizer (adam, sgd, rmsprop, nadam, adagrad)                           | 'adagrad'                                   |
| -bw, --beam_width      | Beam search width                                                        | 1                                           |
| -lp, --length_penalty  | Length penalty for beam search                                           | 0.6                                         |
| -tf, --teacher_forcing | Teacher forcing ratio                                                    | 0.7                                         |
| -bd, --bi_dir          | Use bidirectional encoder                                                | True                                        |
| -wl, --w_log           | Whether to log to WandB (1 for yes, 0 for no)                            | 0                                           |
| -wp, --wandb_project   | Project name used to track experiments in Weights & Biases dashboard     | 'DL-Assignment-3'                           |
| -we, --wandb_entity    | Wandb Entity used to track experiments in the Weights & Biases dashboard | 'cs23m053'                                  |

Example command to run the training script:

```bash
python train_vanilla.py -dp 'your/dataset/path/up/to/aksharantar_sampled' -lg 'hin'
```

## Output Metrics

During training and validation, the following output metrics are provided:

- **Train Accuracy**: The character-level accuracy of predictions on the training data.
- **Train Loss**: The average loss calculated during training.
- **Validation Accuracy Char**: The character-level accuracy of predictions on the validation data.
- **Validation Loss**: The average loss calculated during validation.
- **Validation Accuracy Word**: The word-level accuracy of predictions on the validation data using beam search.
- **Correct Prediction**: The number of correct predictions out of the total validation data samples.

These metrics provide insights into the performance of the Seq2Seq model during training and validation. Character-level accuracy measures how accurately the model predicts individual characters, while word-level accuracy assesses the correctness of entire output sequences.

---

# BASE Seq2Seq MODEL WITH ATTENTION

This repository contains a Python implementation of a sequence-to-sequence (Seq2Seq) model with attention mechanism for sequence prediction tasks. The Seq2Seq model is implemented using PyTorch and includes an attention mechanism to focus on relevant parts of the input sequence during decoding.

## Usage

To use the Seq2Seq model with attention, follow the steps below:

1. Import the necessary classes from the provided code.

2. Initialize an instance of the `Attention` class with the required parameter:

   - `hidden_size`: Size of the hidden state.

3. Initialize an instance of the `Encoder` class with the required parameters:

   - `input_size`: Size of the input vocabulary.
   - `embedding_size`: Size of the embedding layer.
   - `hidden_size`: Size of the hidden state in the RNN.
   - `num_layers`: Number of layers in the RNN.
   - `dropout`: Dropout rate for regularization.
   - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

4. Initialize an instance of the `Decoder` class with the required parameters:

   - `input_size`: Size of the input vocabulary.
   - `embedding_size`: Size of the embedding layer.
   - `hidden_size`: Size of the hidden state in the RNN.
   - `output_size`: Size of the output vocabulary.
   - `num_layers`: Number of layers in the RNN.
   - `dropout`: Dropout rate for regularization.
   - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

5. Initialize an instance of the `Seq2Seq` class with the required parameters:

   - `encoder`: Instance of the Encoder class.
   - `decoder`: Instance of the Decoder class.
   - `output_index_size`: Size of the target vocabulary.
   - `teacher_force_ratio`: Ratio of teacher forcing during training.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

6. Train the Seq2Seq model using the `train()` function with the required arguments:
   - `model`: The Seq2Seq model to be trained.
   - `num_epochs`: Number of training epochs.
   - `criterion`: Loss criterion for training.
   - `optimizer`: Optimizer for training.
   - `input_data`: Training input data batch.
   - `output_data`: Training target data batch.
   - `val_input_data`: Validation input data batch.
   - `val_output_data`: Validation target data batch.
   - `input_index`: Mapping from characters to integers for the input vocabulary.
   - `output_index`: Mapping from characters to integers for the output vocabulary.
   - `output_index_rev`: Reverse mapping from integers to characters for the output vocabulary.
   - `beam_width`: Beam width for beam search.
   - `length_penalty`: Length penalty for beam search.
   - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
   - `max_len`: Maximum length of sequences.
   - `wandb_log`: Whether to log to Weights & Biases (1 for yes, 0 for no).

## Classes

### Encoder

- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
- **Methods**:
  - `__init__()`: Initializes the encoder.
  - `forward()`: Performs forward pass through the encoder.

### Attention

- **Data Members**:
  - `hidden_size`: Size of the hidden state.
- **Methods**:
  - `__init__()`: Initializes the Attention mechanism.
  - `dot_score()`: Calculates the dot product attention scores between the decoder hidden state and encoder outputs.
  - `forward()`: Performs forward pass through the Attention mechanism.

### Decoder

- **Data Members**:
  - `input_size`: Size of the input vocabulary.
  - `embedding_size`: Size of the embedding layer.
  - `hidden_size`: Size of the hidden state in the RNN.
  - `output_size`: Size of the output vocabulary.
  - `num_layers`: Number of layers in the RNN.
  - `dropout`: Dropout rate for regularization.
  - `bidirectional`: Boolean indicating whether the RNN is bidirectional.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').
- **Methods**:
  - `__init__()`: Initializes the decoder.
  - `forward()`: Performs forward pass through the decoder.

### Seq2Seq

- **Data Members**:

  - `encoder`: Instance of the Encoder class.
  - `decoder`: Instance of the Decoder class.
  - `target_vocab_size`: Size of the target vocabulary.
  - `teacher_force_ratio`: Ratio of teacher forcing during training.
  - `cell_type`: Type of RNN cell used ('LSTM', 'GRU', or 'RNN').

- **Methods**:
  - `__init__()`: Initializes the Seq2Seq model.
  - `forward()`: Performs forward pass through the model.

## Command-line Arguments

| Argument               | Description                                                              | Default Value                                |
| ---------------------- | ------------------------------------------------------------------------ | -------------------------------------------- |
| -dp, --dataset_path    | Path to the data folder                                                  | '/kaggle/input/dl-ass3/aksharantar_sampled/' |
| -lg, --language        | Language for which training is to be done                                | 'hin'                                        |
| -es, --embedding_size  | Embedding size                                                           | 256                                          |
| -hs, --hidden_size     | Hidden size                                                              | 512                                          |
| -nl, --num_layers      | Number of layers                                                         | 3                                            |
| -ct, --cell_type       | Cell type (RNN, LSTM, GRU)                                               | 'LSTM'                                       |
| -do, --dropout         | Dropout rate                                                             | 0.3                                          |
| -lr, --learning_rate   | Learning rate                                                            | 0.01                                         |
| -bs, --batch_size      | Batch size                                                               | 32                                           |
| -ep, --num_epochs      | Number of epochs                                                         | 10                                           |
| -op, --optimizer       | Optimizer (adam, sgd, rmsprop, nadam, adagrad)                           | 'adagrad'                                    |
| -bw, --beam_width      | Beam search width                                                        | 1                                            |
| -lp, --length_penalty  | Length penalty for beam search                                           | 0.6                                          |
| -tf, --teacher_forcing | Teacher forcing ratio                                                    | 0.7                                          |
| -bd, --bi_dir          | Use bidirectional encoder                                                | True                                         |
| -wl, --w_log           | Whether to log to WandB (1 for yes, 0 for no)                            | 0                                            |
| -wp, --wandb_project   | Project name used to track experiments in Weights & Biases dashboard     | 'DL-Assignment-3'                            |
| -we, --wandb_entity    | Wandb Entity used to track experiments in the Weights & Biases dashboard | 'cs23m053'                                   |

## Example Usage

```bash
python train_attention.py -dp 'your/dataset/path/up/to/aksharantar_sampled' -lg 'hin'
```

## Output Metrics

During training and validation, the following output metrics are provided:

- **Train Accuracy**: The character-level accuracy of predictions on the training data.
- **Train Loss**: The average loss calculated during training.
- **Validation Accuracy Char**: The character-level accuracy of predictions on the validation data.
- **Validation Loss**: The average loss calculated during validation.
- **Validation Accuracy Word**: The word-level accuracy of predictions on the validation data using beam search.
- **Correct Prediction**: The number of correct predictions out of the total validation data samples.

These metrics provide insights into the performance of the Seq2Seq model during training and validation. Character-level accuracy measures how accurately the model predicts individual characters, while word-level accuracy assesses the correctness of entire output sequences.
