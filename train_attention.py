# adding necessary imports
import csv
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import random
import wandb
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL-Assignment-3')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m053')
parser.add_argument('-dp', '--dataset_path' , help='Specify path to dataset ../aksharantar_sampled.' , type=str, default='/kaggle/input/dl-ass3/aksharantar_sampled')
parser.add_argument('-ct', '--cell_type' , help='Type of cell.' , type=str,choices = ['RNN', 'LSTM', 'GRU'], default='LSTM')
parser.add_argument('-bd', '--bi_dir' , help="choices: ['True', 'False']",choices = [True, False], type=bool, default=True)
parser.add_argument('-nl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=3)
parser.add_argument('-hs', '--hidden_size', help ='Size of hidden layer.', type=int, default=512)
parser.add_argument('-es', '--embedding_size', help ='Size of embedding layer..', type=int, default=256)
parser.add_argument('-op', '--optimizer', help = "choices: ['sgd', 'rmsprop', 'adam', 'adagrad']", choices = ['sgd','rmsprop', 'adam', 'adagrad'],type=str, default = 'adagrad')
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.01)
parser.add_argument('-bs', '--batch_size', help='Batch size used to train neural network.', type=int, default=32)
parser.add_argument('-do', '--dropout', help='Dropout probabilty.', type=float, default=0.3)
parser.add_argument('-tf', '--teacher_fr', help='Teacher forcing ratio.', type=float, default=0.7)
parser.add_argument('-ep', '--num_epochs', help='Number of epochs to train neural network.', type=int, default=10)
parser.add_argument('-lp', '--length_penalty', help='Length penalty for beam search.', type=float, default=0.6)
parser.add_argument('-bw', '--beam_width', help='Beam width for beam search.', type=int, default=1)
parser.add_argument('-tp', '--total_params', help='To print total parameters of the model.', choices = [0, 1], type=int, default=1)
parser.add_argument('-em', '--evaluate_model', help='To evaluate model on test data.', choices = [0, 1], type=int, default=1)
parser.add_argument('-wl', '--w_log', help='To log results on wandb.', type=int, default=0)
# parser.add_argument('-ag', '--plot_attention_heatmap', help='To plot attention heatmaps.', choices = [0, 1], type=int, default=1)

arguments = parser.parse_args()



# Setting devide to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def loadData(params):
    """
    This function loads and preprocesses the data for machine translation.

    Args:
        params (dict): A dictionary containing parameters for data loading.
            - 'language': The language of the dataset (e.g., 'en', 'fr').
            - 'dataset_path': The path to the directory containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed data.
    """
    language = params['language']
    dataset_path = params['dataset_path']
    # Construct file paths for training, validation, and testing data
    train_path = os.path.join(dataset_path, language, language + '_train.csv')
    val_path = os.path.join(dataset_path, language, language + '_valid.csv')
    test_path = os.path.join(dataset_path, language, language + '_test.csv')

    # Open and read data from CSV files using UTF-8 encoding for proper character handling
    train_data = csv.reader(open(train_path, encoding='utf8'))
    val_data = csv.reader(open(val_path, encoding='utf8'))
    test_data = csv.reader(open(test_path, encoding='utf8'))

    # Initialize empty lists to store source and target language sentences
    train_words, train_translations = [], []
    val_words, val_translations = [], []
    test_words, test_translations = [], []

    # Define special symbols for padding, sentence start, and sentence end
    pad, start, end = '', '^', '$'

    # Preprocess data by adding special symbols to sentence ends
    for pair in train_data:
        train_words.append(pair[0] + end)
        train_translations.append(start + pair[1] + end)
    for pair in val_data:
        val_words.append(pair[0] + end)
        val_translations.append(start + pair[1] + end)
    for pair in test_data:
        test_words.append(pair[0] + end)
        test_translations.append(start + pair[1] + end)
    
    # Convert lists to NumPy arrays for efficient processing
    train_words , train_translations = np.array(train_words), np.array(train_translations)
    val_words , val_translations = np.array(val_words), np.array(val_translations)
    test_words , test_translations = np.array(test_words), np.array(test_translations)
    
    # Create sets to store unique characters in source and target vocabulary
    input_vocab = set()
    output_vocab = set()
    
    # Iterate through words to collect all unique characters
    for w in train_words:
        for c in w:
            input_vocab.add(c)
    for w in val_words:
        for c in w:
            input_vocab.add(c)
    for w in test_words:
        for c in w:
            input_vocab.add(c)
            
    for w in train_translations:
        for c in w:
            output_vocab.add(c)
    for w in val_translations:
        for c in w:
            output_vocab.add(c)
    for w in test_translations:
        for c in w:
            output_vocab.add(c)
    
    # Remove special symbols from vocabulary sets
    input_vocab.remove(end)
    output_vocab.remove(start)
    output_vocab.remove(end)

    # Sort vocabulary sets and add special symbols as prefixes
    input_vocab, output_vocab = [pad, start, end] + list(sorted(input_vocab)), [pad, start, end] + list(sorted(output_vocab))

    # Create dictionaries to map characters to their indices and vice versa
    input_index = {char: idx for idx, char in enumerate(input_vocab)}
    output_index = {char: idx for idx, char in enumerate(output_vocab)}
    input_index_rev = {idx: char for char, idx in input_index.items()}
    output_index_rev = {idx: char for char, idx in output_index.items()}

    # Find the maximum length of sentences in source and target data
    max_enc_len = max([len(word) for word in np.hstack((train_words, test_words, val_words))])
    max_dec_len = max([len(word) for word in np.hstack((train_translations, val_translations, test_translations))])
    max_len = max(max_enc_len, max_dec_len)
      
    # returning data
    preprocessed_data = {
        'SOS' : start,
        'EOS' : end,
        'PAD' : pad,
        'train_words' : train_words,
        'train_translations' : train_translations,
        'val_words' : val_words,
        'val_translations' : val_translations,
        'test_words' : test_words,
        'test_translations' : test_translations,
        'max_enc_len' : max_enc_len,
        'max_dec_len' : max_dec_len,
        'max_len' : max_len,
        'input_index' : input_index,
        'output_index' : output_index,
        'input_index_rev' : input_index_rev,
        'output_index_rev' : output_index_rev
    }
    return preprocessed_data

def create_tensor(preprocessed_data):
    """
    This function creates PyTorch tensors from the preprocessed data.

    Args:
        preprocessed_data (dict): A dictionary containing the preprocessed data.

    Returns:
        dict: A dictionary containing PyTorch tensors for training, validation, and testing data.
    """

    # Define the maximum sequence length based on preprocessed data
    max_len = preprocessed_data['max_len']

    # Create empty NumPy arrays to store padded sequences for training, validation, and testing
    input_data = np.zeros((max_len, len(preprocessed_data['train_words'])), dtype='int64')
    output_data = np.zeros((max_len, len(preprocessed_data['train_words'])), dtype='int64')
    
    val_input_data = np.zeros((max_len, len(preprocessed_data['val_words'])), dtype='int64')
    val_output_data = np.zeros((max_len, len(preprocessed_data['val_words'])), dtype='int64')
    
    test_input_data = np.zeros((max_len, len(preprocessed_data['test_words'])), dtype='int64')
    test_output_data = np.zeros((max_len, len(preprocessed_data['test_words'])), dtype='int64')

    # Iterate through training data and populate tensors with character indices
    for idx, (w, t) in enumerate(zip(preprocessed_data['train_words'], preprocessed_data['train_translations'])):
        for i, char in enumerate(w):
            input_data[i, idx] = preprocessed_data['input_index'][char]
        for i, char in enumerate(t):
            output_data[i, idx] = preprocessed_data['output_index'][char]

    # Repeat the process for validation and testing data
    for idx, (w, t) in enumerate(zip(preprocessed_data['val_words'], preprocessed_data['val_translations'])):
        for i, char in enumerate(w):
            val_input_data[i, idx] = preprocessed_data['input_index'][char]
        for i, char in enumerate(t):
            val_output_data[i, idx] = preprocessed_data['output_index'][char]

    for idx, (w, t) in enumerate(zip(preprocessed_data['test_words'], preprocessed_data['test_translations'])):
        for i, char in enumerate(w):
            test_input_data[i, idx] = preprocessed_data['input_index'][char]
        for i, char in enumerate(t):
            test_output_data[i, idx] = preprocessed_data['output_index'][char]

    # Convert NumPy arrays to PyTorch tensors for efficient GPU processing (if available)
    input_data, output_data = torch.tensor(input_data, dtype=torch.int64), torch.tensor(output_data, dtype=torch.int64)
    val_input_data, val_output_data = torch.tensor(val_input_data, dtype=torch.int64), torch.tensor(val_output_data, dtype=torch.int64)
    test_input_data, test_output_data = torch.tensor(test_input_data, dtype=torch.int64), torch.tensor(test_output_data, dtype=torch.int64)

    # Create a dictionary to store all the tensors
    tensors = {
        'input_data': input_data,
        'output_data': output_data,
        'val_input_data': val_input_data,
        'val_output_data': val_output_data,
        'test_input_data': test_input_data,
        'test_output_data': test_output_data
    }

    return tensors

class Attention(nn.Module):
  """
  This class implements an attention mechanism for a Seq2Seq model.

  The attention mechanism allows the decoder to focus on relevant parts of the encoder output
  during the decoding process, improving the model's ability to translate sequences.
  """

  def __init__(self, hidden_size):
    """
    Initializes the attention layer.

    Args:
        hidden_size (int): The size of the hidden state vectors in the model.
    """
    super(Attention, self).__init__()
    self.hidden_size = hidden_size  # Store the hidden size for calculations

  def dot_score(self, hidden_state, encoder_states):
    """
    Calculates the attention scores between the decoder hidden state and encoder outputs.

    Args:
        hidden_state (torch.Tensor): The hidden state of the decoder at a specific time step.
        encoder_states (torch.Tensor): A tensor containing the encoder outputs for all time steps.

    Returns:
        torch.Tensor: A tensor containing the attention scores for each encoder output.
    """
    # Calculate the dot product between the decoder hidden state and each encoder output vector
    return torch.sum(hidden_state * encoder_states, dim=2)  # Summation over the feature dimension

  def forward(self, hidden, encoder_outputs):
    """
    Calculates the attention weights for a given decoder hidden state and encoder outputs.

    Args:
        hidden (torch.Tensor): The hidden state of the decoder at a specific time step.
        encoder_outputs (torch.Tensor): A tensor containing the encoder outputs for all time steps.

    Returns:
        torch.Tensor: A tensor containing the attention weights for each encoder output.
    """
    # Calculate attention scores using dot product
    attn_scores = self.dot_score(hidden, encoder_outputs)

    # Transpose the scores for softmax calculation (scores for each encoder output)
    attn_scores = attn_scores.t()

    # Apply softmax to get normalized attention weights (sum to 1)
    attn_weights = F.softmax(attn_scores, dim=1)

    # Unsqueeze to add a dimension for compatibility with decoder calculations
    return attn_weights.unsqueeze(1)

class Encoder_Attention(nn.Module):
  """
  This class implements the encoder part of a Seq2Seq model with attention.

  The encoder takes a sequence of word indices as input and processes it to
  generate an encoded representation that captures the meaning of the sequence.
  """

  def __init__(self, params, preprocessed_data):
    """
    Initializes the encoder.

    Args:
        params (dict): A dictionary containing hyperparameters for the model.
        preprocessed_data (dict): A dictionary containing the preprocessed data.
    """
    super(Encoder_Attention, self).__init__()

    # Get hyperparameters
    self.cell_type = params['cell_type']  # Type of RNN cell (RNN, LSTM, or GRU)
    self.bi_directional = params['bi_dir']  # Whether to use a bidirectional RNN
    self.embedding_size = params['embedding_size']  # Dimensionality of word embeddings
    self.hidden_size = params['hidden_size']  # Dimensionality of hidden state
    self.dropout = nn.Dropout(params['dropout'])  # Dropout layer for regularization

    # Embedding layer
    self.embedding = nn.Embedding(len(preprocessed_data['input_index']), self.embedding_size)
    # Look up an embedding vector for each word index in the input sequence

    # Choose RNN cell based on type
    if self.cell_type == 'RNN':
      self.cell = nn.RNN(self.embedding_size, self.hidden_size, params['num_layers_enc'], dropout=params['dropout'], bidirectional=self.bi_directional)
    elif self.cell_type == 'LSTM':
      self.cell = nn.LSTM(self.embedding_size, self.hidden_size, params['num_layers_enc'], dropout=params['dropout'], bidirectional=self.bi_directional)
    elif self.cell_type == 'GRU':
      self.cell = nn.GRU(self.embedding_size, self.hidden_size, params['num_layers_enc'], dropout=params['dropout'], bidirectional=self.bi_directional)
    else:
      raise ValueError("Invalid type. Choose from 'RNN', 'LSTM', or 'GRU'.")

  def forward(self, x):
      """
      Performs the forward pass through the encoder.

      Args:
          x (torch.Tensor): A tensor containing a sequence of word indices.

      Returns:
          tuple:
              - encoder_states (torch.Tensor): The encoded representation of the input sequence for all time steps.
              - hidden (torch.Tensor): The hidden state of the RNN at the last time step (if unidirectional) or a tuple of hidden states for both directions (if bidirectional).
              - cell (torch.Tensor): The cell state of the LSTM at the last time step (if LSTM is used). (Optional, only returned for LSTMs)
      """
      embedding = self.dropout(self.embedding(x))
      if self.cell_type == 'LSTM':
          encoder_states, (hidden, cell) = self.cell(embedding)
          if self.bi_directional:
              encoder_states = encoder_states[:, :, :self.hidden_size] + encoder_states[:, : ,self.hidden_size:]
          return encoder_states, hidden, cell
      else:
          encoder_states, hidden = self.cell(embedding)
          if self.bi_directional:
              encoder_states = encoder_states[:, :, :self.hidden_size] + encoder_states[:, : ,self.hidden_size:]
          return encoder_states, hidden

class Decoder_Attention(nn.Module):
  """
  This class implements the decoder part of a Seq2Seq model with attention.

  The decoder takes an embedded target sequence (one word at a time) and the
  encoder outputs as input, and generates a sequence of predicted words.
  It uses attention to focus on relevant parts of the encoder outputs
  during the decoding process.
  """

  def __init__(self, params, preprocessed_data):
    """
    Initializes the decoder.

    Args:
        params (dict): A dictionary containing hyperparameters for the model.
        preprocessed_data (dict): A dictionary containing the preprocessed data.
    """
    super(Decoder_Attention, self).__init__()

    # Get hyperparameters
    self.cell_type = params['cell_type']  # Type of RNN cell (RNN, LSTM, or GRU)
    self.num_layers = params['num_layers_dec']  # Number of decoder layers
    self.dropout = nn.Dropout(params['dropout'])  # Dropout layer for regularization
    self.embedding_size = params['embedding_size']  # Dimensionality of word embeddings

    # Embedding layer
    self.embedding = nn.Embedding(len(preprocessed_data['output_index']), params['embedding_size'])
    # Look up an embedding vector for each word index in the output sequence

    # Choose RNN cell based on type
    if self.cell_type == 'RNN':
      self.cell = nn.RNN(params['embedding_size'], params['hidden_size'], self.num_layers, dropout=params['dropout'])
    elif self.cell_type == 'LSTM':
      self.cell = nn.LSTM(params['embedding_size'], params['hidden_size'], self.num_layers, dropout=params['dropout'])
    elif self.cell_type == 'GRU':
      self.cell = nn.GRU(params['embedding_size'], params['hidden_size'], self.num_layers, dropout=params['dropout'])
    else:
      raise ValueError("Invalid type. Choose from 'RNN', 'LSTM', or 'GRU'.")

    # Layers for combining decoder output and context vector
    self.concat = nn.Linear(params['hidden_size'] * 2, params['hidden_size'])  # Linear transformation for concatenation
    self.fc = nn.Linear(params['hidden_size'], len(preprocessed_data['output_index']))  # Final linear layer for prediction

    # Attention layer
    self.attn = Attention(params['hidden_size'])  # Attention mechanism

    # Softmax for probability distribution
    self.log_softmax = nn.LogSoftmax(dim = 1)

  def forward(self, x, encoder_states, hidden, cell):
    """
    Performs the forward pass through the decoder for a single time step.

    Args:
        x (torch.Tensor): A tensor containing a single word index (input to decoder at this step).
        encoder_states (torch.Tensor): The encoded representation of the input sequence from the encoder.
        hidden (torch.Tensor): The hidden state of the decoder from the previous time step.
        cell (torch.Tensor): The cell state of the LSTM decoder from the previous time step (if LSTM is used).

    Returns:
        tuple:
            - predictions (torch.Tensor): The log-softmax probabilities of the next predicted word.
            - hidden (torch.Tensor): The hidden state of the decoder for the current time step.
            - cell (torch.Tensor): The cell state of the LSTM decoder for the current time step (if LSTM is used).
            - attention_weights (torch.Tensor): The attention weights for the current time step.
    """
    # Embed the input word
    embedding = self.dropout(self.embedding(x.unsqueeze(0)))
    # Pass the embedded word through the chosen cell
    if self.cell_type == 'LSTM':
        outputs, (hidden, cell) = self.cell(embedding, (hidden, cell))
        attention_weights = self.attn(outputs, encoder_states)
        context = attention_weights.bmm(encoder_states.transpose(0, 1))
        outputs = outputs.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((outputs, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        predictions = self.log_softmax(self.fc(concat_output))
        return predictions, hidden, cell, attention_weights.squeeze(1)
    else:
        outputs, (hidden) = self.cell(embedding, hidden)
        attention_weights = self.attn(outputs, encoder_states)
        context = attention_weights.bmm(encoder_states.transpose(0, 1))
        outputs = outputs.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((outputs, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        predictions = self.log_softmax(self.fc(concat_output))
        return predictions, hidden, attention_weights.squeeze(1)

class Seq2Seq_Attention(nn.Module):
  """
  This class implements a Seq2Seq model with attention mechanism.

  The model takes an encoded source sequence and a target sequence as input,
  and generates a predicted target sequence using the decoder with attention.
  """

  def __init__(self, encoder, decoder, params, preprocessed_data):
    """
    Initializes the Seq2Seq model.

    Args:
        encoder (nn.Module): The encoder module of the model.
        decoder (nn.Module): The decoder module of the model.
        params (dict): A dictionary containing hyperparameters for the model.
        preprocessed_data (dict): A dictionary containing the preprocessed data.
    """
    super(Seq2Seq_Attention, self).__init__()

    # Get hyperparameters
    self.cell_type = params['cell_type']  # Type of RNN cell (RNN, LSTM, or GRU)
    self.encoder = encoder  # Encoder module
    self.decoder = decoder  # Decoder module
    self.num_layers_dec = params['num_layers_dec']  # Number of decoder layers
    self.output_index_len = len(preprocessed_data['output_index'])  # Vocabulary size for output language
    self.tfr = params['teacher_fr']  # Teacher forcing ratio

  def forward(self, source, target):
    """
    Performs the forward pass through the entire Seq2Seq model.

    Args:
        source (torch.Tensor): A tensor containing the source sequence (encoder input).
        target (torch.Tensor): A tensor containing the target sequence (ground truth for training or prediction).

    Returns:
        torch.Tensor: A tensor containing the predicted target sequence log-softmax probabilities.
    """

    # Get batch size and target sequence length
    batch_size, target_len = source.shape[1], target.shape[0]

    # Start with the first word from the target sequence
    x = target[0, :]  # First element from each batch in the target sequence

    # Initialize empty tensor to store predictions
    outputs = torch.zeros(target_len, batch_size, self.output_index_len).to(device)

    # Get encoder outputs (encoded representation of the source sequence)
    if self.cell_type == 'LSTM':
      encoder_op, hidden, cell = self.encoder(source)
      # Truncate cell state to match decoder layer number
      cell = cell[:self.decoder.num_layers]
    else:
      encoder_op, hidden = self.encoder(source)
    # Truncate hidden state to match decoder layer number
    hidden = hidden[:self.decoder.num_layers]

    # Iterate over the target sequence length (decoding process)
    for t in range(1, target_len):
      # Use LSTM cell state or hidden state depending on cell type
      if self.cell_type == 'LSTM':
        output, hidden, cell, _ = self.decoder(x, encoder_op, hidden, cell)
      else:
        output, hidden, _ = self.decoder(x, encoder_op, hidden, None)

      # Store the predicted word probabilities and get the most likely word index
      outputs[t], best_guess = output, output.argmax(1)

      # Teacher forcing: Choose predicted word or target word based on random probability
      x = best_guess if random.random() >= self.tfr else target[t]

    # Return the tensor containing the predicted target sequence log-softmax probabilities
    return outputs

def get_optim(model, params):
    """
    This function creates an optimizer object based on the specified parameters.

    Args:
        model (nn.Module): The Seq2Seq model instance.
        params (dict): A dictionary containing hyperparameters for the optimizer.
            - 'optimizer' (str): The name of the optimizer to use (e.g., 'sgd', 'adam', 'rmsprop', 'adagrad').
            - 'learning_rate' (float): The learning rate for the optimizer.

    Returns:
        optim.Optimizer: An optimizer object for training the model.
    """

    optimizer_name = params['optimizer'].lower()  # Convert optimizer name to lowercase for case-insensitive matching

    # Define the optimizer based on the specified name
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'], alpha=0.99, eps=1e-8)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=params['learning_rate'], lr_decay=0, weight_decay=0,
                                  initial_accumulator_value=0, eps=1e-10)
    else:
        raise ValueError("Invalid optimizer. Choose from 'sgd', 'adam', 'rmsprop', or 'adagrad'.")

    return optimizer

def get_total_parameters(model):
  """
  This function calculates the total number of trainable parameters in a PyTorch model.

  Args:
      model (nn.Module): The PyTorch model to analyze.

  Returns:
      int: The total number of trainable parameters in the model.
  """

  # Filter only trainable parameters
  total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

  return total_params

def beam_search(model, word, preprocessed_data, params, bw = 1, lp = 0.6):
    """
    This function performs beam search to generate a translated sequence for a given source word sequence.

    Args:
        model (nn.Module): The Seq2Seq model instance.
        word (str): The source word sequence to translate.
        preprocessed_data (dict): A dictionary containing the preprocessed data.
        bw (int): Beam width for beam search.
        lp (float): Length penalty factor for beam search.
        ct (str): The type of RNN cell used in the model (LSTM or GRU).

    Returns:
        str: The predicted translated word sequence.
    """
    data = np.zeros((preprocessed_data['max_len']+1, 1), dtype=np.int32)
    for idx, char in enumerate(word):
        data[idx, 0] = preprocessed_data['input_index'][char]
    data[idx + 1, 0] = preprocessed_data['input_index'][preprocessed_data['EOS']]
    data = torch.tensor(data, dtype=torch.int32).to(device)
    with torch.no_grad():
        if params['cell_type'] == 'LSTM':
            outputs, hidden, cell = model.encoder(data)
            cell =  cell[:params['num_layers_dec']]
        else:
            outputs, hidden = model.encoder(data)
    hidden =  hidden[:params['num_layers_dec']]
    output_start = preprocessed_data['output_index'][preprocessed_data['SOS']]
    out_reshape = np.array(output_start).reshape(1,)
    hidden_par = hidden.unsqueeze(0)
    initial_sequence = torch.tensor(out_reshape).to(device)
    beam = [(0.0, initial_sequence, hidden_par)]
    for i in range(len(preprocessed_data['output_index'])):
        candidates = []
        for score, seq, hidden in beam:
            if seq[-1].item() == preprocessed_data['output_index'][preprocessed_data['EOS']]:
                candidates.append((score, seq, hidden))
                continue
            reshape_last = np.array(seq[-1].item()).reshape(1, )
            hdn = hidden.squeeze(0) 
            x = torch.tensor(reshape_last).to(device)
            if params['cell_type'] == 'LSTM':
                output, hidden, cell, _ = model.decoder(x, outputs, hdn, cell)
            else:
                output, hidden, _ = model.decoder(x, outputs, hdn, None)
            topk_probs, topk_tokens = torch.topk(F.softmax(output, dim=1), k = bw)               
            for prob, token in zip(topk_probs[0], topk_tokens[0]):
                new_seq = torch.cat((seq, token.unsqueeze(0)), dim=0)
                candidate_score = score + torch.log(prob).item() / (((len(new_seq) - 1) / 5) ** lp)
                candidates.append((candidate_score, new_seq, hidden.unsqueeze(0)))
        beam = heapq.nlargest(bw, candidates, key=lambda x: x[0])
    _, best_sequence, _ = max(beam, key=lambda x: x[0]) 
    prediction = ''.join([preprocessed_data['output_index_rev'][token.item()] for token in best_sequence[1:]])
    return prediction[:-1]

def train(model, criterion, optimizer, preprocessed_data, tensors, params):
    """
    This function trains the Seq2Seq model and performs validation.

    Args:
        model (nn.Module): The Seq2Seq model instance.
        criterion (nn.Module): The loss function for training (e.g., nn.NLLLoss).
        optimizer (optim.Optimizer): The optimizer used for training (e.g., Adam).
        preprocessed_data (dict): A dictionary containing the preprocessed data.
        tensors (dict): A dictionary containing PyTorch tensors for training and validation data.
        params (dict): A dictionary containing hyperparameters for training and evaluation.

    Returns:
        tuple:
            - model (nn.Module): The trained Seq2Seq model.
            - val_accuracy (float): Overall character-level accuracy on the validation set.
            - val_accuracy_beam (float): Overall word-level accuracy on the validation set using beam search.
    """
    # splitting data in batches
    train_data, train_result = torch.split(tensors['input_data'], params['batch_size'], dim = 1), torch.split(tensors['output_data'], params['batch_size'], dim = 1)
    val_data, val_result = torch.split(tensors['val_input_data'], params['batch_size'], dim=1), torch.split(tensors['val_output_data'], params['batch_size'], dim=1)
    
    # performing epochs
    for epoch in range(params['num_epochs']):
        total_words = 0
        correct_pred = 0
        total_loss = 0
        model.train()
        # training the model
        with tqdm(total = len(train_data), desc = 'Training') as pbar:
            for i, (x, y) in enumerate(zip(train_data, train_result)):
                target, inp_data = y.to(device), x.to(device)
                optimizer.zero_grad()
                output = model(inp_data, target)
                target = target.reshape(-1)
                output = output.reshape(-1, output.shape[2])
                # adding padding mask to ignore 0 paddding while calculating accuracy
                pad_mask = (target != preprocessed_data['output_index'][preprocessed_data['PAD']])
                target = target[pad_mask]
                output = output[pad_mask]
                
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                total_loss += loss.item()
                total_words += target.size(0)
                correct_pred += torch.sum(torch.argmax(output, dim=1) == target).item()
                pbar.update(1)
        train_accuracy = (correct_pred / total_words)*100
        train_loss = total_loss / len(train_data)
        # setting model in evaluation mode to validate on validation set
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            val_total_words = 0
            val_correct_pred = 0
            with tqdm(total = len(val_data), desc = 'Validation') as pbar:
                for x_val, y_val in zip(val_data, val_result):
                    target_val, inp_data_val = y_val.to(device), x_val.to(device)
                    output_val = model(inp_data_val, target_val)
                    target_val = target_val.reshape(-1)
                    output_val = output_val.reshape(-1, output_val.shape[2])
                    
                    pad_mask = (target_val != preprocessed_data['output_index'][preprocessed_data['PAD']])
                    target_val = target_val[pad_mask]
                    output_val = output_val[pad_mask]
                    
                    val_loss = criterion(output_val, target_val)
                    val_total_loss += val_loss.item()
                    val_total_words += target_val.size(0)
                    val_correct_pred += torch.sum(torch.argmax(output_val, dim=1) == target_val).item()
                    pbar.update(1)
            val_accuracy = (val_correct_pred / val_total_words) * 100
            val_loss = val_total_loss / len(val_data)
            
            # checking word level accuracy on validation set
            correct_pred = 0
            total_words = len(preprocessed_data['val_words'])
            with tqdm(total = total_words, desc = 'Beam_Validation') as pbar_:
                for word, translation in zip(preprocessed_data['val_words'], preprocessed_data['val_translations']):
                    ans = beam_search(model, word, preprocessed_data, params, params['beam_width'], params['length_penalty'])
                    if ans == translation[1:-1]:
                        correct_pred += 1
                    pbar_.update(1)
        val_accuracy_beam = (correct_pred / total_words) * 100
        
        # logging the results
        print(f'''Epoch : {epoch+1}
              Train Accuracy : {train_accuracy:.4f}, Train Loss : {train_loss:.4f}
              Validation Accuracy Char Level : {val_accuracy:.4f}, Validation Loss : {val_loss:.4f}
              Validation Accuracy Word Level : {val_accuracy_beam:.4f},  Correctly predicted : {correct_pred}/{total_words}''')
        if params['w_log']:
            wandb.log(
                    {
                        'epoch': epoch+1,
                        'training_loss' : train_loss,
                        'training_accuracy_char' : train_accuracy,
                        'validation_loss' : val_loss,
                        'validation_accuracy_char' : val_accuracy,
                        'validation_accuracy_word' : val_accuracy_beam,
                        'correctly_predicted' : correct_pred
                    }
                )
    return model, val_accuracy, val_accuracy_beam

def evaluate_model(trained_model, data_type, preprocessed_data, params, bw=1, lp=0.6):
  """
  This function evaluates the Seq2Seq model on a specified data type (e.g., 'val', 'test').

  Args:
      trained_model (nn.Module): The trained Seq2Seq model instance.
      data_type (str): The type of data to evaluate on (e.g., 'val', 'test').
      preprocessed_data (dict): A dictionary containing the preprocessed data.
      params (dict): A dictionary containing hyperparameters for evaluation (beam search).
          - 'bw' (int, optional): Beam width for beam search (default 1).
          - 'lp' (float, optional): Length penalty factor for beam search (default 0.6).
      cell_type (str): The type of RNN cell used in the model (extracted from params).

  Returns:
      tuple:
          - words (list): List of source words (without start/end tokens).
          - translations (list): List of reference translations (without start/end tokens).
          - predictions (list): List of predicted translated sequences (without start/end tokens).
          - results (list): List of 'Yes' or 'No' indicating correct/incorrect predictions.
          - accuracy (float): Overall accuracy of the model on the data type.
          - correct_pred (int): Number of correctly predicted translations.
  """

  # Extract data indices based on data type (e.g., 'val_words', 'test_translations')
  data_words = data_type + '_words'
  data_translations = data_type + '_translations'

  # Set the model to evaluation mode
  trained_model.eval()

  # Initialize variables for tracking results
  correct_pred = 0
  words, translations, predictions, results = [], [], [], []
  total_words = len(preprocessed_data[data_words])

  # Progress bar for iterating through data
  with tqdm(total=total_words, desc=data_type) as pbar:
    for word, translation in zip(preprocessed_data[data_words], preprocessed_data[data_translations]):
      # Perform beam search to get the predicted translation
      ans = beam_search(trained_model, word, preprocessed_data, params, bw, lp)

      # Extract source and target sequences without start/end tokens
      words.append(word[:-1])
      translations.append(translation[1:-1])
      predictions.append(ans)

      # Check if the prediction matches the reference translation
      if ans == translation[1:-1]:
        correct_pred += 1
        results.append('Yes')
      else:
        results.append('No')

      # Update progress bar
      pbar.update(1)

  # Calculate overall accuracy
  accuracy = (correct_pred / total_words) * 100

  return words, translations, predictions, results, accuracy, correct_pred

def predict(model, word, preprocessed_data, params):
    """
    This function generates a predicted translation for a given source word sequence.

    Args:
        model (nn.Module): The trained Seq2Seq model instance.
        word (str): The source word sequence to translate.
        preprocessed_data (dict): A dictionary containing the preprocessed data.
        params (dict): A dictionary containing hyperparameters for the model.
            - 'cell_type' (str): The type of RNN cell used in the model.

    Returns:
        str: The predicted translated word sequence.
    """

    # Create a zero-filled data tensor with extra row for end-of-sequence (EOS) token
    data = np.zeros((preprocessed_data['max_len'] + 1, 1), dtype=int)
    pred = ''  # Initialize an empty string to store the predicted translation

    # Encode the source word sequence (one word at a time)
    for t, char in enumerate(word):
        data[t, 0] = preprocessed_data['input_index'][char]
    data[(t + 1), 0] = preprocessed_data['input_index'][preprocessed_data['EOS']]  # Add EOS token
    data = torch.tensor(data, dtype=torch.int64).to(device)

  # Disable gradient calculation for efficiency during prediction
    with torch.no_grad():
    # Get the hidden state(s) from the encoder
        if params['cell_type'] == 'LSTM':
            outputs, hidden, cell = model.encoder(data)
            cell =  cell[:params['num_layers_dec']]
        else:
            outputs, hidden = model.encoder(data)
    hidden =  hidden[:params['num_layers_dec']]
    # Start token (SOS) for the decoder
    x = torch.tensor([preprocessed_data['output_index'][preprocessed_data['SOS']]]).to(device)
    attentions = torch.zeros(preprocessed_data['max_len'] + 1, 1, preprocessed_data['max_len'] + 1)
    
    # Greedy search for predicted translation
    for t in range(1, len(preprocessed_data['output_index'])):
        if params['cell_type'] == 'LSTM':
            output, hidden, cell, attn = model.decoder(x, outputs, hidden, cell)
        else:
            output, hidden, attn = model.decoder(x, outputs, hidden, None)
        
        # Convert the decoder output to the predicted character
        character = preprocessed_data['output_index_rev'][output.argmax(1).item()]
        attentions[t] = attn
        if character != preprocessed_data['EOS']:
            pred = pred + character
        else:
            break
        
        # Use the predicted character as the next input to the decoder
        x = torch.tensor([output.argmax(1)]).to(device)        
    return pred, attentions[:t+1]

def plot_attention_grid(model):
# getting 10 sample random pairs of words and translation from test data
  random_pairs = random.sample(list(zip(preprocessed_data['test_words'], preprocessed_data['test_translations'])), 10)
  inputs, outputs, attentions = [], [], []
  for i, (word_and_eos, translation_and_eos) in enumerate(random_pairs):
    word = word_and_eos[:-1]
    translation = translation_and_eos[:-1]
    output, attention = predict(model, word, preprocessed_data, params)
    attention = attention[1:, :, :(len(word))]
    inputs.append(word)
    outputs.append(' ' + output)  # Add space before predicted translation for better readability
    attentions.append(attention)

  fig, axes = plt.subplots(4, 3, figsize=(15, 15))
  fig.suptitle('Attention Matrix Grid', fontsize=14)

  for i in range(len(inputs)):
    word = inputs[i]  # Get the input word sequence
    translation = outputs[i]  # Get the corresponding translated sequence
    attention = attentions[i][:len(translation), :len(word)].squeeze(1).detach().numpy()  # Extract and reshape attention matrix

    ax = axes.flat[i]

    sns.heatmap(attention, cmap='YlGnBu', ax=ax)
    ax.set_xticks(np.arange(len(word)))  # X-axis ticks for input words
    ax.set_xticklabels(word, size=8)  # X-axis labels with word text
    ax.set_yticks(np.arange(len(translation)))  # Y-axis ticks for translated words
    hindi_font = FontProperties(fname='/kaggle/input/wordcloud-hindi-font/Nirmala.ttf')  # Assuming Hindi font path
    ax.set_yticklabels(translation, size=8, fontproperties=hindi_font)  # Y-axis labels with translated text (using Hindi font)
    ax.set_xlabel('Input Sequence', fontsize=10)
    ax.set_ylabel('Output Sequence', fontsize=10)

    ax.grid(color='lightgray', linestyle='-', linewidth=1)

  for ax in axes.flat[10:]:
    ax.axis('off')

  fig.tight_layout()
  plt.show()

params = {
    'language' : 'hin',
    'dataset_path' : arguments.dataset_path,
    'embedding_size': arguments.embedding_size,
    'hidden_size': arguments.hidden_size,
    'num_layers_enc': arguments.num_layers,
    'num_layers_dec': arguments.num_layers,
    'cell_type': arguments.cell_type,
    'dropout': arguments.dropout,
    'optimizer' : arguments.optimizer,
    'learning_rate': arguments.learning_rate,
    'batch_size': arguments.batch_size,
    'num_epochs': arguments.num_epochs,
    'teacher_fr' : arguments.teacher_fr,
    'length_penalty' : arguments.length_penalty,
    'beam_width': arguments.beam_width,
    'bi_dir' : arguments.bi_dir,
    'w_log' : arguments.w_log,
}

# pre precessing data and getting tensor representations
preprocessed_data = loadData(params)
tensors = create_tensor(preprocessed_data)

# defining Encoder, Decoder and Model
encoder = Encoder_Attention(params, preprocessed_data).to(device)
decoder = Decoder_Attention(params, preprocessed_data).to(device)
model = Seq2Seq_Attention(encoder, decoder, params, preprocessed_data).to(device)  

# defining Loss function and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = get_optim(model,params)

# Print total number of parameters in the model
if arguments.total_params == 1:
    total_parameters = get_total_parameters(model)
    print(f'Total Trainable Parameters: {total_parameters}')

# logging to wandb
if params['w_log']:
    # wandb.login(key = '3c81526a5ec348850a4c9d0f852f6631959307ed')
    wandb.login()
    wandb.init(project = arguments.wandb_project, entity = arguments.wandb_entity)
    wandb.run.name = (
        'check_c:' + params['cell_type'] +
        '_e:' + str(params['num_epochs']) +
        '_es:' + str(params['embedding_size']) +
        '_hs:' + str(params['hidden_size']) +
        '_nle:' + str(params['num_layers_enc']) +
        '_nld:' + str(params['num_layers_dec']) +
        '_o:' + params['optimizer'] +
        '_lr:' + str(params['learning_rate']) +
        '_bs:' + str(params['batch_size']) +
        '_tf:' + str(params['teacher_fr']) +
        '_lp:' + str(params['length_penalty']) +
        '_b:' + str(params['bi_dir']) +
        '_bw:' + str(params['beam_width'])
    )
# training the model
trained_model, _, _ = train(model, criterion, optimizer, preprocessed_data, tensors, params)

if params['w_log']:
    wandb.finish()

if arguments.evaluate_model == 1:
    print("Evaluating Model\n")
    _, _, _, results_test, accuracy_test_word_level, correct_pred_test = evaluate_model(trained_model, 'test', preprocessed_data, params, params['beam_width'], params['length_penalty'])
    print(f'Test Accuracy Word Level : {accuracy_test_word_level}, Correctly Predicted : {correct_pred_test}/{len(results_test)}')

# Plot the attention grid for the sampled word-translation pairs
# if arguments.plot_attention_heatmap == 1:
#     plot_attention_grid(trained_model)