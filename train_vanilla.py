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
import random
import wandb
import warnings
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL-Assignment-3')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m053')
parser.add_argument('-dp', '--dataset_path' , help='Specify path to dataset ../aksharantar_sampled.' , type=str, default='/kaggle/input/dl-ass3/aksharantar_sampled')
parser.add_argument('-ct', '--cell_type' , help='Type of cell.' , type=str,choices = ['RNN', 'LSTM', 'GRU'], default='LSTM')
parser.add_argument('-bd', '--bi_dir' , help="choices: ['True', 'False']",choices = [True, False], type=bool, default=True)
parser.add_argument('-nl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=2)
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

arguments = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadData(params):
    '''
    This function loads and preprocesses the data for machine translation.

    Args:
        params (dict): A dictionary containing parameters for data loading.
            - 'language': The language of the dataset (e.g., 'en', 'fr').
            - 'dataset_path': The path to the directory containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed data.
    '''
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
    '''
    This function creates PyTorch tensors from the preprocessed data.

    Args:
        preprocessed_data (dict): A dictionary containing the preprocessed data.

    Returns:
        dict: A dictionary containing PyTorch tensors for training, validation, and testing data.
    '''

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

class Encoder(nn.Module):
    '''
    This class defines the Encoder model for machine translation.

    Args:
        params (dict): A dictionary containing hyperparameters for the Encoder.
            - 'cell_type': The type of RNN cell to use (e.g., 'RNN', 'LSTM', 'GRU').
            - 'dropout': The dropout probability for regularization.
            - 'embedding_size': The size of the word embedding vectors.
            - 'hidden_size': The size of the hidden state in the RNN cell.
            - 'num_layers_enc': The number of layers to stack in the RNN.
            - 'bi_dir': Whether to use a bidirectional RNN (True) or unidirectional (False).
        preprocessed_data (dict): A dictionary containing the preprocessed data.
    '''

    def __init__(self, params, preprocessed_data):
        '''
        Initializes the Encoder model with hyperparameters and embedding layer.

        Args:
            params (dict): See Encoder class docstring for details.
            preprocessed_data (dict): See Encoder class docstring for details.
        '''

        super(Encoder, self).__init__()  # Call the superclass constructor (nn.Module)

        self.cell_type = params['cell_type']
        self.dropout = nn.Dropout(params['dropout'])

        # Embedding layer to map integer-encoded words to dense vectors
        self.embedding = nn.Embedding(len(preprocessed_data['input_index']), params['embedding_size'])

        # Choose the appropriate RNN cell based on the specified cell type
        if self.cell_type == 'RNN':
            self.cell = nn.RNN(params['embedding_size'], params['hidden_size'], params['num_layers_enc'], dropout=params['dropout'], bidirectional=params['bi_dir'])
        elif self.cell_type == 'LSTM':
            self.cell = nn.LSTM(params['embedding_size'], params['hidden_size'], params['num_layers_enc'], dropout=params['dropout'], bidirectional=params['bi_dir'])
        elif self.cell_type == 'GRU':
            self.cell = nn.GRU(params['embedding_size'], params['hidden_size'], params['num_layers_enc'], dropout=params['dropout'], bidirectional=params['bi_dir'])
        else:
            raise ValueError("Invalid type. Choose from 'RNN', 'LSTM', or 'GRU'.")

    def forward(self, x):
        '''
        Performs the forward pass through the Encoder model.

        Args:
            x (torch.Tensor): A tensor of integer-encoded word sequences.

        Returns:
            tuple:
                - hidden (torch.Tensor): The final hidden state(s) from the RNN.
                - cell (optional, torch.Tensor): The final cell state(s) for LSTMs (if applicable).
        '''

        # Pass the input sequence through the embedding layer
        drop_par = self.embedding(x)

        # Apply dropout for regularization
        drop_par = self.dropout(drop_par)

        # Forward pass through the RNN cell(s)
        if self.cell_type == 'LSTM':
            _, (hidden, cell) = self.cell(drop_par)
            return hidden, cell  # Return both hidden and cell states for LSTMs
        else:
            _, hidden = self.cell(drop_par)
            return hidden  # Return only the hidden state for RNNs and GRUs

class Decoder(nn.Module):
    '''
    This class defines the Decoder model for machine translation.

    Args:
        params (dict): A dictionary containing hyperparameters for the Decoder.
            - 'cell_type': The type of RNN cell to use (e.g., 'RNN', 'LSTM', 'GRU').
            - 'dropout': The dropout probability for regularization.
            - 'embedding_size': The size of the word embedding vectors.
            - 'hidden_size': The size of the hidden state in the RNN cell.
            - 'num_layers_dec': The number of layers to stack in the decoder RNN.
            - 'bi_dir': Whether to use a bidirectional RNN (True) or unidirectional (False).
        preprocessed_data (dict): A dictionary containing the preprocessed data.
    '''

    def __init__(self, params, preprocessed_data):
        '''
        Initializes the Decoder model with hyperparameters and embedding layer.

        Args:
            params (dict): See Decoder class docstring for details.
            preprocessed_data (dict): See Decoder class docstring for details.
        '''

        super(Decoder, self).__init__()  # Call the superclass constructor (nn.Module)

        self.cell_type = params['cell_type']
        self.dropout = nn.Dropout(params['dropout'])

        # Embedding layer to map integer-encoded words to dense vectors
        self.embedding = nn.Embedding(len(preprocessed_data['output_index']), params['embedding_size'])

        # Choose the appropriate RNN cell based on the specified cell type
        if self.cell_type == 'RNN':
            self.cell = nn.RNN(params['embedding_size'], params['hidden_size'], params['num_layers_dec'], dropout=params['dropout'], bidirectional=params['bi_dir'])
        elif self.cell_type == 'LSTM':
            self.cell = nn.LSTM(params['embedding_size'], params['hidden_size'], params['num_layers_dec'], dropout=params['dropout'], bidirectional=params['bi_dir'])
        elif self.cell_type == 'GRU':
            self.cell = nn.GRU(params['embedding_size'], params['hidden_size'], params['num_layers_dec'], dropout=params['dropout'], bidirectional=params['bi_dir'])
        else:
            raise ValueError("Invalid type. Choose from 'RNN', 'LSTM', or 'GRU'.")

        # Linear layer to map decoder output to vocabulary probabilities
        self.fc = nn.Linear(params['hidden_size'] * 2 if params['bi_dir'] == True else params['hidden_size'],
                             len(preprocessed_data['output_index']))

    def forward(self, x, hidden, cell=None):
        '''
        Performs the forward pass through the Decoder model.

        Args:
            x (torch.Tensor): A tensor of integer-encoded word (usually a single word).
            hidden (torch.Tensor): The hidden state(s) from the Encoder (optional for LSTMs).
            cell (torch.Tensor, optional): The cell state(s) from the Encoder (required for LSTMs).

        Returns:
            tuple:
                - predictions (torch.Tensor): A tensor of log softmax probabilities for the next word.
                - hidden (torch.Tensor): The final hidden state(s) from the decoder RNN.
                - cell (torch.Tensor, optional): The final cell state(s) for LSTMs (if applicable).
        '''

        # Pass the input word through the embedding layer
        embedding = self.embedding(x.unsqueeze(0))  # Add an extra dimension for batch processing

        # Apply dropout for regularization
        embedding = self.dropout(embedding)

        # Forward pass through the RNN cell(s)
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.cell(embedding, (hidden, cell))
        else:
            outputs, hidden = self.cell(embedding, hidden)

        # Pass the RNN output through a linear layer to get logits
        predictions = self.fc(outputs).squeeze(0)

        # Apply log softmax for probability distribution
        if self.cell_type == 'LSTM':
            predictions = F.log_softmax(predictions, dim=1)
            return predictions, hidden, cell  # Return all states for LSTMs
        else:
            return predictions, hidden  # Return only hidden state for RNNs and GR

class Seq2Seq(nn.Module):
    '''
    This class defines the Seq2Seq model for machine translation.

    Args:
        encoder (Encoder): An instance of the Encoder class.
        decoder (Decoder): An instance of the Decoder class.
        params (dict): A dictionary containing hyperparameters for the model.
            - 'cell_type': The type of RNN cell to use (e.g., 'RNN', 'LSTM', 'GRU').
            - 'teacher_fr': The teacher forcing ratio (probability of using ground truth in training).
        preprocessed_data (dict): A dictionary containing the preprocessed data.
    '''

    def __init__(self, encoder, decoder, params, preprocessed_data):
        '''
        Initializes the Seq2Seq model with the encoder, decoder, and hyperparameters.

        Args:
            encoder (Encoder): See Seq2Seq class docstring for details.
            decoder (Decoder): See Seq2Seq class docstring for details.
            params (dict): See Seq2Seq class docstring for details.
            preprocessed_data (dict): See Seq2Seq class docstring for details.
        '''

        super(Seq2Seq, self).__init__()  # Call the superclass constructor (nn.Module)

        self.cell_type = params['cell_type']
        self.encoder = encoder
        self.decoder = decoder

        # Length of the output vocabulary for calculating predictions
        self.output_index_len = len(preprocessed_data['output_index'])

        # Teacher forcing ratio (probability of using ground truth in training)
        self.tfr = params['teacher_fr']

    def forward(self, source, target):
        '''
        Performs the forward pass through the entire Seq2Seq model.

        Args:
            source (torch.Tensor): A tensor of integer-encoded source sequences.
            target (torch.Tensor): A tensor of integer-encoded target sequences (ground truth).

        Returns:
            torch.Tensor: A tensor containing the predicted target sequence log-softmax probabilities.
        '''

        batch_size, target_len = source.shape[1], target.shape[0]  # Get batch size and target sequence length

        # Initialize the predicted sequence with the first word from the target (start token)
        x = target[0]

        # Create an empty tensor to store predicted sequence probabilities
        outputs = torch.zeros(target_len, batch_size, self.output_index_len).to(device)

        # Get the hidden state(s) from the encoder
        if self.cell_type == 'LSTM':
            hidden, cell = self.encoder(source)
        else:
            hidden = self.encoder(source)

        # Loop through each timestep in the target sequence
        for t in range(1, target_len):
            # Pass the previous predicted word and hidden state(s) through the decoder
            if self.cell_type == 'LSTM':
                output, hidden, cell = self.decoder(x, hidden, cell)
            else:
                output, hidden = self.decoder(x, hidden, None)

            # Store the predicted word probabilities in the outputs tensor
            outputs[t], best_guess = output, output.argmax(1)  # Get the word with highest probability

            # Teacher forcing: Choose predicted word or ground truth word based on a random value
            x = best_guess if random.random() >= self.tfr else target[t]

        return outputs

def get_optim(model, params):
    '''
    This function creates an optimizer object based on the specified parameters.

    Args:
        model (nn.Module): The Seq2Seq model instance.
        params (dict): A dictionary containing hyperparameters for the optimizer.
            - 'optimizer' (str): The name of the optimizer to use (e.g., 'sgd', 'adam', 'rmsprop', 'adagrad').
            - 'learning_rate' (float): The learning rate for the optimizer.

    Returns:
        optim.Optimizer: An optimizer object for training the model.
    '''

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
  '''
  This function calculates the total number of trainable parameters in a PyTorch model.

  Args:
      model (nn.Module): The PyTorch model to analyze.

  Returns:
      int: The total number of trainable parameters in the model.
  '''

  # Filter only trainable parameters
  total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

  return total_params

def beam_search(model, word, preprocessed_data, bw, lp, ct):
    '''
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
    '''
    data = np.zeros((preprocessed_data['max_len']+1, 1), dtype=np.int32)
    for idx, char in enumerate(word):
        data[idx, 0] = preprocessed_data['input_index'][char]
    data[idx + 1, 0] = preprocessed_data['input_index'][preprocessed_data['EOS']]
    data = torch.tensor(data, dtype=torch.int32).to(device)
    with torch.no_grad():
        if ct == 'LSTM':
           hidden, cell = model.encoder(data)
        else:
           hidden = model.encoder(data)
    output_start = preprocessed_data['output_index'][preprocessed_data['SOS']]
    out_reshape, hidden_par  = np.array(output_start).reshape(1,), hidden.unsqueeze(0)
    initial_sequence = torch.tensor(out_reshape).to(device)
    beam = [(0.0, initial_sequence, hidden_par)]
    for i in range(len(preprocessed_data['output_index'])):
        candidates = []
        for score, seq, hidden in beam:
            if seq[-1].item() == preprocessed_data['output_index'][preprocessed_data['EOS']]:
                candidates.append((score, seq, hidden))
                continue
            reshape_last, hdn = np.array(seq[-1].item()).reshape(1, ), hidden.squeeze(0)
            x = torch.tensor(reshape_last).to(device)
            if ct == 'LSTM':
                output, hidden, cell = model.decoder(x, hdn, cell)
            else:
                output, hidden = model.decoder(x, hdn, None)
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
    '''
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
    '''
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
                    ans = beam_search(model, word, preprocessed_data, params['beam_width'], params['length_penalty'], params['cell_type'])
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
  '''
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
  '''

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
  with tqdm(total=total_words, desc='testing') as pbar:
    for word, translation in zip(preprocessed_data[data_words], preprocessed_data[data_translations]):
      # Perform beam search to get the predicted translation
      ans = beam_search(trained_model, word, preprocessed_data, bw, lp, params['cell_type'])

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
  '''
  This function generates a predicted translation for a given source word sequence.

  Args:
      model (nn.Module): The trained Seq2Seq model instance.
      word (str): The source word sequence to translate.
      preprocessed_data (dict): A dictionary containing the preprocessed data.
      params (dict): A dictionary containing hyperparameters for the model.
          - 'cell_type' (str): The type of RNN cell used in the model.

  Returns:
      str: The predicted translated word sequence.
  '''

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
      hidden, cell = model.encoder(data)
    else:
      hidden = model.encoder(data)

    # Start token (SOS) for the decoder
    x = torch.tensor([preprocessed_data['output_index'][preprocessed_data['SOS']]]).to(device)

    # Greedy search for predicted translation
    for t in range(1, len(preprocessed_data['output_index'])):
      # Get the decoder output and update hidden state(s)
      if params['cell_type'] == 'LSTM':
        output, hidden, cell = model.decoder(x, hidden, cell)
      else:
        output, hidden = model.decoder(x, hidden, None)

      # Convert the decoder output to the predicted character
      character = preprocessed_data['output_index_rev'][output.argmax(1).item()]

      # Stop prediction if EOS is encountered
      if character != preprocessed_data['EOS']:
        pred = pred + character
      else:
        break

      # Use the predicted character as the next input to the decoder
      x = torch.tensor([output.argmax(1)]).to(device)

  return pred

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
encoder = Encoder(params, preprocessed_data).to(device)
decoder = Decoder(params, preprocessed_data).to(device)
model = Seq2Seq(encoder, decoder, params, preprocessed_data).to(device)  

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