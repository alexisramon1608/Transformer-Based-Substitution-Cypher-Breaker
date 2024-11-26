import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess_dataset import preprocess_text
import bitsandbytes as bnb
from invariants import get_data_pairs
from french_dataset import get_full_dataset
from torch.utils.data import Dataset, DataLoader
from cipher_8bit import load_or_save_symbols, substitution_cipher, encode_text_with_indices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQUENCE_LENGTH = 512
PAD_IDX_SRC, PAD_IDX_TGT, BOS_IDX_SRC, BOS_IDX_TGT = 698, 257, 699, 258

training_pairs = get_data_pairs(get_full_dataset())
def create_mask(src):
    src_seq_len = src.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX_SRC).transpose(0, 1)
    return src_padding_mask
class CipherDataset(Dataset):
    def __init__(self, pairs):
        self.encodings = []
        self.symbol_indices = []
        bulk_dataset = pairs
        for entry in bulk_dataset:

            if len(entry[0]) < MAX_SEQUENCE_LENGTH:
                self.encodings.append([BOS_IDX_SRC] + entry[0] + [PAD_IDX_SRC] * (MAX_SEQUENCE_LENGTH - len(entry[0]) - 1))
            elif len(entry[0]) > MAX_SEQUENCE_LENGTH:
                self.encodings.append([BOS_IDX_SRC] + entry[0][:MAX_SEQUENCE_LENGTH - 1])
            else:
                self.encodings.append([BOS_IDX_SRC] + entry[0][:-1])

            if len(entry[1]) < MAX_SEQUENCE_LENGTH:
                self.symbol_indices.append([BOS_IDX_TGT] + entry[1] + [PAD_IDX_TGT] * (MAX_SEQUENCE_LENGTH - len(entry[1]) - 1))
            elif len(entry[1]) > MAX_SEQUENCE_LENGTH:
                self.symbol_indices.append([BOS_IDX_TGT] + entry[1][:MAX_SEQUENCE_LENGTH - 1])
            else:
                self.symbol_indices.append([BOS_IDX_TGT] + entry[1][:-1])

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx]), torch.tensor(self.symbol_indices[idx])


# Wrap data in the simplest possible way to enable PyTorch data fetching
# https://pytorch.org/docs/stable/data.html

BATCH_SIZE = 64
TRAIN_FRAC = 0.97

dataset = CipherDataset(training_pairs)
N = len(training_pairs)
print(N)
S  = 512
C = 700
# Split into train and val
num_train = int(N * TRAIN_FRAC)
num_val = N - num_train
data_train, data_val = torch.utils.data.random_split(dataset, (num_train, num_val))

dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE)
dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE)

# Sample batch
x, y = next(iter(dataloader_train))
class PositionalEncoding(nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding.
    From PyTorch docs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask. From PyTorch docs."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Transformer(nn.Module):
    """
    Classic Transformer that both encodes and decodes.
    
    Prediction-time inference is done greedily.

    NOTE: start token is hard-coded to be 0, end token to be 1. If changing, update predict() accordingly.
    """

    def __init__(self, num_classes: int, max_output_length: int, dim: int = 768):
        super().__init__()

        # Parameters
        self.dim = dim
        self.max_output_length = max_output_length
        nhead = 16
        num_layers = 8
        dim_feedforward = dim

        # Encoder part
        self.x_embedding = nn.Embedding(700, dim)
        self.y_embedding = nn.Embedding(259, dim)
        self.pos_encoder = PositionalEncoding(d_model=self.dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )

        # Decoder part

        self.y_mask = generate_square_subsequent_mask(self.max_output_length)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc = nn.Linear(self.dim, 259)

        # It is empirically important to initialize weights properly
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.x_embedding.weight.data.uniform_(-initrange, initrange)
        self.y_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
      
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        x_pad_mask= create_mask(x)

        encoded_x = self.encode(x, x_pad_mask)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def encode(self, x: torch.Tensor, x_pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (Sx, B, E) embedding
        """
        x = x.permute(1, 0)  # (Sx, B, E)
        x = self.x_embedding(x) * math.sqrt(self.dim)  # (Sx, B, E)
        x = self.pos_encoder(x)  # (Sx, B, E)
        x = self.transformer_encoder(x, None, x_pad_mask.transpose(0,1))  # (Sx, B, E)
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """
        Input
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (Sy, B, C) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.y_embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.

        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        x_pad_mask = create_mask(x)

        encoded_x = self.encode(x, x_pad_mask)
        
        output_tokens = (torch.ones((x.shape[0], self.max_output_length))).type_as(x).long() # (B, max_length)
        output_tokens[:, 0] = BOS_IDX_TGT  # Set start token
        for Sy in range(1, self.max_output_length):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(y, encoded_x)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[-1:]  # Set the last output token
        return output_tokens


class LitModel(pl.LightningModule):
    """Simple PyTorch-Lightning model to train our Transformer."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')

    def training_step(self, batch, batch_ind):
        x, y = batch
        # Teacher forcing: model gets input up to the last character,
        # while ground truth is from the second character onward.
        logits = self.model(x, y[:, :-1])
        loss = self.loss(logits, y[:, 1:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_ind):
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)
    
    def configure_optimizers(self):
        return bnb.optim.AdamW8bit(self.parameters(), lr=0.0003, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)


model = Transformer(num_classes=700, max_output_length=y.shape[1])
lit_model = LitModel(model)
early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss')
trainer = pl.Trainer(max_epochs=100)
trainer.fit(lit_model, dataloader_train, dataloader_val)

# We can see that the decoding works correctly
torch.save(model.state_dict(), "cipher.pth")

x, y = next(iter(dataloader_val))
print('Input:', x[:1])
pred = lit_model.model.predict(x[:1])
print('Truth/Pred:')
tokens = torch.cat((y[:1], pred)).cpu().numpy()
symbols = load_or_save_symbols([])
print(tokens)
print(''.join([symbols[x] if x < 256 else "#" for x in tokens[0][1:]]))
print(''.join([symbols[x] if x < 256 else "#" for x in tokens[1][1:]]))

