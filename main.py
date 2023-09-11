import torch
import boto3
from typing import List
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
from torch.nn import Transformer
from torch import Tensor
import math
import pandas as pd
from io import BytesIO

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def transfer(model: torch.nn.Module, src_sentence: str, text_transform, sentiment, vocab_transform):
    model.eval()
    src = text_transform[sentiment](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[sentiment].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

def get_access_keys(filename):
    info = pd.read_csv(filename, header=None).values.tolist()
    access_key = info[0][0].split('=', 1)[1]
    secret_access_key = info[1][0].split('=', 1)[1]
    return access_key, secret_access_key

def get_file_from_s3(bucket_name, access_key, secret_acces, filename, is_model):
    client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_acces)
    obj = BytesIO(client.get_object(Bucket=bucket_name, Key=filename)['Body'].read())

    if is_model:
        if torch.cuda.is_available():
            return torch.load(obj)
        else:
            return torch.load(obj, map_location=torch.device('cpu'))
    else:
        return torch.load(obj)


if __name__=="__main__":
    access_key, secret_access_key = get_access_keys('rootkey.csv')
    s3_bucket = 'ee641-model-bucket'

    SENTIMENT_POS = 'pos'
    SENTIMENT_NEG = 'neg'

    tokenizer = get_tokenizer('basic_english')

    vocab_transform_pos = get_file_from_s3(s3_bucket, access_key, secret_access_key, 'pos_vocab.pth', False)
    vocab_transform_neg = get_file_from_s3(s3_bucket, access_key, secret_access_key, 'neg_vocab.pth', False)
    vocab_transform = {SENTIMENT_POS: vocab_transform_pos, SENTIMENT_NEG: vocab_transform_neg}

    torch.manual_seed(0)

    POS_VOCAB_SIZE = len(vocab_transform[SENTIMENT_POS])
    NEG_VOCAB_SIZE = len(vocab_transform[SENTIMENT_NEG])
    EMB_SIZE = 128
    NHEAD = 8
    FFN_HID_DIM = 128
    BATCH_SIZE = 2
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2

    text_transform = {}
    for ln in [SENTIMENT_POS, SENTIMENT_NEG]:
        text_transform[ln] = sequential_transforms(tokenizer,
                                                vocab_transform[ln],
                                                tensor_transform)

    inp = int(input('Enter 1 for positive to negative style transfer or 2 for negative to positive style transfer: '))

    if inp == 1:
        model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, POS_VOCAB_SIZE, NEG_VOCAB_SIZE, FFN_HID_DIM)
        loaded_file = get_file_from_s3(s3_bucket, access_key, secret_access_key, 'G_POS_TO_NEG.pkl', True)
        model.load_state_dict(loaded_file)
        
        input_review = input('Input a positive sentiment statement: ')
        sentiment = SENTIMENT_POS
    else:
        model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, NEG_VOCAB_SIZE, POS_VOCAB_SIZE, FFN_HID_DIM)
        loaded_file = get_file_from_s3(s3_bucket, access_key, secret_access_key, 'G_NEG_to_POS.pkl', True)
        model.load_state_dict(loaded_file)
        
        input_review = input('Input a negative sentiment statement: ')
        sentiment = SENTIMENT_NEG

    transferred_review = transfer(model, input_review, text_transform, sentiment, vocab_transform)
    print(f'The transferred review is:\n{transferred_review}')