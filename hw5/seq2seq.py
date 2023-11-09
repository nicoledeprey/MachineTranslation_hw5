#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division
import datetime
import argparse
import logging
import random
import time
import math
from io import open
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
import torch.nn.init as init
from math import log
logging.basicConfig(level=logging.DEBUG,
 format='%(asctime)s %(levelname)s %(message)s')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
 """ This class handles the mapping between the words and their indicies
 """
 def __init__(self, lang_code):
 self.lang_code = lang_code
 self.word2index = {}
 self.word2count = {}
 self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
 self.n_words = 2 # Count SOS and EOS

 def add_sentence(self, sentence):
 for word in sentence.split(' '):
 self._add_word(word)

 def _add_word(self, word):
 if word not in self.word2index:
 self.word2index[word] = self.n_words
 self.word2count[word] = 1
 self.index2word[self.n_words] = word
 self.n_words += 1
 else:
 self.word2count[word] += 1


######################################################################


def split_lines(input_file):
 """split a file like:
 first src sentence|||first tgt sentence
 second src sentence|||second tgt sentence
 into a list of things like
 [("first src sentence", "first tgt sentence"), 
 ("second src sentence", "second tgt sentence")]
 """
 logging.info("Reading lines of %s...", input_file)
 # Read the file and split into lines
 lines = open(input_file, encoding='utf-8').read().strip().split('\n')
 # Split every line into pairs
 pairs = [l.split('|||') for l in lines]
 return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
 """ Creates the vocabs for each of the langues based on the training corpus.
 """
 src_vocab = Vocab(src_lang_code)
 tgt_vocab = Vocab(tgt_lang_code)

 train_pairs = split_lines(train_file)

 for pair in train_pairs:
 src_vocab.add_sentence(pair[0])
 tgt_vocab.add_sentence(pair[1])

 logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
 logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

 return src_vocab, tgt_vocab, train_pairs

######################################################################

def indexesFromSentence(lang, sentence):
 indexes = []
 for word in sentence.split():
 try:
 indexes.append(lang.word2index[word])
 except KeyError:
 pass
 indexes.append(EOS_index)
 return indexes

def tensorFromSentence(lang, sentence):
 indexes = indexesFromSentence(lang, sentence)
 indexes.append(EOS_index)
 return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)



def tensor_from_sentence(vocab, sentence):
 """creates a tensor from a raw sentence
 """
 indexes = []
 for word in sentence.split():
 try:
 indexes.append(vocab.word2index[word])
 except KeyError:
 pass
 indexes.append(EOS_index)
 return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
 """creates a tensor from a raw sentence pair
 """
 input_tensor = tensor_from_sentence(src_vocab, pair[0])
 target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
 return input_tensor, target_tensor

class EncoderRNN(nn.Module):
 def __init__(self, input_size, hidden_size, dropout_p=0.1):
 super(EncoderRNN, self).__init__()
 self.hidden_size = hidden_size

 self.embedding = nn.Embedding(input_size, hidden_size)
 # self.gru = GRU(hidden_size, hidden_size, batch_first=True)
 ## changing GRU to nn.Gru 
 self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
 self.dropout = nn.Dropout(dropout_p)

 def forward(self, input):
 embedded = self.dropout(self.embedding(input))
 output, hidden = self.gru(embedded)
 return output, hidden

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
 decoder_optimizer,
 criterion):

 total_loss = 0
 for data in dataloader:
 input_tensor, target_tensor = data

 
 encoder_optimizer.zero_grad()
 decoder_optimizer.zero_grad()
 
 encoder_outputs, encoder_hidden = encoder(input_tensor)
 
 decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
 
 loss = criterion(
 decoder_outputs.view(-1, decoder_outputs.size(-1)),
 target_tensor.view(-1)

 )
 
 loss.backward()

 encoder_optimizer.step()
 decoder_optimizer.step()
 

 total_loss += loss.item()

 return total_loss / len(dataloader)



class Beam_Search:
 def __init__(self, size):
 self.size = size
 self.candidate_sequences = [[]]
 self.sequence_scores = [0]

 def addcandidates(self, token_probabilities):
 all_sequences = []
 for seq_index, sequence in enumerate(self.candidate_sequences):
 for token_index, token_prob in enumerate(token_probabilities):
 extended_sequence = sequence + [token_index]
 sequence_score = self.sequence_scores[seq_index] - log(token_prob)
 all_sequences.append((extended_sequence, sequence_score))
 
 sorted_sequences = sorted(all_sequences, key=lambda tup: tup[1])
 
 self.candidate_sequences = [sequence[0] for sequence in sorted_sequences[:self.size]]
 self.sequence_scores = [sequence[1] for sequence in sorted_sequences[:self.size]]

 def get_most_probable_sequence(self):
 return self.candidate_sequences[0]


def translatebeam(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
 encoder.eval()
 decoder.eval()
 beam_size=5 ## need to experiment with it 
 with torch.no_grad():
 input_tensor = tensor_from_sentence(src_vocab, sentence)
 encoder_outputs, encoder_hidden = encoder(input_tensor)
 decoder_input = torch.tensor([[SOS_index]], device=device).long()
 decoder_hidden = encoder_hidden

 beam = Beam_Search(beam_size)
 
 for di in range(max_length):
 if di == 0:
 ## get initial probabilities from decoder
 decoder_output, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)
 ## get top 5 indices 
 _, top_indices = decoder_output.topk(beam_size)
 ## store those indices in the list 
 beam.current_sequences = [[idx.item()] for idx in top_indices[0]]
 else:
 all_decoder_outputs = []
 for idx in beam.current_sequences:
 decoder_input = torch.tensor([idx[-1]], device=device).unsqueeze(0).long()
 decoder_output, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)
 all_decoder_outputs.append(decoder_output)
 probs = F.softmax(torch.cat(all_decoder_outputs, dim=1), dim=-1)
 beam.addcandidates(probs[0])

 if beam.get_best()[-1] == EOS_index:
 break

 decoded_words = [tgt_vocab.index2word[idx] for idx in beam.get_most_probable_sequence()]
 return decoded_words




def evaluate(encoder, decoder, sentence, input_lang, output_lang):
 with torch.no_grad():
 input_tensor =tensorFromSentence(input_lang, sentence)
 

 encoder_outputs, encoder_hidden = encoder(input_tensor)
 decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
 
 _, topi = decoder_outputs.topk(1)
 
 decoded_ids = topi.squeeze()

 decoded_words = []
 
 for idx in decoded_ids:
 
 if idx.item() == EOS_index:
 decoded_words.append('<EOS>')
 break
 decoded_words.append(output_lang.index2word[idx.item()])
 return decoded_words, decoder_attn


def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
 output_sentences = []
 for pair in pairs[:max_num_sentences]:
 output_words, attentions = evaluate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
 output_sentence = ' '.join(output_words)
 output_sentences.append(output_sentence)
 return output_sentences




def evaluateAndShowAttention(encoder, decoder, input_sentence, input_lang, output_lang):
 output_words, attentions = translate_and_show_attention(input_sentence, encoder, decoder, input_lang, output_lang)
 print('input =', input_sentence)
 print('output =', ' '.join(output_words))
 

def show_attention(input_sentence, output_words, attentions, fig, ax):
 """visualize the attention mechanism. And save it to a file.
 Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
 You plots should include axis labels and a legend.
 you may want to use matplotlib.
 """

 "*** YOUR CODE HERE ***"

 ax.matshow(attentions.numpy(), cmap='gray')
 ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
 ax.set_yticklabels([''] + output_words)
 plt.show()


 # raise NotImplementedError
 


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, fig, ax):
 output_words, attentions = translatebeam(encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
 print('input =', input_sentence)
 print('output =', ' '.join(output_words))
 show_attention(input_sentence, output_words, attentions, fig,ax)


def clean(strx):
 """
 input: string with bpe, EOS
 output: list without bpe, EOS
 """
 return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


def asMinutes(s):
 m = math.floor(s / 60)
 s -= m * 60
 return '%dm %ds' % (m, s)

def timeSince(since, percent):
 now = time.time()
 s = now - since
 es = s / (percent)
 rs = es - s
 return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
 print_every=100, plot_every=100):
 start = time.time()
 plot_losses = []
 print_loss_total = 0 # Reset every print_every
 plot_loss_total = 0 # Reset every plot_every

 encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
 decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
 criterion = nn.NLLLoss()

 for epoch in range(1, n_epochs + 1):
 loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
 print_loss_total += loss
 print('loss:{}'.format(loss))

 if epoch % print_every == 0:
 print_loss_avg = print_loss_total / print_every
 print_loss_total = 0
 print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
 epoch, epoch / n_epochs * 100, print_loss_avg))
 
 return

def evaluateRandomly(encoder, decoder, test_pairs, input_lang, output_lang, n=10):
 
 pair = random.choice(test_pairs)
 print('pair:{}'.format(pair))
 # print('test pairs:{}'.format(test_pairs))
 print('>', pair[0])
 # print('=', pair[1])
 output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
 output_sentence = ' '.join(output_words)
 print('<', output_sentence)
 print('')



def showAttention(input_sentence, output_words, attentions):
 fig = plt.figure()
 ax = fig.add_subplot(111)
 
 # Limiting the displayed attentions
 attentions_to_display = attentions[:, :len(input_sentence.split(' ')) + 1]
 
 cax = ax.matshow(attentions_to_display.cpu().numpy(), cmap='bone')
 fig.colorbar(cax)

 ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
 ax.set_yticklabels([''] + output_words)

 # Show label at every tick
 ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
 ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
 timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
 
 plt.savefig(f'heatmap_{input_sentence}.png', format='png')


def translate_and_show_attention(input_sentence, encoder, decoder, input_lang, output_lang):
 output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang,output_lang )

 print('input =', input_sentence)
 print('output =', ' '.join(output_words))
 showAttention(input_sentence, output_words, attentions[0, :len(output_words), :])

######################################################################

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument('--hidden_size', default=256, type=int,
 help='hidden size of encoder/decoder, also word vector size')
 ap.add_argument('--n_iters', default=10, type=int,
 help='total number of examples to train on')
 ap.add_argument('--print_every', default=5000, type=int,
 help='print loss info every this many training examples')
 ap.add_argument('--checkpoint_every', default=10000, type=int,
 help='write out checkpoint every this many training examples')
 ap.add_argument('--initial_learning_rate', default=0.001, type=int,
 help='initial learning rate')
 ap.add_argument('--src_lang', default='fr',
 help='Source (input) language code, e.g. "fr"')
 ap.add_argument('--tgt_lang', default='en',
 help='Source (input) language code, e.g. "en"')
 ap.add_argument('--train_file', default='data/fren.train.bpe',
 help='training file. each line should have a source sentence,' +
 'followed by "|||", followed by a target sentence')
 ap.add_argument('--dev_file', default='data/fren.dev.bpe',
 help='dev file. each line should have a source sentence,' +
 'followed by "|||", followed by a target sentence')
 ap.add_argument('--test_file', default='data/fren.test.bpe',
 help='test file. each line should have a source sentence,' +
 'followed by "|||", followed by a target sentence' +
 ' (for test, target is ignored)')
 ap.add_argument('--out_file', default='translations',
 help='output file for test translations')
 ap.add_argument('--load_checkpoint', nargs=1,
 help='checkpoint file to start from')

 args = ap.parse_args()



 if args.load_checkpoint is not None:
 state = torch.load(args.load_checkpoint[0])
 iter_num = state['iter_num']
 src_vocab = state['src_vocab']
 tgt_vocab = state['tgt_vocab']
 else:
 iter_num = 0
 src_vocab, tgt_vocab, train_pairs = make_vocabs(args.src_lang,
 args.tgt_lang,
 args.train_file)
 
 print(f"tgt_vocab.n_words = {tgt_vocab.n_words}")
 

 def batching(batch_size):
 input_lang, output_lang, pairs = make_vocabs(args.src_lang,
 args.tgt_lang,
 args.train_file)

 n = len(pairs)
 input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
 target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

 for idx, (inp, tgt) in enumerate(pairs):
 inp_ids = indexesFromSentence(input_lang, inp)
 tgt_ids = indexesFromSentence(output_lang, tgt)
 inp_ids.append(EOS_index)
 tgt_ids.append(EOS_index)
 input_ids[idx, :len(inp_ids)] = inp_ids
 target_ids[idx, :len(tgt_ids)] = tgt_ids
 

 
 
 train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
 torch.LongTensor(target_ids).to(device))

 train_sampler = RandomSampler(train_data)
 train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
 return input_lang, output_lang, train_dataloader
 

 src_vocab, tgt_vocab, train_dataloader = batching(32)

 encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
 decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

 
 train_pairs = split_lines(args.train_file)
 dev_pairs = split_lines(args.dev_file)
 test_pairs = split_lines(args.test_file)

 params = list(encoder.parameters()) + list(decoder.parameters()) # .parameters() returns generator
 optimizer = optim.Adam(params, lr=args.initial_learning_rate)
 criterion = nn.NLLLoss()
 
 start = time.time()
 print_loss_total = 0 # Reset every args.print_every

 while iter_num < args.n_iters:
 iter_num += 1
 if iter_num % 100 == 0:
 print(f"now at iter {iter_num}")
 training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
 input_tensor = training_pair[0]
 target_tensor = training_pair[1]
 loss = train(train_dataloader, encoder, decoder, 100)
 print_loss_total += loss

 if iter_num % args.checkpoint_every == 0:
 state = {'iter_num': iter_num,
 'enc_state': encoder.state_dict(),
 'dec_state': decoder.state_dict(),
 'opt_state': optimizer.state_dict(),
 'src_vocab': src_vocab,
 'tgt_vocab': tgt_vocab,
 }
 filename = 'state_%010d.pt' % iter_num
 torch.save(state, filename)
 logging.debug('wrote checkpoint to %s', filename)

 if iter_num % args.print_every == 0:
 print_loss_avg = print_loss_total / args.print_every
 print_loss_total = 0
 logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
 time.time() - start,
 iter_num,
 iter_num / args.n_iters * 100,
 print_loss_avg)
 # translate from the dev set
 translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

 references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
 candidates = [clean(sent).split() for sent in translated_sentences]
 dev_bleu = corpus_bleu(references, candidates)
 logging.info('Dev BLEU score: %.2f', dev_bleu)
 
 print("Translating the test set")
 # translate test set and write to file
 translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
 with open(args.out_file, 'wt', encoding='utf-8') as outf:
 for sent in translated_sentences:
 outf.write(clean(sent) + '\n')


if __name__ == '__main__':
 main()
