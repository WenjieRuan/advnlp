import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchtext import data, datasets
from model import EncoderRNN, DecoderRNN, VAE


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=8,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=1.0,
                   help='initial learning rate')
    return p.parse_args()

def evaluate(model, val_iter, vocab_size, kld_weight, use_cuda):
    model.eval()
    total_loss = 0
    for b, batch in enumerate(val_iter):
        x, y = batch.text, batch.label
        if use_cuda:
            x.cuda(); y.cuda()

        m, l, z, decoded = model(x, None)
        recon_loss = F.cross_entropy(decoded.view(-1, vocab_size), x.contiguous().view(-1))
        kl_loss = -0.5 * (2 * l - torch.pow(m, 2) - torch.pow(torch.exp(l), 2) + 1)
        kl_loss = torch.clamp(kl_loss.mean(), min=0.2).squeeze()
        loss = recon_loss + kl_loss * kld_weight

        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train(epoch, model, optimizer, train_iter, vocab_size,
          kld_weight, temperature, grad_clip, use_cuda, TEXT):
    model.train()
    total_loss = 0
    for b, batch in enumerate(train_iter):
        x, y = batch.text, batch.label
        if use_cuda:
            x.cuda(); y.cuda()
        optimizer.zero_grad()

        m, l, z, decoded = model(x, temperature)
        recon_loss = F.cross_entropy(decoded.view(-1, vocab_size), x.contiguous().view(-1))
        kl_loss = -0.5 * (2 * l - torch.pow(m, 2) - torch.pow(torch.exp(l), 2) + 1)
        kl_loss = torch.clamp(kl_loss.mean(), min=0.2).squeeze()
        loss = recon_loss + kl_loss * kld_weight

        if epoch > 1 and kld_weight < 0.1:
            kld_weight += 0.000002
        if temperature > 0.5:
            temperature -= 0.000002

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()

        sys.stdout.write('\r[%d] [loss] %.4f - recon_loss: %.4f - kl_loss: %.4f - kld-weight: %.4f - temp: %4f'
                         % (b, loss.data[0], recon_loss.data[0], kl_loss.data[0], kld_weight, temperature))
        total_loss += loss.data[0]

        if b % 200 == 0 and b != 0:
            total_loss = total_loss / 200
            print("\n[avg loss] - ", total_loss)
            _, sample = decoded.data.cpu()[:,0,:].topk(1)
            print("[ORI]: ", " ".join([TEXT.vocab.itos[i] for i in x.data[:,0]]))
            print("[GEN]: ", " ".join([TEXT.vocab.itos[i] for i in sample.squeeze()]))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 300
    embed_size = 50
    kld_weight = 0.05
    temperature = 0.9
    use_cuda = torch.cuda.is_available()

    print("[!] preparing dataset...")
    TEXT = data.Field(lower=True, fix_length=30)
    LABEL = data.Field(sequential=False)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data, max_size=250000)
    LABEL.build_vocab(train_data)
    train_iter, test_iter = data.BucketIterator.splits(
            (train_data, test_data), batch_size=args.batch_size, repeat=False)
    vocab_size = len(TEXT.vocab) + 2

    print("[!] Instantiating models...")
    encoder = EncoderRNN(vocab_size, hidden_size, embed_size,
                         n_layers=2, dropout=0.5, use_cuda=use_cuda)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size,
                         n_layers=2, dropout=0.5, use_cuda=use_cuda)
    vae = VAE(encoder, decoder)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    if use_cuda:
        print("[!] Using CUDA...")
        vae.cuda()

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, vae, optimizer, train_iter, vocab_size,
              kld_weight, temperature, args.grad_clip, use_cuda, TEXT)
        val_loss = evaluate(vae, test_iter, vocab_size, kld_weight, use_cuda)
        print("[Epoch: %d] val_loss:%5.3f | val_pp:%5.2fS"
               % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(vae.state_dict(), './snapshot/vae_{}.pt'.format(e))
            best_val_loss = val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
