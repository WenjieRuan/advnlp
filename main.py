import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torchtext import data, datasets
from language_model import RNNModel
from classifiers import ConvText


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=2,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=0.25,
                   help='initial learning rate')
    return p.parse_args()


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(classifier, generator, val_iter, beta, TEXT, use_cuda=True, args=None):
    classifier.eval()
    generator.eval()
    avg_loss, avg_loss_x, avg_loss_y = 0, 0, 0
    corrects_t, corrects_f = 0, 0
    vocab_size = len(TEXT.vocab)

    hidden = generator.init_hidden(val_iter.batch_size)
    for b, batch in enumerate(val_iter):
        text, label = batch.text, batch.label
        label.data.sub_(1)
        target = Variable(-1 * (label.data.sub(1)))
        if use_cuda:
            text.cuda(), label.cuda(), target.cuda()
        y_t = classifier(text)
        generated, hidden = generator(text, hidden)
        _, topi = generated.data.topk(1)
        gen_text = Variable(topi.squeeze())
        hidden = repackage_hidden(hidden)
        y_f = classifier(gen_text)

        loss_x = F.cross_entropy(generated.view(-1, vocab_size), text.view(-1))
        loss_y = F.cross_entropy(y_f, target, size_average=False)
        loss = beta * loss_x + loss_y
        avg_loss += loss.data[0]
        avg_loss_x += loss_x.data[0]
        avg_loss_y += loss_y.data[0]

        corrects_t += (torch.max(y_t, 1)[1].view(label.size()).data == label.data).sum()
        corrects_f += (torch.max(y_f, 1)[1].view(label.size()).data == label.data).sum()
    avg_loss = avg_loss / len(val_iter)
    avg_loss_x = avg_loss_x / len(val_iter)
    avg_loss_y = avg_loss_y / len(val_iter)
    accuracy_t = 100.0 * corrects_t / len(val_iter.dataset)
    accuracy_f = 100.0 * corrects_f / len(val_iter.dataset)
    return avg_loss, avg_loss_x, avg_loss_y, accuracy_t, accuracy_f


def train(classifier, generator, generator_op, train_iter, beta, TEXT, use_cuda=True, args=None):
    classifier.eval()
    generator.train()
    total_loss, total_loss_x, total_loss_y = 0, 0, 0
    corrects_t, corrects_f = 0, 0
    vocab_size = len(TEXT.vocab)

    hidden = generator.init_hidden(train_iter.batch_size)
    for b, batch in enumerate(train_iter):
        text, label = batch.text, batch.label
        label.data.sub_(1)
        target = Variable(-1 * (label.data.sub(1)))
        if use_cuda:
            text.cuda(), label.cuda(), target.cuda()
        generator_op.zero_grad()
        y_t = classifier(text)
        generated, hidden = generator(text, hidden)
        _, topi = generated.data.topk(1)
        gen_text = Variable(topi.squeeze())
        hidden = repackage_hidden(hidden)
        y_f = classifier(gen_text)

        loss_x = F.cross_entropy(generated.view(-1, vocab_size), text.view(-1))
        loss_y = F.cross_entropy(y_f, target)
        loss = beta * loss_x + loss_y


        loss.backward()
        nn.utils.clip_grad_norm(generator.parameters(), args.grad_clip)
        generator_op.step()

        corrects_t += (torch.max(y_t, 1)[1].view(label.size()).data == label.data).sum()
        corrects_f += (torch.max(y_f, 1)[1].view(label.size()).data == label.data).sum()
        total_loss += loss.data[0]
        total_loss_x += loss_x.data[0]
        total_loss_y += loss_y.data[0]
        if b % 500 == 0 and b != 0:
            total_loss, total_loss_x, total_loss_y = total_loss/500, total_loss_x/500, total_loss_y/500
            accuracy_t = 100.0 * corrects_t / (500 * train_iter.batch_size)
            accuracy_f = 100.0 * corrects_f / (500 * train_iter.batch_size)
            log = '[{}] [loss] {:.3f} | loss_x:{:.3f} | x_pp:{:.3f} | loss_y:{:.3f} | accuracy_t:{:.3f} | accuracy_f:{:.3f} | beta:{:.3f}'.format(b, total_loss, total_loss_x, math.exp(total_loss_x), total_loss_y, accuracy_t, accuracy_f, beta)
            print(log)
            total_loss, total_loss_x, total_loss_y = 0, 0, 0
            corrects_f, corrects_t = 0, 0
            _, sample = generated.data.cpu()[:,0,:].topk(1)
            text_sample = " ".join([TEXT.vocab.itos[i] for i in text.data[:,0]])
            generated_sample = " ".join([TEXT.vocab.itos[i] for i in sample.squeeze()])
            with open("./temp/log.txt", "a+") as f:
                f.write("\n" + log + "\n")
                f.write("[ORI]: %s\n" % (text_sample))
                f.write("[GEN]: %s\n" % (generated_sample))


def main():
    args = parse_arguments()
    use_cuda = torch.cuda.is_available()

    print("[!] preparing dataset...")
    TEXT = data.Field(lower=True)
    LABEL = data.Field(sequential=False)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data, min_freq=5)
    LABEL.build_vocab(train_data)
    train_iter, test_iter = data.BucketIterator.splits((train_data, test_data),
            batch_size=args.batch_size, repeat=False)
    vocab_size = len(TEXT.vocab)
    print("[TRAIN]: %d (dataset %d) \t [TEST]: %d (dataset %d) \t [VOCAB] %d"
          % (len(train_iter), len(train_iter.dataset), len(test_iter), len(test_iter.dataset), vocab_size))

    print("[!] Instantiating models...")
    classifier = ConvText(vocab_size=vocab_size, embed_size=128, n_classes=2)
    classifier.load_state_dict(torch.load("./classifiers/snapshot/cnn_5_128.pt"))
    generator = RNNModel('LSTM', ntoken=vocab_size, ninp=128, nhid=128, nlayers=2, dropout=0.4)
    generator_op = optim.Adam(generator.parameters(), lr=args.lr)
    if use_cuda:
        print("[!] Using CUDA...")
        classifier.cuda()
        generator.cuda()

    best_val_loss = None
    beta = 1.0
    for e in range(1, args.epochs+1):
        train(classifier, generator, generator_op, train_iter, beta, TEXT, use_cuda, args)
        stats = evaluate(classifier, generator, test_iter, beta, TEXT, use_cuda, args)
        val_loss, val_loss_x, val_loss_y, val_accuracy_t, val_accuracy_f = stats
        print("\n[Epoch: %d] val_loss:%5.2f | val_pp:%.2f | loss_x:%.2f | loss_y:%.2f | accuracy_t:%.2f | accuracy_f:%.2f "
               % (e, val_loss, math.exp(val_loss), val_loss_x, val_loss_y, val_accuracy_t, val_accuracy_f))

        if beta > 0.5:
            beta *= .8
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model")
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(generator.state_dict(), './snapshot/ratn_{}.pt'.format(e))
            best_val_loss = val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
