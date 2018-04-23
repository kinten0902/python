# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
from chainer import cuda, Chain, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import numpy as np
import math
import datetime
import time
from chainer.functions.array import *
from chainer.functions.math import matmul
import sys
import types

try:
    cuda.check_cuda_available()
    import cupy
    xp = cupy
except:
    xp = np
'''
The abstract class.
'''


class LSTMBase(Chain):
    def update(self):
        self.optimizer.zero_grads()
        self.loss_var.backward()  # calc grads
        self.optimizer.clip_grads(1.0)  # truncate
        self.optimizer.update()
        self.reset_state()

    def score(self, ys, ts):
        ret = []
        top5s = np.argsort(ys, axis=1)[:, ::-1][:, :5]
        self.top5s = top5s
        for t, top5 in zip(ts, top5s):
            if t in top5:
                j = np.nonzero(top5 == t)[0] + 1
                ret.append(1.0 / np.log2(j + 1))
            else:
                ret.append(0.0)
        return np.asarray(ret, dtype=np.float32)

    def feed_for_learn(self, xs, ts):
        self.whole = True
        xss, tss = xs, Variable(xp.asarray(ts))
        yss = self(xss, train=True)
        loss_tmp = F.softmax_cross_entropy(yss, tss)
        out = loss_tmp.creator.inputs[0].data
        if xp is not np: out = out.get()
        corrects = (np.argmax(out, axis=1) == ts)
        score_top5 = self.score(out, ts)
        self.loss_var += loss_tmp
        return corrects, score_top5

    def only_feed_for_learn(self, xs):
        self.whole = False
        self(xs, train=True)
        return

    def feed(self, xs):
        # volatile = True -> BP chains are teard -> a test don't use BPs -> rapid calculation
        self.whole = True
        xss = xs
        yss = self(xss, train=False)
        out = F.softmax(yss).data
        if xp is not np: out = out.get()
        return yss, out

    def check(self, xs, ts):
        self.whole = True
        yss, out = self.feed(xs)
        ts = np.asarray(ts)
        ac = F.accuracy(yss, Variable(xp.asarray(ts),
                                      volatile=True)).data.tolist()
        corrects = (np.argmax(out, axis=1) == ts)
        score_top5 = self.score(out, ts)
        return corrects, score_top5

    def only_feed(self, xs):
        # volatile = True -> BP chains are teard -> a test don't use BPs -> rapid calculation
        self.whole = False
        self(xs, train=False)
        return


'''
Simple LSTM model.
'''


class Simple(LSTMBase):
    def __init__(self,
                 vocab_size,
                 dim_embed=200,
                 dim1=800,
                 dim2=800,
                 dim3=200,
                 class_size=None):
        if class_size is None:
            class_size = vocab_size
        super(Simple, self).__init__(
            embed2=L.EmbedID(vocab_size, dim_embed),
            lay2=L.LSTM(dim_embed, dim1, forget_bias_init=1),
            lay_int=L.LSTM(dim1, dim2, forget_bias_init=1),
            lin1=L.Linear(dim2, dim3),
            lin2=L.Linear(dim3, class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, xs, train):
        x_uni = xs[0]
        x_rev = xs[1]
        #if False:
        #    x_3gram = xs[0]
        #    sp2 = xs[1]
        #    x_uni = x_3gram[:,0]
        #else:
        #    x_uni = xs
        y = Variable(x_uni, volatile=not train)
        y = self.embed2(y)
        y2 = self.lay2(y)
        y2 = self.lay_int(y2)
        y = y2
        y = self.lin1(F.dropout(y, train=train))
        y = F.relu(y)
        y = self.lin2(F.dropout(y, train=train))

        return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # for safty
        self.loss_var = Variable(xp.zeros((),
                                          dtype=xp.float32))  # reset loss_var
        self.lay2.reset_state()
        self.lay_int.reset_state()
        return


'''
Simple LSTM model.
'''


class Simple_bothway(LSTMBase):
    def __init__(self,
                 vocab_size,
                 dim_embed=200,
                 dim1=800,
                 dim2=800,
                 dim=200,
                 class_size=None,
                 num_use_out=2):
        if class_size is None:
            class_size = vocab_size
        self.num_use_out = num_use_out
        super(Simple_bothway, self).__init__(
            embed=L.EmbedID(vocab_size, dim_embed),
            embed_rev=L.EmbedID(vocab_size, dim_embed),
            bn=L.BatchNormalization(2),
            lay1=L.LSTM(dim_embed, dim1, forget_bias_init=1),
            lay2=L.LSTM(dim1, dim2, forget_bias_init=1),
            lay1_rev=L.LSTM(dim_embed, dim1, forget_bias_init=1),
            lay2_rev=L.LSTM(dim1, dim2, forget_bias_init=1),
            lin_rnn=L.Linear(dim2 * self.num_use_out, dim),
            lin_int=L.Linear(dim, dim),
            lin_sm=L.Linear(dim, class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()
        self.use_w2v = False

    def __call__(self, xs, train):
        x_uni = xs[0]
        x_rev = xs[1]

        y = Variable(x_uni, volatile=not train)
        y_rev = Variable(x_rev, volatile=not train)

        if not self.use_w2v:
            y = self.embed(y)
            y_rev = self.embed_rev(y_rev)
            y = F.dropout(y, train=train)
            y_rev = F.dropout(y_rev, train=train)

        ys = [y, y_rev]

        y = self.lay1(ys[0])
        y = self.lay2(y)

        y_rev = self.lay1_rev(ys[1])
        y_rev = self.lay2_rev(y_rev)

        if self.whole:
            z = concat.concat([y, y_rev][0:self.num_use_out])
            z = F.dropout(z, train=train)
            z = F.relu(self.lin_rnn(z))
            z = F.dropout(z, train=train)
            z = F.relu(self.lin_int(z))
            z = F.dropout(z, train=train)
            z = self.lin_sm(z)
            return z

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # for safty
        self.loss_var = Variable(xp.zeros((),
                                          dtype=xp.float32))  # reset loss_var
        self.lay1.reset_state()
        self.lay2.reset_state()
        self.lay1_rev.reset_state()
        self.lay2_rev.reset_state()
        return


class Simple_bothway_seq2seq(LSTMBase):
    def __init__(self,
                 vocab_size,
                 dim_embed=200,
                 dim1=600,
                 dim2=600,
                 dim=200,
                 class_size=None,
                 num_use_out=2):
        if class_size is None:
            class_size = vocab_size
        self.num_use_out = num_use_out
        super(Simple_bothway_seq2seq, self).__init__(
            embed=L.EmbedID(vocab_size, dim_embed),
            embed_rev=L.EmbedID(vocab_size, dim_embed),
            pembed=L.EmbedID(vocab_size, dim_embed),
            pembed_rev=L.EmbedID(vocab_size, dim_embed),
            lay1=L.LSTM(dim_embed, dim1, forget_bias_init=1),
            lay2=L.LSTM(dim1, dim2, forget_bias_init=1),
            lay1_rev=L.LSTM(dim_embed, dim1, forget_bias_init=1),
            lay2_rev=L.LSTM(dim1, dim2, forget_bias_init=1),
            play1=L.LSTM(dim_embed, dim1, forget_bias_init=1),
            play2=L.LSTM(dim1, dim2, forget_bias_init=1),
            play1_rev=L.LSTM(dim_embed, dim1, forget_bias_init=1),
            play2_rev=L.LSTM(dim1, dim2, forget_bias_init=1),
            lin_rnn=L.Linear(dim2 * self.num_use_out, dim),
            lin_int=L.Linear(dim, dim),
            lin_sm=L.Linear(dim, class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.dim_embed = dim_embed
        #self.optimizer = optimizers.MomentumSGD()
        #self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def connect_mid(self):
        self.is_pre[0] = False
        self.lay1.set_state(self.play1.c, self.play1.h)
        self.lay2.set_state(self.play2.c, self.play2.h)
        self.mid_out = self.play2.h

    def connect_mid_rev(self):
        self.is_pre[1] = False
        self.lay1_rev.set_state(self.play1_rev.c, self.play1_rev.h)
        self.lay2_rev.set_state(self.play2_rev.c, self.play2_rev.h)
        self.mid_rev_out = self.play2_rev.h

    def one_way(self, y, is_pre_rev, train):
        if is_pre_rev == (True, True):
            embed = self.pembed_rev
            lay1 = self.play1_rev
            lay2 = self.play2_rev
        elif is_pre_rev == (False, True):
            embed = self.embed_rev
            lay1 = self.lay1_rev
            lay2 = self.lay2_rev
        elif is_pre_rev == (True, False):
            embed = self.pembed
            lay1 = self.play1
            lay2 = self.play2
        else:
            embed = self.embed
            lay1 = self.lay1
            lay2 = self.lay2
        y = embed(y)
        y = F.dropout(y, train=train)
        y = lay1(y)
        y = lay2(y)
        return y

    def __call__(self, xs, train):
        x = Variable(xs[0], volatile=not train)
        x_rev = Variable(xs[1], volatile=not train)

        is_pre_rev = (self.is_pre[0], False)
        y = self.one_way(x, is_pre_rev, train=train)
        is_pre_rev = (self.is_pre[1], True)
        y_rev = self.one_way(x_rev, is_pre_rev, train=train)

        if self.whole:
            use_out = [y, y_rev, self.mid_out,
                       self.mid_rev_out][0:self.num_use_out]
            z = concat.concat(use_out)
            #  z = concat.concat((y,y_rev))
            z = F.dropout(z, train=train)
            z = F.relu(self.lin_rnn(z))
            z = F.dropout(z, train=train)
            z = F.relu(self.lin_int(z))
            z = F.dropout(z, train=train)
            z = self.lin_sm(z)
            return z

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # for safty
        self.loss_var = Variable(xp.zeros((),
                                          dtype=xp.float32))  # reset loss_var
        self.lay1.reset_state()
        self.lay2.reset_state()
        self.lay1_rev.reset_state()
        self.lay2_rev.reset_state()
        self.play1.reset_state()
        self.play2.reset_state()
        self.play1_rev.reset_state()
        self.play2_rev.reset_state()
        self.is_pre = [True, True]
        self.mid_out = None
        self.mid_rev_out = None


'''
The model using Sp2.
'''


class Sp2(LSTMBase):
    def __init__(self,
                 vocab_size,
                 dim_embed=33 * 3,
                 dim1=400,
                 dim2=100,
                 dim3=100,
                 class_size=None):
        if class_size is None:
            class_size = vocab_size
        super(Sp2, self).__init__(
            embed2=L.EmbedID(vocab_size, dim_embed),
            embed3=L.Linear(vocab_size, dim_embed),
            lin3=L.Linear(dim_embed, dim1),
            lay2=L.LSTM(dim_embed, dim1, forget_bias_init=0),
            lay_int=L.LSTM(dim1, dim2, forget_bias_init=0),
            lin1=L.Linear(dim1 + dim2, dim3),
            lin2=L.Linear(dim3, class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, xs, train):
        x_3gram = xs[0]
        sp2 = xs[1]

        x_uni = x_3gram[:, 0]
        y = Variable(x_uni, volatile=not train)
        y = self.embed2(y)
        y2 = self.lay2(y)
        y2 = self.lay_int(y2)

        y = Variable(sp2, volatile=not train)
        y = self.embed3(y)
        y = self.lin3(y)
        y3 = F.relu(y)

        y = concat.concat((y2, y3))
        #y = concat.concat((y3,y3) )
        y = self.lin1(F.dropout(y, train=train))
        y = F.relu(y)
        y = self.lin2(F.dropout(y, train=train))

        return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # 念のため
        self.loss_var = Variable(xp.zeros((),
                                          dtype=xp.float32))  # reset loss_var
        self.lay2.reset_state()
        self.lay_int.reset_state()
        return


'''
The model using Bigrams.
'''


class Bigram(LSTMBase):
    def __init__(self,
                 vocab_size,
                 dim_embed=33 * 3,
                 dim1=400,
                 dim2=400,
                 dim3=200,
                 class_size=None):
        if class_size is None:
            class_size = vocab_size
        super(Bigram, self).__init__(
            embed_uni=L.EmbedID(vocab_size, dim_embed),
            embed_bi=L.EmbedID(vocab_size * vocab_size, dim_embed),
            lay_uni=L.LSTM(dim_embed, dim1, forget_bias_init=0),
            lay_bi=L.StatelessLSTM(dim_embed, dim1, forget_bias_init=0),
            lay_int=L.LSTM(dim1 * 3, dim2, forget_bias_init=0),
            lin1=L.Linear(dim2, dim3),
            lin2=L.Linear(dim3, class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, xs, train):

        x_3gram = xs[0]
        sp2 = xs[1]

        x_uni = x_3gram[:, 0]
        y = Variable(x_uni, volatile=not train)
        y = self.embed_uni(y)
        y_uni = self.lay_uni(y)

        ## bigram です。
        x_bi = x_3gram[:, 0] * self.vocab_size + x_3gram[:, 1]
        y = Variable(x_bi, volatile=not train)
        y = self.embed_bi(y)
        if self.is_odd:
            self.c_odd, self.h_odd = self.lay_bi(self.c_odd, self.h_odd, y)
            if self.h_evn is None:
                self.h_evn = Variable(
                    xp.zeros_like(self.h_odd.data), volatile=not train)
            y = concat.concat((y_uni, self.h_odd, self.h_evn))
        else:
            self.c_evn, self.h_evn = self.lay_bi(self.c_evn, self.h_evn, y)
            y = concat.concat((y_uni, self.h_evn, self.h_odd))
        self.is_odd = not self.is_odd

        y = self.lay_int(y)
        y = self.lin1(F.dropout(y, train=train))
        y = F.relu(y)
        y = self.lin2(F.dropout(y, train=train))

        return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # 念のため
        self.loss_var = Variable(xp.zeros((),
                                          dtype=xp.float32))  # reset loss_var
        self.lay_uni.reset_state()
        self.is_odd = True
        self.c_odd = None
        self.c_evn = None
        self.h_odd = None
        self.h_evn = None
        self.lay_int.reset_state()
        return


'''
The abstract class.
'''


class CNNBase(Chain):
    def update(self):
        self.optimizer.zero_grads()
        self.loss_var.backward()  # calc grads
        self.optimizer.clip_grads(1.0)  # truncate
        self.optimizer.update()

    def score(self, ys, ts):
        ret = []
        top5s = np.argsort(ys, axis=1)[:, ::-1][:, :5]
        self.top5s = top5s
        for t, top5 in zip(ts, top5s):
            if t in top5:
                j = np.nonzero(top5 == t)[0] + 1
                ret.append(1.0 / np.log2(j + 1))
            else:
                ret.append(0.0)
        return np.asarray(ret, dtype=np.float32)

    def feed_for_learn(self, xs, ts):
        xss, tss = xs, Variable(xp.asarray(ts))
        yss = self(xss, train=True)
        loss_tmp = F.softmax_cross_entropy(yss, tss)
        out = loss_tmp.creator.inputs[0].data
        if xp is not np: out = out.get()
        corrects = (np.argmax(out, axis=1) == ts)
        score_top5 = self.score(out, ts)
        self.loss_var = loss_tmp
        return corrects, score_top5

    def feed(self, xs):
        # volatile = True -> BP chains are teard -> a test don't use BPs -> rapid calculation
        xss = xs
        yss = self(xss, train=False)
        return yss, F.softmax(yss).data

    def check(self, xs, ts):
        yss, out = self.feed(xs)
        if xp is not np: out = out.get()
        ts = np.asarray(ts)
        ac = F.accuracy(yss, Variable(xp.asarray(ts),
                                      volatile=True)).data.tolist()
        corrects = (np.argmax(out, axis=1) == ts)
        score_top5 = self.score(out, ts)
        return corrects, score_top5


'''
Simple CNN model.
'''


class SimpleCNN(CNNBase):
    def __init__(self,
                 vocab_size,
                 dim_embed=200,
                 ch=256,
                 dim1=200,
                 gram=4,
                 class_size=None,
                 w2v=True):
        super(SimpleCNN, self).__init__(
            embed=L.EmbedID(vocab_size, dim_embed),
            conv=L.Convolution2D(1, ch, (gram, dim_embed), pad=(gram / 2, 0)),
            lin1=L.Linear(ch, dim1),
            lin2=L.Linear(dim1, class_size),
        )
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.w2v = w2v
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))

    def __call__(self, xs, train):
        #if len(xs.shape) == 3 :
        if self.w2v:
            ys = Variable(xs[0], volatile=not train)
        else:
            ys = self.embed(Variable(xs[1], volatile=not train))
        sh = ys.data.shape
        y = reshape.reshape(ys, (sh[0], 1, sh[1], sh[2]))
        y = F.relu(self.conv(y))
        #print y.data.shape
        length = y.data.shape[2]
        y = F.max_pooling_2d(y, (length, 1))
        y = F.dropout(y, train=train)
        y = F.relu(self.lin1(y))
        y = F.dropout(y, train=train)
        y = self.lin2(y)
        return y


'''
Simple CNN model.
'''


class DeepCNN(CNNBase):
    def __init__(self,
                 vocab_size,
                 dim_embed=128,
                 ch1=96,
                 ch2=128,
                 ch3=256,
                 dim1=256,
                 gram=11,
                 class_size=None):
        super(DeepCNN, self).__init__(
            embed_a=L.EmbedID(vocab_size, dim_embed),
            embed_b=L.EmbedID(vocab_size, dim_embed),
            embed_c=L.EmbedID(vocab_size, dim_embed),
            bn=L.BatchNormalization(3),
            lay1_a=L.Convolution2D(3, ch1, (2, gram), pad=(0, (gram - 1) / 2)),
            lay1_b=L.Convolution2D(
                ch1, ch2, (2, gram), pad=(0, (gram - 1) / 2)),
            lay1_c=L.Convolution2D(ch2, ch3, (2, dim_embed), pad=0),
            lin1=L.Linear(ch3, ch3),
            lin2=L.Linear(ch3, dim1),
            lin3=L.Linear(dim1, class_size),
        )
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))

    def __call__(self, xs, train):
        #if len(xs.shape) == 3 :
        if False:
            sh = xs[0].shape
            y_b = Variable(
                xs[0].reshape((sh[0], 1, sh[1], sh[2])), volatile=not train)
        #else:
        if True:
            #sh = xs.shape
            #print sh
            #xs = [self.embed( Variable(x, volatile = not train) ) for x in xs]
            #xs = concat.concat(xs)
            embeds = [self.embed_a, self.embed_b, self.embed_c]
            ys = [
                embed(Variable(xs[1], volatile=not train)) for embed in embeds
            ]
            sh = ys[0].data.shape
            ys = [reshape.reshape(y, (sh[0], 1, sh[1], sh[2])) for y in ys]
            y = concat.concat(ys, axis=1)
        #y = self.bn(y, test=not train)
        #y = F.dropout(y, train=train)
        #print y.data.shape
        y = F.relu(self.lay1_a(y))
        #print y.data.shape
        y = F.relu(self.lay1_b(y))
        y = F.relu(self.lay1_c(y))
        #print y.data.shape
        length = y.data.shape[2]
        y = F.max_pooling_2d(y, (length, 1))
        y = F.dropout(y, train=train)
        y = F.relu(self.lin1(y))
        y = F.dropout(y, train=train)
        y = F.relu(self.lin2(y))
        y = F.dropout(y, train=train)
        y = self.lin3(y)
        return y


'''
Combination model (LSTM+CNN).
'''
'''
Simple LSTM model.
'''


class Combination(LSTMBase):
    def __init__(self,
                 vocab_size,
                 dim_embed_rnn=200,
                 dim_LSTM=800,
                 class_size=None,
                 dim_embed=128,
                 ch1=64,
                 ch2=128,
                 ch3=256,
                 dim1=256,
                 gram=11):
        if class_size is None:
            class_size = vocab_size
        super(Combination, self).__init__(
            embed=L.EmbedID(vocab_size, dim_embed_rnn),
            embed_rev=L.EmbedID(vocab_size, dim_embed_rnn),
            rnn1=L.LSTM(dim_embed_rnn, dim_LSTM, forget_bias_init=0),
            rnn2=L.LSTM(dim_LSTM, dim_LSTM, forget_bias_init=0),
            rnn1_rev=L.LSTM(dim_embed_rnn, dim_LSTM, forget_bias_init=0),
            rnn2_rev=L.LSTM(dim_LSTM, dim_LSTM, forget_bias_init=0),
            lin_rnn=L.Linear(dim_LSTM * 2, dim1),
            embed_a=L.EmbedID(vocab_size, dim_embed),
            embed_b=L.EmbedID(vocab_size, dim_embed),
            embed_c=L.EmbedID(vocab_size, dim_embed),
            bn=L.BatchNormalization(3),
            lay1_a=L.Convolution2D(3, ch1, (2, gram), pad=(0, (gram - 1) / 2)),
            lay1_b=L.Convolution2D(
                ch1, ch2, (2, gram), pad=(0, (gram - 1) / 2)),
            lay1_c=L.Convolution2D(ch2, ch3, (2, dim_embed), pad=0),
            lin_cnn=L.Linear(ch3, dim1),
            lin_int=L.Linear(dim1 * 2, dim1),
            lin_sm=L.Linear(dim1, class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print('run on the GPU.')
        except:
            print('run on the CPU.')
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, xs, train):
        if not self.whole:
            return

        #only LSTM
        x_uni = xs[0]
        x_rev = xs[1]

        y = Variable(x_uni, volatile=not train)
        y = self.embed(y)
        #y = F.dropout(y, train=train)

        y = self.rnn1(y)
        y = self.rnn2(y)

        y_rev = Variable(x_rev, volatile=not train)
        y_rev = self.embed_rev(y_rev)
        #y_rev = F.dropout(y_rev, train=train)

        y_rev = self.rnn1_rev(y_rev)
        y_rev = self.rnn2_rev(y_rev)

        y_rnn = concat.concat((y, y_rev))
        y_rnn = F.dropout(y_rnn, train=train)
        y_rnn = F.relu(self.lin_rnn(y_rnn))

        if self.whole:
            ## add cnn
            embeds = [self.embed_a, self.embed_b, self.embed_c]
            ys = [
                embed(Variable(xs[2], volatile=not train)) for embed in embeds
            ]
            sh = ys[0].data.shape
            ys = [reshape.reshape(y, (sh[0], 1, sh[1], sh[2])) for y in ys]
            y = concat.concat(ys, axis=1)
            y = self.bn(y, test=not train)
            y = F.relu(self.lay1_a(y))
            y = F.relu(self.lay1_b(y))
            y = F.relu(self.lay1_c(y))
            length = y.data.shape[2]
            y = F.max_pooling_2d(y, (length, 1))
            y = F.dropout(y, train=train)
            y_cnn = F.relu(self.lin_cnn(y))

            y = concat.concat((y_rnn, y_cnn))
            y = F.dropout(y, train=train)
            y = F.relu(self.lin_int(y))
            y = F.dropout(y, train=train)
            y = self.lin_sm(y)
            return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # for safty
        self.loss_var = Variable(xp.zeros((),
                                          dtype=xp.float32))  # reset loss_var
        self.rnn1.reset_state()
        self.rnn2.reset_state()
        self.rnn1_rev.reset_state()
        self.rnn2_rev.reset_state()
        return