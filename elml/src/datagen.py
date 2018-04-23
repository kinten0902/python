import numpy as np
from numpy import random
import numpy as np
import math
import datetime
import time
import sys

#import models


class DataGenerator:
    '''
    Read the data from a file.
    start_symbol   : -2 mod |Sigma|
    end_symbol     : -1 mod |Sigma|
    padding_symbol : -1 mod |Sigma|
    '''

    def make_pos_data(self, tmp, start_symbol, end_symbol, padding_symbol):
        if end_symbol is not None:
            tmp = [[start_symbol] + s + [end_symbol] for s in tmp]
        pos = np.cumsum([0] + [len(lst) for lst in tmp])
        pos = np.c_[pos, np.roll(pos, -1)][:-1]
        data = np.asarray([a for lst in tmp for a in lst], dtype=np.int32)
        return pos, data

    def __init__(self,
                 filename,
                 filename2=None,
                 fmt="pautomac",
                 start_symbol=-2,
                 end_symbol=-1,
                 padding_symbol=-1,
                 all_train=False,
                 all_test=False):
        self.all_train = all_train
        self.all_test = all_test
        self.has_pre = (filename2 != None)
        if fmt == "pautomac":
            tmp = [l.strip().split()[1:]
                   for l in open(filename).readlines()][1:]
            self.label = None
        elif fmt == "labeled":
            tmp = [l.strip().split()[1:] for l in open(filename).readlines()]
            self.label = np.asarray(
                [l.strip().split()[0] for l in open(filename).readlines()],
                dtype=np.int32)
        elif fmt == "2-labeled":
            tmp = [l.strip().split()[2:] for l in open(filename).readlines()]
            if self.has_pre:
                tmp_pre = [
                    l.strip().split() for l in open(filename2).readlines()
                ]
            self.label = np.asarray(
                [l.strip().split()[0:2] for l in open(filename).readlines()],
                dtype=np.int32)

        self.pos, self.data = self.make_pos_data(tmp, start_symbol, end_symbol,
                                                 padding_symbol)
        if self.has_pre:
            self.pos_pre, self.data_pre = self.make_pos_data(
                tmp_pre, start_symbol, end_symbol, padding_symbol)
        self.kind = np.max(self.data) + 3
        self.data = self.data % self.kind
        self.start_symbol = start_symbol % self.kind
        self.end_symbol = end_symbol % self.kind
        self.padding_symbol = padding_symbol % self.kind
        self.pos0_to_id = {p[0]: i for i, p in enumerate(self.pos)}
        self.make_cv(0)

    def get_ids_from_batch(self, batch_ids, train=True):
        if train:
            pos = self.pos_tr[batch_ids]
        else:
            pos = self.pos_te[batch_ids]
        return np.asarray([self.pos0_to_id[p[0]] for p in pos])

    def make_cv(self, i):
        te = np.arange(len(self.pos)) % 10 == i
        #te = np.arange(len(pos))>(len(pos)*0.9)
        pos = self.pos
        if self.all_train:
            self.pos_tr = pos
            self.pos_te = pos[0]
            self.label_tr = self.label
            self.label_te = self.label[0]
            if self.has_pre:
                pos_pre = self.pos_pre
                self.pos_pre_tr = pos_pre
                self.pos_pre_te = pos_pre[0]
        elif self.all_test:
            self.pos_tr = pos[0]
            self.pos_te = pos
            self.label_tr = self.label[0]
            self.label_te = self.label
            if self.has_pre:
                pos_pre = self.pos_pre
                self.pos_pre_tr = pos_pre[0]
                self.pos_pre_te = pos_pre
        else:
            self.pos_tr = pos[~te]
            self.pos_te = pos[te]
            self.label_tr = self.label[~te]
            self.label_te = self.label[te]
            if self.has_pre:
                pos_pre = self.pos_pre
                self.pos_pre_tr = pos_pre[~te]
                self.pos_pre_te = pos_pre[te]

    '''
    Batched letters in batched sentences are yeilded sequentially.
    If the length of sentences are different in the batch,
    We padding the ends of sentences by -1 after shorter sentences are finished.
    arg: batch_ids, an array of ids for sentences.
    return : an array of letters.  
    '''

    def sequence(self, batch_ids, train=True):
        if train:
            pos = self.pos_tr[batch_ids]
        else:
            pos = self.pos_te[batch_ids]
        ids = pos[:, 0]
        end = pos[:, 1] - 1
        xs = self.data[ids]
        yield xs.copy(), np.ones(len(xs), dtype=np.bool)
        dif = ids < end
        while np.any(dif):
            ids = ids + dif
            xs[~dif] = self.padding_symbol
            xs[dif] = self.data[ids][dif]
            yield xs.copy(), dif
            dif = ids < end

    def private_sequence_bothway(self, pos, data, bsize):
        offset = pos[:, 0]
        topside = pos[:, 1]
        rem = pos[:, 1] - offset
        k = np.max(rem)
        xs = np.zeros((2, bsize), dtype=np.int32)
        while k > 0:
            k = k - 1
            inRange = k < rem
            xs[1, ~inRange] = self.padding_symbol
            xs[1, inRange] = data[offset[inRange] + k]
            xs[0, ~inRange] = self.padding_symbol
            xs[0, inRange] = data[topside[inRange] - 1 - k]
            yield xs[0].copy(), xs[1].copy(), inRange

    def sequence_bothway_seq2seq(self, batch_ids, train=True):
        t = [[], [], [], []]
        if train:
            pos_pre = self.pos_pre_tr[batch_ids]
            pos = self.pos_tr[batch_ids]
        else:
            pos_pre = self.pos_pre_te[batch_ids]
            pos = self.pos_te[batch_ids]
        for xs, xs_rev, flags in self.private_sequence_bothway(
                pos, self.data, len(batch_ids)):
            t[0].append(xs)
            t[1].append(xs_rev)
        for xs, xs_rev, flags in self.private_sequence_bothway(
                pos_pre, self.data_pre, len(batch_ids)):
            t[2].append(xs)
            t[3].append(xs_rev)

        for i, (xs, xs_rev) in enumerate(zip(t[2] + t[0], t[1] + t[3])):
            is_change = (i == len(t[2]) - 1, i == len(t[1]) - 1)
            yield xs, xs_rev, is_change

    def sequence_bothway(self, batch_ids, train=True):
        if train:
            pos = self.pos_tr[batch_ids]
        else:
            pos = self.pos_te[batch_ids]
        offset = pos[:, 0]
        topside = pos[:, 1]
        rem = pos[:, 1] - offset
        k = np.max(rem)
        xs = np.zeros((2, len(batch_ids)), dtype=np.int32)
        while k > 0:
            k = k - 1
            inRange = k < rem
            xs[1, ~inRange] = self.padding_symbol
            xs[1, inRange] = self.data[offset[inRange] + k]
            xs[0, ~inRange] = self.padding_symbol
            xs[0, inRange] = self.data[topside[inRange] - 1 - k]
            yield xs[0].copy(), xs[1].copy(), inRange

    def sequence_bothway_2D(self, batch_ids, train=True):
        for xs, xs_rev, inRange in self.sequence_bothway(batch_ids, train):
            xs_2D = np.asarray([self.id_to_vec[x] for x in xs])
            xs_rev_2D = np.asarray([self.id_to_vec[x] for x in xs_rev])
            yield xs_2D, xs_rev_2D, inRange

    '''
    Called from the inside.
    The sentences are sorted by the length and suffled.
    '''

    def shuffle(self, shuffle_block_size, train):
        if train:
            lengths = self.pos_tr[:, 1] - self.pos_tr[:, 0]
            perturbation = np.random.uniform(0, 0.999, len(lengths))
        else:
            lengths = self.pos_te[:, 1] - self.pos_te[:, 0]
            perturbation = np.zeros(len(lengths))

        sorted_order = np.argsort(lengths + perturbation)
        '''When the length of data is not devied by block_size, the set of blocks must have another block.'''
        rem = int(len(lengths) % shuffle_block_size != 0)
        blocks = np.arange(len(lengths) / shuffle_block_size + rem)
        if train:
            np.random.shuffle(blocks)
        return blocks, sorted_order

    '''
    Yields batched sentences from the starts to the ends simulteniously.  
    return: an array of ids for sentences.
    '''

    def batched_sentences(self, batch_size=16, train=True):
        blocks, sorted_order = self.shuffle(batch_size, train)
        print("blocks",blocks)
        for b in blocks:
            b = int(b)
            yield sorted_order[b * batch_size:(b + 1) * batch_size]

    def get_labels(self, batch_ids, train=True):
        if self.label is None:
            return None
        if train:
            return self.label_tr[batch_ids]
        else:
            return self.label_te[batch_ids]

    def get_len(self, batch_ids, train=True):
        if train:
            return np.max(self.pos_tr[batch_ids][:, 1] -
                          self.pos_tr[batch_ids][:, 0])
        else:
            return np.max(self.pos_te[batch_ids][:, 1] -
                          self.pos_te[batch_ids][:, 0])

    def prepare_w2v_jawiki(self, w2v_model, wd_to_id, wd_set):
        def get_id(wd):
            if wd.decode('utf-8') not in w2v_model.vocab:
                print(("Exception:", wd, wd_set[wd]))
                return np.random.normal(size=w2v_model.vector_size)
            else:
                return w2v_model[wd.decode('utf-8')]

        id_to_vec = {
            wd_to_id[wd]: np.asarray(get_id(wd), dtype=np.float32)
            for wd in list(wd_to_id.keys())
        }
        id_to_vec[self.start_symbol] = np.zeros(
            w2v_model.vector_size, dtype=np.float32)
        id_to_vec[self.end_symbol] = np.zeros(
            w2v_model.vector_size, dtype=np.float32)
        id_to_vec[self.padding_symbol] = np.zeros(
            w2v_model.vector_size, dtype=np.float32)
        self.id_to_vec = id_to_vec
        self.vec_size = w2v_model.vector_size

    def sentence_2D_batch(self, batch_ids, train=True):
        max_len = self.get_len(batch_ids, train=train)
        imgs = np.zeros(
            (len(batch_ids), max_len, self.vec_size), dtype=np.float32)

        for i, (xs, dif) in enumerate(self.sequence(batch_ids, train=train)):
            for b, wid in enumerate(xs):
                imgs[b][i] = self.id_to_vec[wid]
        return imgs

    def sentence_1D_batch(self, batch_ids, train=True):
        max_len = self.get_len(batch_ids, train=train)
        ids = np.zeros((len(batch_ids), max_len), dtype=np.int32)

        for i, (xs, dif) in enumerate(self.sequence(batch_ids, train=train)):
            for b, wid in enumerate(xs):
                ids[b][i] = wid
        return ids