import numpy as np
class DataSet(object):
    @property
    def vectors(self):
        return self._vectors

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __init__(self, vectors, labels):
        self._num_examples = vectors.shape[0]
        self._vectors = vectors
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._vectors = self._vectors[perm]
            self._labels = self._labels[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._vectors[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

def extract_labels(data, num_classes, one_hot):
        if one_hot:
            return dense_to_one_hot(data, num_classes=num_classes)
        return data

def read_data_sets(vectors, labels, num_classes):
        class DataSets(object):
            pass

        data_sets = DataSets()

        train_flag = np.arange(len(vectors)) % 10 != 0

        train_vectors = vectors[train_flag]
        train_labels = labels[train_flag]

        test_vectors = vectors[~train_flag]
        test_labels = extract_labels(
            labels[~train_flag], num_classes=num_classes, one_hot=one_hot)

        data_sets.train = DataSet(train_vectors, train_labels)
        data_sets.test = DataSet(test_vectors, test_labels)
        return data_sets
    
def sen_to_fv(sen_id, sen_pre_id, max_len, model):
    sen_fv = []
    sen_pre_fv = []
    for s, s_pre in zip(sen_id, sen_pre_id):
        s_pad = np.pad(
            list(map(int, s)), (0, max_len - len(s)),
            'constant',
            constant_values=(0, 0))
        s_pre_pad = np.pad(
            list(map(int, s_pre)), (0, max_len - len(s_pre)),
            'constant',
            constant_values=(0, 0))
        s_vec_1d = np.array([model.wv[str(w)] for w in s_pad]).ravel()
        s_pre_vec_1d = np.array([model.wv[str(w)] for w in s_pre_pad]).ravel()
        sen_fv.append(s_vec_1d)
        sen_pre_fv.append(s_pre_vec_1d)

    return np.array(sen_fv), np.array(sen_pre_fv)