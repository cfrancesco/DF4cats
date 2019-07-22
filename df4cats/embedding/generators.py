import numpy as np
from keras.utils import Sequence


class BaseGenerator(Sequence):
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = self._build_index()
        self.on_epoch_end()

    def __len__(self):
        return len(self.y) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _build_index(self):
        return np.arange(len(self.y))

    def _get_data(self, indexes):
        samples = self.X[indexes]
        labels = self.y[indexes]
        assert len(samples) == len(labels)
        return samples, labels

    def _get_indexes(self, index):
        return self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

    def __getitem__(self, index):
        indexes = self._get_indexes(index)
        samples, labels = self._get_data(indexes)
        return samples, labels


class BaseGeneratorDF(BaseGenerator):
    def __init__(self, X, y, batch_size, shuffle=True):
        super().__init__(X, y, batch_size, shuffle)

    def _get_data(self, indexes):
        samples = self.X.iloc[indexes]
        labels = self.y.iloc[indexes]
        assert len(samples) == len(labels)
        return samples, labels


class SiameseGenerator(BaseGenerator):
    def __init__(self, X, y, batch_size, shuffle=True):
        self.label_set = list(set(y))
        super().__init__(X, y, batch_size, shuffle)

    def _get_derangement(self, array):
        """ Get a random derangement of a list of indexes. """

        ders = [(x, np.random.choice(array[np.where(array != x)])) for x in array]
        return np.array(ders)

    def _get_indexes(self, index):
        pair_indexes = {}
        start = self.batch_size * index
        for kind in ['positive', 'negative']:
            pair_indexes[kind] = self.indexes[kind][start : start + self.batch_size]
        return pair_indexes

    def _build_index(self):
        indexes = {}
        indexes['positive'] = self._build_pairs(positive=True)
        indexes['negative'] = self._build_pairs(positive=False)
        return indexes

    def _get_data(self, indexes):
        positive_matches = self.X[indexes['positive']]
        negative_matches = self.X[indexes['negative']]
        samples = np.concatenate([positive_matches, negative_matches])
        labels = np.append(np.ones(len(positive_matches)), np.zeros(len(negative_matches)))
        assert len(negative_matches) * 2 == len(positive_matches) * 2 == len(labels)
        return samples, labels

    def _build_pairs(self, positive):
        first = True
        if self.shuffle:
            labels = np.random.permutation(self.label_set)
        else:
            labels = self.label_set
        for label in labels:
            same_idx = np.where(self.y == label)[0]
            ## DO POS OR NEG LOGIC
            if positive:  # Build positive pairs
                pairs = self._get_derangement(same_idx)
            else:  # Build negative pairs
                diff_idx = np.where(self.y != label)[0]
                replace = False
                if len(same_idx) > len(diff_idx):
                    replace = True
                match = np.random.choice(diff_idx, size=len(same_idx), replace=replace)
                pairs = np.array(list(zip(same_idx, match)))
            if not first:
                all_pairs = np.concatenate([all_pairs, pairs])
            else:
                first = False
                all_pairs = pairs
        return all_pairs

    def on_epoch_end(self):
        self.indexes = self._build_index()
        if self.shuffle:
            np.random.shuffle(self.indexes['positive'])
            np.random.shuffle(self.indexes['negative'])
            assert len(self.indexes['positive']) == len(self.indexes['negative'])


class SiameseGeneratorDF(SiameseGenerator):
    def __init__(self, X, y, batch_size, columns):
        super().__init__(X=X, y=y, batch_size=batch_size)
        self.columns = columns

    def _get_siamese_input_dict(self, left, right):
        di = {}
        for cat in self.columns:
            di[cat + '_left'] = np.array(left[cat])
            di[cat + '_right'] = np.array(right[cat])
        return di

    def _get_indexes(self, index):
        start = self.batch_size * index
        indexes = {}
        indexes['left'] = np.concatenate(
            [
                self.indexes['positive'][start : start + self.batch_size, 0],
                self.indexes['negative'][start : start + self.batch_size, 0],
            ]
        )
        indexes['right'] = np.concatenate(
            [
                self.indexes['positive'][start : start + self.batch_size, 1],
                self.indexes['negative'][start : start + self.batch_size, 1],
            ]
        )
        return indexes

    def _get_data(self, indexes):
        pairs = self._get_siamese_input_dict(
            left=self.X.iloc[indexes['left']], right=self.X.iloc[indexes['right']]
        )
        labels = np.append(np.ones(self.batch_size), np.zeros(self.batch_size))
        assert (
            len(self.X.iloc[indexes['right']]) == len(self.X.iloc[indexes['left']]) == len(labels)
        )
        return pairs, labels
