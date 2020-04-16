import abc


class Dataset(abc.ABC):
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def num_features(self):
        return self.x.shape[1]
