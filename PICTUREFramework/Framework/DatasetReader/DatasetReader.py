from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt


# Whenever this class is extended it has to follow
# the dataset structure rules as shown in the documentation
class DatasetReader(ABC):
    _dataset = None
    _url = ""
    _index = 0

    @abstractmethod
    def __init__(self, url):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    def set_url(self, url):
        self._url = url

    def get_next_sample(self):
        self._index += 1
        if (self._index - 1) < len(self._dataset):
            return self._dataset.iloc[self._index - 1]
        return None

    def get_n_next_samples(self, n):
        tmp = self._dataset.iloc[self._index:self._index + n]
        self._index += n
        return tmp

    def has_n_more_samples(self, n):
        if self._index + n < len(self._dataset):
            return True
        else:
            return len(self._dataset) - self._index + 1

    def reset_index(self):
        self._index = 0

    def plot(self):
        pass


# This class implement a dataset reader
# for the specific dataset Accountant
# taken from Kaggle
class TimeSeriesAccountantDataset(DatasetReader):
    def __init__(self, url):
        self.set_url(url)
        data = pd.read_csv(self._url + ".csv")
        data['Date'] = pd.to_datetime(data['Date'])
        s1p1 = data[(data['store'] == 0) & (data['product'] == 0)]
        s1p1 = s1p1.drop('store', axis=1)
        s1p1 = s1p1.drop('product', axis=1)
        s1p1 = s1p1.drop('Date', axis=1)
        s1p1.index.name = None
        # index = []
        # for i in range(0, len(s1p1)):
        #     index.append(i)
        # s1p1.insert(0, "time", index)
        self._dataset = s1p1
        self._index = 0

    def get_dataset(self) -> pd.DataFrame:
        return self._dataset

    def plot(self):
        self._dataset.plot(x='time', y='number_sold', color="b", figsize=(16, 2))
        plt.title("Synthetic Accountant Dataset")
        plt.show()

    def get_title(self):
        return 'Accountant Dataset'


class TimeSeriesAccelerometersDataset(DatasetReader):
    def __init__(self, dataset_name, offset=0):
        # data = pd.read_csv(dataset_name + ".csv", index_col=0, nrows=10000)
        data = pd.read_csv(dataset_name + ".csv", index_col=0)
        index = []
        for i in range(0, len(data)):
            index.append(i)
        # data.insert(0, "time", index)
        data = data.drop(['AccY', 'AccZ'], axis=1)
        data = data.rename(columns={'AccX': 'number_sold'})
        data['number_sold'] += offset
        data['number_sold'] *= 100
        self._dataset = data
        self._index = 0

    def get_dataset(self):
        return self._dataset

    def get_title(self):
        return 'Accelerometer Dataset'