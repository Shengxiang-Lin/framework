import pandas as pd
from pathlib import Path
import pickle
from abc import *
import numpy as np
import torch
from torch.utils.data import DataLoader

RAW_DATASET_ROOT_FOLDER = '../data'

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self,
                 target_behavior,
                 multi_behavior,
                 min_uc):
        self.target_behavior = target_behavior
        self.multi_behavior = multi_behavior
        self.min_uc = min_uc
        self.bmap = None
        assert self.min_uc >= 2, 'Need at least 2 items per user for validation and test'
        self.split = 'leave_one_out'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def load_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        df = self.load_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap, bmap = self.densify_index(df)
        self.bmap = bmap
        train, train_b, val, val_b, val_num = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'train_b': train_b,
                   'val_b': val_b,
                   'val_num': val_num,
                   'umap': umap,
                   'smap': smap,
                   'bmap': bmap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def make_implicit(self, df):
        print('Behavior selection')
        if self.multi_behavior:
            pass
        else:
            df = df[df['behavior'] == self.target_behavior]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: (i + 1) for i, u in enumerate(set(df['uid']))}
        smap = {s: (i + 1) for i, s in enumerate(set(df['sid']))}
        bmap = {b: (i + 1) for i, b in enumerate(set(df['behavior']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        df['behavior'] = df['behavior'].map(bmap)
        return df, umap, smap, bmap

    def split_df(self, df, user_count):
        if self.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.apply(lambda d: list(d['sid']))
            user2behaviors = user_group.apply(lambda d: list(d['behavior']))
            train, train_b, val, val_b = {}, {}, {}, {}
            for user in range(1, user_count + 1):
                items = user2items[user]
                behaviors = user2behaviors[user]
                if behaviors[-1] == self.bmap[self.target_behavior]:
                    train[user], val[user] = items[:-1], items[-1:]
                    train_b[user], val_b[user] = behaviors[:-1], behaviors[-1:]
                else:
                    train[user] = items
                    train_b[user] = behaviors
            return train, train_b, val, val_b, len(val)
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}-min_uc{}-target_B{}_MB{}-split{}' \
            .format(self.code(), self.min_uc, self.target_behavior, self.multi_behavior, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')


class RetailDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'retail'

    def load_df(self):
        folder_path = self._get_rawdata_root_path()
        file_path = folder_path.joinpath('retail.txt')
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'sid', 'behavior', 'timestamp']
        return df


class IjcaiDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ijcai'

    def load_df(self):
        folder_path = self._get_rawdata_root_path()
        file_path = folder_path.joinpath('ijcai.txt')
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'sid', 'behavior', 'timestamp']
        return df


class YelpDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'yelp'

    def load_df(self):
        folder_path = self._get_rawdata_root_path()
        file_path = folder_path.joinpath('yelp.txt')
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'sid', 'behavior', 'timestamp']
        return df


DATASETS = {
    RetailDataset.code(): RetailDataset,
    IjcaiDataset.code(): IjcaiDataset,
    YelpDataset.code(): YelpDataset
}


def dataset_factory(dataset_code, target_behavior, multi_behavior, min_uc):
    dataset = DATASETS[dataset_code]
    return dataset(target_behavior, multi_behavior, min_uc)


# 示例使用
if __name__ == "__main__":
    dataset_code = 'ijcai'  # 可以替换为 'retail' 或 'yelp'
    target_behavior = 'buy'
    multi_behavior = True
    min_uc = 3

    dataset = dataset_factory(dataset_code, target_behavior, multi_behavior, min_uc)
    processed_data = dataset.load_dataset()
    print(processed_data.keys())