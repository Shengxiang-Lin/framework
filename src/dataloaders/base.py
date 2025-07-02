RAW_DATASET_ROOT_FOLDER = 'data'  # 定义原始数据集根目录路径
import pandas as pd
from tqdm import tqdm  # type: ignore # 用于显示进度条
tqdm.pandas()  # 使pandas操作支持进度条
from abc import *  # 抽象基类支持
from pathlib import Path  # 路径操作库
import pickle  # 对象序列化库

# 定义抽象数据集基类
class AbstractDataset(metaclass=ABCMeta):
    def __init__(self,
            target_behavior,  # 目标行为类型（如'buy'）
            multi_behavior,   # 是否使用多行为数据（True/False）
            min_uc            # 用户最小交互次数（过滤不活跃用户）
        ):
        # 初始化参数
        self.target_behavior = target_behavior
        self.multi_behavior = multi_behavior
        self.min_uc = min_uc
        self.bmap = None  # 行为类型映射字典
        # 验证参数有效性
        assert self.min_uc >= 2, '每个用户至少需要2个交互项用于验证和测试'
        self.split = 'leave_one_out'  # 数据集拆分策略（留一法）

    @classmethod
    @abstractmethod
    def code(cls):
        """抽象方法：返回数据集唯一标识符（子类必须实现）"""
        pass

    @classmethod
    def raw_code(cls):
        """返回原始数据集标识符（默认为code）"""
        return cls.code()

    @abstractmethod
    def load_df(self):
        """抽象方法：加载原始数据为DataFrame（子类必须实现）"""
        pass

    def load_dataset(self):
        """加载预处理后的数据集"""
        self.preprocess()  # 确保数据已预处理
        dataset_path = self._get_preprocessed_dataset_path()
        # 从pickle文件加载数据集
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        """预处理数据集（过滤、索引映射、拆分）"""
        dataset_path = self._get_preprocessed_dataset_path()
        # 检查是否已预处理过
        if dataset_path.is_file():
            print('数据已预处理，跳过预处理步骤')
            return
        # 确保预处理目录存在
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        # 加载原始数据
        df = self.load_df()
        # 行为选择（单行为/多行为）
        df = self.make_implicit(df)
        # 过滤不活跃用户
        df = self.filter_triplets(df)
        # 创建密集索引映射
        df, umap, smap, bmap = self.densify_index(df)
        self.bmap = bmap
        # 拆分数据集（训练集/验证集）
        train, train_b, val, val_b, val_num = self.split_df(df, len(umap))
        # 构建数据集字典
        dataset = {'train': train,     # 训练集物品序列
                   'val': val,         # 验证集物品序列
                   'train_b': train_b,  # 训练集行为序列
                   'val_b': val_b,      # 验证集行为序列
                   'val_num': val_num,  # 验证集用户数量
                   'umap': umap,       # 用户ID映射字典
                   'smap': smap,       # 物品ID映射字典
                   'bmap': bmap}       # 行为类型映射字典
        # 保存预处理后的数据集
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def make_implicit(self, df):
        """行为选择处理（根据是否使用多行为）"""
        print('行为选择处理')
        if self.multi_behavior:
            # 多行为模式：保留所有行为
            pass
        else:
            # 单行为模式：只保留目标行为
            df = df[df['behavior'] == self.target_behavior]
        return df

    def filter_triplets(self, df):
        """过滤交互次数不足的用户"""
        print('过滤用户交互三元组')
        if self.min_uc > 0:
            # 计算每个用户的交互次数
            user_sizes = df.groupby('uid').size()
            # 保留交互次数≥min_uc的用户
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    def densify_index(self, df):
        print('创建密集索引映射')
        # 获取唯一值并排序，以确保映射的一致性
        unique_users = sorted(set(df['uid']))
        unique_items = sorted(set(df['sid']))
        unique_behaviors = sorted(set(df['behavior']))
        # 创建用户ID映射 (原始ID → 连续整数)
        umap = {u: (i+1) for i, u in enumerate(unique_users)}
        # 创建物品ID映射 (原始ID → 连续整数)
        smap = {s: (i+1) for i, s in enumerate(unique_items)}
        # 创建行为类型映射 (行为名称 → 连续整数)
        bmap = {b: (i+1) for i, b in enumerate(unique_behaviors)}
        # 应用映射到DataFrame
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        df['behavior'] = df['behavior'].map(bmap)
        return df, umap, smap, bmap

    def split_df(self, df, user_count):
        """拆分数据集为训练集和验证集（留一法）"""      
        if self.split == 'leave_one_out':
            print('拆分数据集')
            # 按用户分组
            user_group = df.groupby('uid')
            # 获取每个用户的物品序列（保持原始顺序）
            user2items = user_group.progress_apply(lambda d: list(d['sid']))
            # 获取每个用户的行为序列（保持原始顺序）
            user2behaviors = user_group.progress_apply(lambda d: list(d['behavior']))
            # 初始化训练集和验证集字典
            train, train_b, val, val_b = {}, {}, {}, {}
            # 遍历所有用户
            for user in range(1, user_count+1):
                items = user2items[user]
                behaviors = user2behaviors[user]
                # 检查最后一个行为是否是目标行为
                if behaviors[-1] == self.bmap[self.target_behavior]:
                    # 目标行为用户：最后一个物品作为验证集
                    train[user], val[user] = items[:-1], items[-1:]
                    train_b[user], val_b[user] = behaviors[:-1], behaviors[-1:]
                else:
                    # 非目标行为用户：全部作为训练集
                    train[user] = items
                    train_b[user] = behaviors
            return train, train_b, val, val_b, len(val)
        else:
            raise NotImplementedError('只支持留一法拆分策略')

    def _get_rawdata_root_path(self):
        """获取原始数据集根路径"""
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_preprocessed_root_path(self):
        """获取预处理数据根路径"""
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        """获取预处理数据文件夹路径（包含参数信息）"""
        preprocessed_root = self._get_preprocessed_root_path()
        # 文件夹名称包含数据集参数信息
        folder_name = '{}-min_uc{}-target_B{}_MB{}-split{}' \
            .format(self.code(), self.min_uc, self.target_behavior, 
                    self.multi_behavior, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        """获取预处理数据集文件路径"""
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')  # 固定文件名