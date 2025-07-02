import pandas as pd
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.dataloaders.base import AbstractDataset, RAW_DATASET_ROOT_FOLDER

# 创建一个具体的子类用于测试
class MockDataset(AbstractDataset):
    def code(self):
        return "mock"
    
    def load_df(self):
        # 创建模拟数据
        data = {
            'uid': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            'sid': [101, 102, 103, 101, 104, 105, 102, 103, 106, 107, 108],
            'behavior': ['click', 'cart', 'buy', 'click', 'cart', 'buy', 'click', 'click', 'click', 'cart', 'buy'],
            'timestamp': [100, 200, 300, 100, 200, 300, 100, 200, 100, 200, 300]
        }
        return pd.DataFrame(data)

def test_make_implicit_single_behavior(tmp_path):
    # 测试单行为模式
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        dataset = MockDataset(
            target_behavior='buy',
            multi_behavior=False,
            min_uc=2
        )
        df = pd.DataFrame({
            'uid': [1, 1, 2, 2],
            'behavior': ['click', 'buy', 'click', 'buy']
        })
        result = dataset.make_implicit(df)
        # 应只保留目标行为
        assert len(result) == 2
        assert all(result['behavior'] == 'buy')

def test_make_implicit_multi_behavior(tmp_path):
    # 测试多行为模式
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        dataset = MockDataset(
            target_behavior='buy',
            multi_behavior=True,
            min_uc=2
        )
        df = pd.DataFrame({
            'uid': [1, 1, 2, 2],
            'behavior': ['click', 'buy', 'click', 'buy']
        })
        result = dataset.make_implicit(df)
        # 应保留所有行为
        assert len(result) == 4

def test_filter_triplets(tmp_path):
    # 测试用户过滤
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        dataset = MockDataset(
            target_behavior='buy',
            multi_behavior=True,
            min_uc=2
        )
        df = pd.DataFrame({
            'uid': [1, 1, 2, 2, 3, 4, 4, 4],
            'sid': [101, 102, 101, 102, 103, 104, 105, 106]
        })
        result = dataset.filter_triplets(df)
        # 用户3（只有1次交互）应被过滤
        assert set(result['uid'].unique()) == {1, 2, 4}

def test_densify_index(tmp_path):
    # 测试索引密集化
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        dataset = MockDataset(
            target_behavior='buy',
            multi_behavior=True,
            min_uc=2
        )
        df = pd.DataFrame({
            'uid': [10, 10, 20, 20],
            'sid': [101, 102, 101, 103],
            'behavior': ['click', 'buy', 'click', 'cart']
        })
        result, umap, smap, bmap = dataset.densify_index(df)
        # 检查映射是否正确（排序后）
        assert umap == {10: 1, 20: 2}
        assert smap == {101: 1, 102: 2, 103: 3}  # 排序后
        assert bmap == {'buy': 1, 'cart': 2, 'click': 3}  # 排序后
        # 检查DataFrame是否被转换
        assert list(result['uid']) == [1, 1, 2, 2]
        assert list(result['sid']) == [1, 2, 1, 3]
        assert list(result['behavior']) == [3, 1, 3, 2]  # 映射后

def test_split_df(tmp_path):
    # 准备测试数据
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        dataset = MockDataset(
            target_behavior='buy',
            multi_behavior=True,
            min_uc=2
        )
        df = pd.DataFrame({
            'uid': [1, 1, 1, 2, 2, 3, 3],
            'sid': [101, 102, 103, 101, 104, 102, 103],
            'behavior': ['click', 'cart', 'buy', 'click', 'buy', 'click', 'click']
        })
        user_count = 3
        densified, umap, smap, bmap = dataset.densify_index(df)
        dataset.bmap = bmap
        # 执行拆分
        train, train_b, val, val_b, val_num = dataset.split_df(densified, user_count)
        # 验证结果
        assert val_num == 2  # 用户1和用户2有目标行为结尾
        # 获取物品ID映射（排序后）
        item_mapping = {v: k for k, v in smap.items()}
        # 用户1的验证项
        assert item_mapping[val[1][0]] == 103  # 验证原始ID
        # 用户2的验证项
        assert item_mapping[val[2][0]] == 104  # 验证原始ID

def test_full_preprocess_flow(tmp_path):
    # 使用临时路径
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        dataset = MockDataset(
            target_behavior='buy',
            multi_behavior=True,
            min_uc=2
        )
        # 使用dataset的方法获取预处理文件夹路径
        preprocessed_folder = dataset._get_preprocessed_folder_path()
        # 确保预处理目录不存在
        assert not preprocessed_folder.exists()
        # 执行预处理
        dataset.preprocess()
        # 检查预处理文件是否创建
        dataset_file = dataset._get_preprocessed_dataset_path()
        assert dataset_file.exists()
        # 加载预处理数据
        with dataset_file.open('rb') as f:
            dataset_data = pickle.load(f)
        # 验证基本结构
        assert 'train' in dataset_data
        assert 'val' in dataset_data
        assert 'umap' in dataset_data
        assert 'smap' in dataset_data
        assert 'bmap' in dataset_data

def test_load_preprocessed_dataset(tmp_path):
    # 使用临时路径
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        dataset = MockDataset(
            target_behavior='buy',
            multi_behavior=True,
            min_uc=2
        )
        # 先执行预处理
        dataset.preprocess()
        # 测试加载预处理数据
        dataset_data = dataset.load_dataset()
        # 验证基本结构
        assert isinstance(dataset_data, dict)
        assert 'train' in dataset_data
        assert 'val' in dataset_data

def test_min_uc_validation(tmp_path):
    # 测试min_uc参数验证
    with patch("src.dataloaders.base.RAW_DATASET_ROOT_FOLDER", str(tmp_path)):
        # 测试min_uc参数验证
        exception_raised = False
        exception_message = ""
        try:
            # 这行应该引发断言错误
            MockDataset(target_behavior='buy', multi_behavior=True, min_uc=1)
        except AssertionError as e:
            exception_raised = True
            exception_message = str(e)
        # 验证是否引发了异常
        assert exception_raised, "预期会引发 AssertionError 但没有发生"
        # 验证异常消息
        assert "每个用户至少需要2个交互项" in exception_message