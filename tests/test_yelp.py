import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import patch
from src.datasets.yelp import YelpDataset

# 测试数据样例 - 使用少量数据
YELP_DATA = """123\t456\tclick\t1577836800
123\t789\tbuy\t1577836801
456\t101\tclick\t1577836802
456\t102\tbuy\t1577836803
"""

def test_yelp_dataset_code():
    """测试 YelpDataset 的 code() 方法"""
    yelp = YelpDataset(target_behavior='buy', multi_behavior=True, min_uc=2)
    assert yelp.code() == 'yelp'

def test_yelp_dataset_raw_code():
    """测试 YelpDataset 的 raw_code() 方法"""
    yelp = YelpDataset(target_behavior='buy', multi_behavior=True, min_uc=2)
    assert yelp.raw_code() == 'yelp'

def test_load_df_with_mock(tmp_path):
    """测试 load_df() 方法使用模拟数据"""
    # 创建临时文件
    file_path = tmp_path / 'yelp.txt'
    file_path.write_text(YELP_DATA)
    # 模拟原始数据根路径
    with patch.object(YelpDataset, '_get_rawdata_root_path') as mock_get_path:
        mock_get_path.return_value = tmp_path
        yelp = YelpDataset(target_behavior='buy', multi_behavior=True, min_uc=2)
        df = yelp.load_df()
        # 验证数据框结构
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['uid', 'sid', 'behavior', 'timestamp']
        assert len(df) == 4
        # 验证具体数据
        assert df.iloc[0]['uid'] == 123
        assert df.iloc[0]['sid'] == 456
        assert df.iloc[0]['behavior'] == 'click'

def test_load_df_with_real_file(tmp_path):
    """测试 load_df() 方法使用真实文件"""
    # 创建数据文件
    file_path = tmp_path / 'yelp.txt'
    file_path.write_text(YELP_DATA)
    # 模拟原始数据根路径
    with patch.object(YelpDataset, '_get_rawdata_root_path') as mock_get_path:
        mock_get_path.return_value = tmp_path
        yelp = YelpDataset(target_behavior='buy', multi_behavior=True, min_uc=2)
        df = yelp.load_df()
        # 验证数据
        assert len(df) == 4
        assert list(df['uid'].unique()) == [123, 456]

def test_file_not_found(tmp_path):
    """测试当文件不存在时的行为"""
    # 确保文件不存在
    file_path = tmp_path / 'yelp.txt'
    if file_path.exists():
        file_path.unlink()
    # 模拟原始数据根路径
    with patch.object(YelpDataset, '_get_rawdata_root_path') as mock_get_path:
        mock_get_path.return_value = tmp_path
        yelp = YelpDataset(target_behavior='buy', multi_behavior=True, min_uc=2)
        # 验证加载数据会引发异常
        with pytest.raises(FileNotFoundError):
            yelp.load_df()

def test_yelp_dataset_integration(tmp_path):
    """测试 YelpDataset 的完整集成"""
    # 创建数据文件
    file_path = tmp_path / 'yelp.txt'
    file_path.write_text(YELP_DATA)
    # 模拟原始数据根路径
    with patch.object(YelpDataset, '_get_rawdata_root_path') as mock_get_path:
        mock_get_path.return_value = tmp_path
        yelp = YelpDataset(target_behavior='buy', multi_behavior=True, min_uc=2)
        # 加载原始数据
        df = yelp.load_df()
        assert len(df) == 4
        # 预处理数据
        yelp.preprocess()
        # 加载预处理后的数据集
        dataset = yelp.load_dataset()
        # 验证数据集结构
        assert 'train' in dataset
        assert 'val' in dataset
        assert 'umap' in dataset
        assert 'smap' in dataset
        assert 'bmap' in dataset
        # 验证用户映射 - 2个用户
        assert len(dataset['umap']) == 2
        # 验证物品映射 - 4个物品
        assert len(dataset['smap']) == 4
        # 验证行为映射 - 2个行为
        assert len(dataset['bmap']) == 2
        # 验证训练集和验证集
        assert len(dataset['train']) == 2  # 2个用户
        assert len(dataset['val']) == 2    # 2个用户都有验证项