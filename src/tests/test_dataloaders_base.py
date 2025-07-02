import unittest
import os
import shutil
import tempfile
import pandas as pd
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入要测试的基类
from src.dataloaders import AbstractDataset, RAW_DATASET_ROOT_FOLDER
# 创建测试用的具体数据集类
class MockDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return "mock_dataset"
    
    def load_df(self):
        # 创建测试数据
        data = {
            'uid': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4],
            'sid': [101, 102, 103, 201, 202, 203, 301, 302, 401, 402, 403, 404],
            'behavior': ['pv', 'fav', 'buy', 'pv', 'cart', 'buy', 'pv', 'pv', 'fav', 'cart', 'buy', 'pv'],
            'timestamp': [1000, 1001, 1002, 2000, 2001, 2002, 3000, 3001, 4000, 4001, 4002, 4003]
        }
        return pd.DataFrame(data)

class TestAbstractDataset(unittest.TestCase):
    def setUp(self):
        # 创建临时目录作为数据根目录
        self.test_dir = tempfile.mkdtemp()
        
        # 先声明全局变量
        global RAW_DATASET_ROOT_FOLDER
        self.orig_raw_data_root = RAW_DATASET_ROOT_FOLDER  # 保存原始值
        RAW_DATASET_ROOT_FOLDER = self.test_dir  # 修改为临时目录
        
        # 创建原始数据目录结构
        os.makedirs(os.path.join(self.test_dir, 'mock_dataset'))
    
    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)
        
        # 恢复原始数据根目录
        global RAW_DATASET_ROOT_FOLDER
        RAW_DATASET_ROOT_FOLDER = self.orig_raw_data_root
    
    def test_code_abstract_method(self):
        """测试抽象方法code()必须由子类实现"""
        with self.assertRaises(TypeError):
            # 尝试实例化抽象类应该引发错误
            dataset = AbstractDataset(target_behavior='buy', multi_behavior=True, min_uc=2)
    
    def test_load_dataset_with_preprocessing(self):
        """测试数据集加载（包含预处理）"""
        dataset = MockDataset(target_behavior='buy', multi_behavior=False, min_uc=2)
        
        # 加载数据集（应该触发预处理）
        result = dataset.load_dataset()
        
        # 验证预处理目录已创建
        preprocessed_path = Path(self.test_dir) / 'preprocessed' / 'mock_dataset-min_uc2-target_Bbuy_MBFalse-splitleave_one_out'
        self.assertTrue(preprocessed_path.is_dir())
        
        # 验证数据集文件已创建
        dataset_path = preprocessed_path / 'dataset.pkl'
        self.assertTrue(dataset_path.is_file())
        
        # 验证数据集内容
        self.assertIn('train', result)
        self.assertIn('val', result)
        self.assertIn('train_b', result)
        self.assertIn('val_b', result)
        self.assertIn('umap', result)
        self.assertIn('smap', result)
        self.assertIn('bmap', result)
        
        # 验证用户数量
        self.assertEqual(len(result['umap']), 2)  # 过滤后应该有2个用户（用户1和用户2）
    
    def test_make_implicit_single_behavior(self):
        """测试单行为模式下的行为选择"""
        dataset = MockDataset(target_behavior='buy', multi_behavior=False, min_uc=0)
        df = dataset.load_df()
        
        # 应用行为选择
        filtered_df = dataset.make_implicit(df)
        
        # 验证只保留了目标行为
        self.assertEqual(len(filtered_df), 3)
        self.assertTrue((filtered_df['behavior'] == 'buy').all())
    
    def test_make_implicit_multi_behavior(self):
        """测试多行为模式下的行为选择"""
        dataset = MockDataset(target_behavior='buy', multi_behavior=True, min_uc=0)
        df = dataset.load_df()
        
        # 应用行为选择
        filtered_df = dataset.make_implicit(df)
        
        # 验证保留了所有行为
        self.assertEqual(len(filtered_df), len(df))
        self.assertEqual(set(filtered_df['behavior']), {'pv', 'fav', 'cart', 'buy'})
    
    def test_filter_triplets(self):
        """测试用户过滤功能"""
        dataset = MockDataset(target_behavior='buy', multi_behavior=True, min_uc=3)
        df = dataset.load_df()
        
        # 应用用户过滤
        filtered_df = dataset.filter_triplets(df)
        
        # 验证用户过滤结果
        self.assertEqual(len(filtered_df), 8)  # 用户1:3, 用户2:3, 用户4:4 -> 总计10条，但用户3只有2条被过滤
        self.assertEqual(set(filtered_df['uid']), {1, 2, 4})
    
    def test_densify_index(self):
        """测试索引映射功能"""
        dataset = MockDataset(target_behavior='buy', multi_behavior=True, min_uc=0)
        df = dataset.load_df()
        
        # 应用索引映射
        df, umap, smap, bmap = dataset.densify_index(df)
        
        # 验证用户映射
        self.assertEqual(len(umap), 4)
        self.assertEqual(df['uid'].min(), 1)
        self.assertEqual(df['uid'].max(), 4)
        
        # 验证物品映射
        self.assertEqual(len(smap), 8)
        self.assertEqual(df['sid'].min(), 1)
        self.assertEqual(df['sid'].max(), 8)
        
        # 验证行为映射
        self.assertEqual(len(bmap), 4)
        self.assertEqual(set(bmap.keys()), {'pv', 'fav', 'cart', 'buy'})
    
    def test_split_df(self):
        """测试数据集拆分功能"""
        dataset = MockDataset(target_behavior='buy', multi_behavior=True, min_uc=0)
        df = dataset.load_df()
        
        # 应用索引映射
        df, umap, smap, bmap = dataset.densify_index(df)
        dataset.bmap = bmap
        
        # 拆分数据集
        train, train_b, val, val_b, val_num = dataset.split_df(df, len(umap))
        
        # 验证验证集数量
        self.assertEqual(val_num, 2)  # 用户1和用户2有目标行为作为最后一项
        
        # 验证用户1的数据
        self.assertEqual(len(train[1]), 2)
        self.assertEqual(len(val[1]), 1)
        self.assertEqual(train_b[1], [bmap['pv'], bmap['fav']])
        self.assertEqual(val_b[1], [bmap['buy']])
        
        # 验证用户3的数据（没有目标行为作为最后一项）
        self.assertEqual(len(train[3]), 2)
        self.assertNotIn(3, val)  # 用户3不在验证集中
    
    def test_preprocessed_path_generation(self):
        """测试预处理路径生成逻辑"""
        dataset = MockDataset(target_behavior='buy', multi_behavior=True, min_uc=5)
        
        # 获取预处理路径
        preprocessed_folder = dataset._get_preprocessed_folder_path()
        dataset_path = dataset._get_preprocessed_dataset_path()
        
        # 验证路径格式
        expected_folder = Path(self.test_dir) / 'preprocessed' / 'mock_dataset-min_uc5-target_Bbuy_MBTrue-splitleave_one_out'
        self.assertEqual(str(preprocessed_folder), str(expected_folder))
        self.assertEqual(str(dataset_path), str(expected_folder / 'dataset.pkl'))
    
    @patch('dataloaders.base.pickle.load')
    @patch('dataloaders.base.Path.is_file', return_value=True)
    def test_load_dataset_without_preprocessing(self, mock_is_file, mock_pickle_load):
        """测试跳过预处理直接加载数据集"""
        # 创建模拟数据集
        mock_dataset = {'train': {}, 'val': {}}
        mock_pickle_load.return_value = mock_dataset
        
        dataset = MockDataset(target_behavior='buy', multi_behavior=False, min_uc=2)
        
        # 加载数据集
        result = dataset.load_dataset()
        
        # 验证跳过了预处理
        mock_is_file.assert_called()
        mock_pickle_load.assert_called()
        self.assertEqual(result, mock_dataset)

if __name__ == '__main__':
    unittest.main(verbosity=2)