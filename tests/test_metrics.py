import torch
import unittest
import math
from src.metrics import recall, ndcg, recalls_and_ndcgs_for_ks, split_at_index

class TestMetrics(unittest.TestCase):
    def setUp(self):
        """设置测试数据"""
        # 基础测试数据
        self.scores = torch.tensor([
            [0.2, 0.5, 0.3, 0.7, 0.1],  # 用户1的预测分数
            [0.9, 0.4, 0.6, 0.3, 0.8]   # 用户2的预测分数
        ])
        self.labels = torch.tensor([
            [0, 1, 0, 1, 0],  # 用户1的真实标签（2个相关物品）
            [1, 0, 1, 0, 1]   # 用户2的真实标签（3个相关物品）
        ])
        # 边界情况测试数据
        self.zero_labels = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        # 大规模测试数据
        self.batch_size = 1000
        self.num_items = 500
        self.scores_large = torch.randn(self.batch_size, self.num_items)
        self.labels_large = torch.randint(0, 2, (self.batch_size, self.num_items)).float()
        # 确保每个用户至少有一个相关物品
        self.labels_large[:, 0] = 1
    
    def test_recall(self):
        """测试召回率计算"""
        # 测试K=3的情况
        recall_k3 = recall(self.scores, self.labels, 3)
        self.assertAlmostEqual(recall_k3, 1.0, delta=1e-4)
        # 测试K=2的情况
        recall_k2 = recall(self.scores, self.labels, 2)
        self.assertAlmostEqual(recall_k2, 5/6, delta=1e-4)  # (1 + 2/3)/2 = 5/6 ≈ 0.8333
    
    def test_ndcg(self):
        """测试NDCG计算"""
        # 测试K=3的情况
        ndcg_k3 = ndcg(self.scores, self.labels, 3)
        self.assertAlmostEqual(ndcg_k3, 1.0, delta=1e-4)
        # 测试K=2的情况
        ndcg_k2 = ndcg(self.scores, self.labels, 2)
        self.assertAlmostEqual(ndcg_k2, 1.0, delta=1e-4)
    
    def test_recalls_and_ndcgs_for_ks(self):
        """测试多K值指标计算"""
        metrics = recalls_and_ndcgs_for_ks(self.scores, self.labels, [2, 3, 5])
        # 验证Recall指标
        self.assertAlmostEqual(metrics['Recall@2'], 5/6, delta=1e-4)  # ≈0.8333
        self.assertAlmostEqual(metrics['Recall@3'], 1.0, delta=1e-4)
        self.assertAlmostEqual(metrics['Recall@5'], 1.0, delta=1e-4)
        # 验证NDCG指标
        self.assertAlmostEqual(metrics['NDCG@2'], 1.0, delta=1e-4)
        self.assertAlmostEqual(metrics['NDCG@3'], 1.0, delta=1e-4)
        self.assertAlmostEqual(metrics['NDCG@5'], 1.0, delta=1e-4)
    
    def test_zero_labels(self):
        """测试全零标签边界情况"""
        # 对于全零标签，召回率应为NaN
        recall_k2 = recall(self.scores, self.zero_labels, 2)
        self.assertTrue(math.isnan(recall_k2), "全零标签的召回率应为NaN")
        # 对于全零标签，NDCG应为0
        ndcg_k2 = ndcg(self.scores, self.zero_labels, 2)
        self.assertEqual(ndcg_k2, 0.0, "全零标签的NDCG应为0")
    
    def test_large_scale(self):
        """测试大规模数据"""
        ks = [5, 10, 20]
        metrics = recalls_and_ndcgs_for_ks(self.scores_large, self.labels_large, ks)
        # 验证指标在合理范围内
        for k in ks:
            recall_key = f'Recall@{k}'
            ndcg_key = f'NDCG@{k}'
            self.assertIn(recall_key, metrics)
            self.assertIn(ndcg_key, metrics)
            # 召回率应在0到1之间
            self.assertGreaterEqual(metrics[recall_key], 0.0)
            self.assertLessEqual(metrics[recall_key], 1.0)
            # NDCG应在0到1之间
            self.assertGreaterEqual(metrics[ndcg_key], 0.0)
            self.assertLessEqual(metrics[ndcg_key], 1.0)
    
    def test_split_at_index(self):
        """测试张量分割函数"""
        # 创建测试张量
        t = torch.arange(12).reshape(3, 4)  # 3x4张量
        # 在第1维索引2处分割
        left, right = split_at_index(1, 2, t)
        self.assertTrue(torch.equal(left, torch.tensor([[0, 1], [4, 5], [8, 9]])))
        self.assertTrue(torch.equal(right, torch.tensor([[2, 3], [6, 7], [10, 11]])))
        # 在第0维索引1处分割
        top, bottom = split_at_index(0, 1, t)
        self.assertTrue(torch.equal(top, torch.tensor([[0, 1, 2, 3]])))
        self.assertTrue(torch.equal(bottom, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])))
        # 测试3维张量
        t3d = torch.arange(24).reshape(2, 3, 4)
        split_dim = 2
        split_index = 1
        left3d, right3d = split_at_index(split_dim, split_index, t3d)
        self.assertEqual(left3d.shape, (2, 3, 1))
        self.assertEqual(right3d.shape, (2, 3, 3))