"""
Factorization Machine implemented with PyTorch
"""

__all__ = [
    "FactorizationMachine", "FMBQM"
]

import torch
import torch.nn as nn
import numpy as np


def triu_mask(input_size, device=None):
    """
    產生一個上三角為1(不包含對角線)，其餘為0，
    這個輔助函數只有在你想要查看 Q 矩陣時才需要，
    在高效的前向傳播中並不會用到。
    """
    mask = torch.ones(input_size, input_size, device=device)
    return mask.triu(diagonal=1)

def VtoQ(V):
    """從 V 計算 Q 矩陣: Q = V^T V 並套用上三角遮罩"""
    Q = torch.matmul(V.T, V)
    # 套用triu_mask，只保留上三角部分(不含對角線)
    return Q * triu_mask(Q.size(0), device=Q.device)

class QuadraticLayer(nn.Module):
    """
    帶有二次項的模型的基底類別。
    """
    def __init__(self):
        super().__init__()

    def init_params(self):
        """初始化模型參數"""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif param.dim() > 1:
                # 對於權重或嵌入矩陣使用常見的初始化方法
                nn.init.xavier_uniform_(param)
            else:
                # 對於像 'h' 這樣的一維權重向量
                nn.init.normal_(param, mean=0, std=0.01)

    def get_bhQ(self):
        raise NotImplementedError()

class FactorizationMachine(QuadraticLayer):
    """
    複雜度為 O(dk) 的Forward Propagation FM。

    Args:
        input_size (int): 特徵數量 (d)。
        factorization_size (int): 隱藏因子的維度 (k)。
        act (str, optional): 激活函數 ("identity", "sigmoid", "tanh")。預設為 "identity"。
    """
    def __init__(self, input_size, factorization_size=8, act="identity"):
        super().__init__()
        if factorization_size <= 0:
            raise ValueError("factorization_size 必須是正整數。")
            
        self.input_size = input_size
        self.factorization_size = factorization_size
        self.act_name = act

        # 線性項參數
        self.h = nn.Parameter(torch.empty(input_size))
        # 偏置項或偏移量
        self.bias = nn.Parameter(torch.empty(1))
        # 交互項參數 (k,d) 
        self.V = nn.Parameter(torch.empty(factorization_size, input_size))

        self.init_params() # 創建時即初始化參數

    def forward(self, x):
        """
        使用複雜度 O(dk) 方法計算 FM 輸出。
        x: (batch_size, input_size)
        """
        # 線性項: b + h^T * x
        linear_term = self.bias + torch.matmul(x, self.h)

        # 交互項: 0.5 * sum_f [ (sum_i v_if * x_i)^2 - sum_i (v_if^2 * x_i^2) ]
        # 得到 (batch_size, k)
        interaction_part1 = torch.matmul(x, self.V.T)
        # 得到 (batch_size, k)
        interaction_part2 = torch.matmul(x**2, (self.V**2).T)
        
        # 在因子維度 k 上求和
        interaction_term = 0.5 * torch.sum(interaction_part1**2 - interaction_part2, dim=1)

        out = linear_term + interaction_term

        # 套用激活函數
        if self.act_name == "sigmoid":
            return torch.sigmoid(out).view(-1, 1)
        elif self.act_name == "tanh":
            return torch.tanh(out).view(-1, 1)
        else:
            return out.view(-1, 1)

    def get_bhQ(self):
        """
        返回線性 (h), 偏置 (b) 和二次 (Q) 項的係數。
        主要用於模型檢查，而不是用於前向傳播。
        """
        V_data = self.V.detach().cpu()
        Q = VtoQ(V_data)
        
        bias = self.bias.detach().cpu().item()
        h = self.h.detach().cpu().numpy()
        
        return bias, h, Q.numpy()


FactorizationMachineBinaryQuadraticModel = FactorizationMachine
FMBQM = FactorizationMachine
