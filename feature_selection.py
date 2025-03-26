import pandas as pd
import numpy as np

def remove_highly_correlated_features(X, threshold=0.95):
    """
    移除特征中高度相关的列（默认相关系数大于 0.95）。
    参数:
        X - pandas DataFrame，原始特征数据
        threshold - float，相关性阈值，超过这个值的特征将被移除
    返回:
        reduced_X - 删除冗余后的 DataFrame
        dropped_features - 被删除的列名列表
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
                     columns=corr_matrix.columns,
                     index=corr_matrix.index)
    )

    # 找到高相关特征对
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    reduced_X = X.drop(columns=to_drop)

    return reduced_X, to_drop
