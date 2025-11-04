# cluster_mclust.py
import numpy as np
import warnings

# 安装 rpy2 和 R 包 mclust（首次运行时）
# !pip install rpy2
# 在 R 中执行：install.packages("mclust")

def cluster_mclust(embeddings, n_clusters=6, modelNames='EEE', random_seed=2020):
    """
    使用 R 中的 mclust 对 Python 中的 embedding 进行聚类。

    参数：
        embeddings: numpy array，形状为 (n_cells, n_features)
        n_clusters: 聚类数目
        modelNames: mclust 模型类型，默认为 'EEE'
        random_seed: 随机种子

    返回：
        cluster_labels: np.array，聚类标签，0-indexed
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        robjects.r.library("mclust")
    except Exception as e:
        raise ImportError("请确保已正确安装 rpy2 和 R 包 mclust。错误信息：" + str(e))

    # 设置随机种子
    robjects.r['set.seed'](random_seed)

    # 传入 R 并执行 mclust 聚类
    rmclust = robjects.r['Mclust']
    res = rmclust(numpy2ri.numpy2rpy(embeddings), n_clusters, modelNames)

    # 提取分类结果
    cluster_labels = np.array(res.rx2('classification')).astype(int) - 1  # mclust 从 1 开始编号，改为 0 开始
    return cluster_labels
