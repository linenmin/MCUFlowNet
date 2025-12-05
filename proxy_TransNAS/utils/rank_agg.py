import numpy as np  # 数值计算
from scipy.stats import rankdata  # 排名计算


def nonlinear_ranking_aggregation(proxy_scores_dict, skip_constant=False):  # 非线性排名聚合
    """基于 AZ-NAS 的对数排名聚合。"""  # 函数说明
    m = len(next(iter(proxy_scores_dict.values())))  # 架构数量
    aggregated_scores = np.zeros(m)  # 初始化聚合分数
    skipped_proxies = []  # 记录被跳过的常数 proxy
    print(f"\n=== 非线性排名聚合 ===")  # 提示信息
    print(f"架构数量: {m}")  # 打印架构数量
    print(f"Proxy 数量: {len(proxy_scores_dict)}")  # 打印 proxy 数量
    for proxy_name, scores in proxy_scores_dict.items():  # 遍历每个 proxy
        scores_array = np.array(scores)  # 转为数组
        score_std = np.std(scores_array)  # 计算标准差
        score_range = np.max(scores_array) - np.min(scores_array)  # 计算范围
        print(f"  {proxy_name}: 分数范围 [{np.min(scores_array):.4f}, {np.max(scores_array):.4f}], 标准差={score_std:.6f}")  # 打印统计
        if skip_constant and (score_std < 1e-10 or score_range < 1e-10):  # 常数 proxy 判定
            print(f"    ⚠️  警告：{proxy_name} 是常数 proxy，跳过聚合")  # 提示跳过
            skipped_proxies.append(proxy_name)  # 记录跳过
            continue  # 继续下一个 proxy
        ranks = rankdata(scores_array, method="ordinal")  # 计算排名
        for i in range(m):  # 遍历每个架构
            aggregated_scores[i] += np.log(ranks[i] / m)  # 累加对数排名
    if skipped_proxies:  # 若有跳过
        print(f"  已跳过常数 proxy: {', '.join(skipped_proxies)}")  # 打印列表
        print(f"  实际参与聚合的 proxy 数量: {len(proxy_scores_dict) - len(skipped_proxies)}")  # 打印数量
    print(f"聚合分数范围: [{np.min(aggregated_scores):.4f}, {np.max(aggregated_scores):.4f}]")  # 打印聚合范围
    return aggregated_scores.tolist()  # 返回列表

