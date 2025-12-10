import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 后端设为非交互式，方便脚本运行
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)


try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# ---------------------------
# 数据读取与预处理
# ---------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """
    读取数据文件，支持：
      - .json: 格式为 list[dict]，每个元素包含 "id", "content", "label"
      - .jsonl: 每行一个 json 对象，包含同样字段

    返回：DataFrame，至少包含 'id', 'content', 'label'
    """
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(obj)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        raise ValueError(f"Unsupported input file format: {path}")

    df = pd.DataFrame(records)
    if not {"id", "content", "label"}.issubset(df.columns):
        raise ValueError("Input data must contain columns: 'id', 'content', 'label'.")

    # 确保类型正确
    df["id"] = df["id"].astype(str)
    df["content"] = df["content"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


# ---------------------------
# Embedding 计算
# ---------------------------

def compute_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True
) -> np.ndarray:
    """
    使用 SentenceTransformer 计算文本向量。

    参数：
      - model: 已加载好的 SentenceTransformer 模型
      - texts: 文本列表
      - batch_size: batch 大小
      - normalize: 是否进行 L2 归一化（bge 系列推荐归一化，并配合余弦度量）

    返回：
      - embeddings: (N, d) 的 numpy 数组
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return emb


# ---------------------------
# 降维（UMAP / t-SNE）
# ---------------------------

def reduce_dimension(
    embeddings: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
) -> np.ndarray:
    """
    将高维向量降到 2 维，方便可视化。

    method:
      - "umap"
      - "tsne"
      - "none": 直接返回 None
    """
    if method == "none":
        return None

    if method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Please `pip install umap-learn`.")
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",  # bge 向量推荐使用 cos 距离
            random_state=random_state,
        )
        coords = reducer.fit_transform(embeddings)
        return coords

    if method == "tsne":
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            max_iter=tsne_n_iter,
            metric="cosine",
            random_state=random_state,
            init="random",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(embeddings)
        return coords

    raise ValueError(f"Unknown dimensionality reduction method: {method}")


# ---------------------------
# 聚类（K-Means / HDBSCAN）
# ---------------------------

def run_clustering(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 10,
    hdbscan_min_cluster_size: int = 15,
    hdbscan_min_samples: int = 5,
) -> np.ndarray:
    """
    对 embeddings 做聚类，返回 cluster labels（长度为 N 的整数数组）。

    method:
      - "kmeans"
      - "hdbscan"
    """
    if method == "kmeans":
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0 for KMeans.")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto"
        )
        clusters = kmeans.fit_predict(embeddings)
        return clusters

    if method == "hdbscan":
        if not HAS_HDBSCAN:
            raise ImportError("hdbscan is not installed. Please `pip install hdbscan`.")
        # 使用欧式距离或余弦距离都可以；此处用 'euclidean'，因为前面已归一化
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
        )
        clusters = clusterer.fit_predict(embeddings)
        return clusters

    raise ValueError(f"Unknown clustering method: {method}")


# ---------------------------
# 可视化
# ---------------------------

def plot_scatter(
    df_view: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    save_path: str,
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.7,
    s: int = 6,
    metrics: Dict[str, float] = None,  # 新增：聚类指标字典
):
    """
    通用散点图绘制函数。
    df_view: DataFrame，必须包含 x_col, y_col, color_col
    color_col: 用于着色的列（可以是 cluster 或 label）
    metrics: 可选，用于在图中显示的聚类指标，例如：
        {
            "silhouette": 0.31,
            "calinski_harabasz": 123.4,
            "davies_bouldin": 0.85,
            "nmi": 0.42,
            "ari": 0.36
        }
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # 如果 color_col 为类别型（字符串），映射到整数 id
    unique_vals = df_view[color_col].astype(str).unique()
    val2id = {v: i for i, v in enumerate(sorted(unique_vals))}
    colors = df_view[color_col].astype(str).map(val2id)

    scatter = ax.scatter(
        df_view[x_col],
        df_view[y_col],
        c=colors,
        s=s,
        alpha=alpha,
        cmap="tab20",
    )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    # 构建 legend
    handles = []
    labels = []
    for v, idx in val2id.items():
        handles.append(plt.Line2D(
            [], [], marker="o", linestyle="",
            color=scatter.cmap(scatter.norm(idx))
        ))
        labels.append(v)
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # 如果传入了 metrics，则在图的右上角绘制一块文字区域
    if metrics is not None and len(metrics) > 0:
        # 把 NaN 处理成 "nan"，并控制小数位
        lines = []
        # 为了控制展示顺序，这里手动排一下常见 key
        order_keys = [
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "nmi",
            "ari",
        ]
        # 先按预设顺序展示，再补上其他可能的 key
        already = set()
        for k in order_keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    line = f"{k}: {v:.4f}" if not np.isnan(v) else f"{k}: nan"
                else:
                    line = f"{k}: {v}"
                lines.append(line)
                already.add(k)
        for k, v in metrics.items():
            if k in already:
                continue
            if isinstance(v, float):
                line = f"{k}: {v:.4f}" if not np.isnan(v) else f"{k}: nan"
            else:
                line = f"{k}: {v}"
            lines.append(line)

        metrics_text = "\n".join(lines)

        # 在坐标轴的右上角添加文本（使用 axes 坐标系）
        ax.text(
            1.02, 0.12,  # 相对轴坐标 (x>1 表示在图外侧一点)
            metrics_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_internal_metrics(
    X: np.ndarray,
    clusters: np.ndarray,
    metric: str = "euclidean",
) -> Dict[str, float]:
    """
    计算基于几何结构的内部聚类指标：
      - silhouette_score
      - calinski_harabasz_score
      - davies_bouldin_score

    对于 HDBSCAN 等有噪音簇（cluster = -1）的情况：
      - 先过滤掉 cluster = -1 的样本
    """
    # 过滤噪音：HDBSCAN 会用 -1 标记噪音；KMeans 不会出现 -1
    mask = clusters != -1
    X_valid = X[mask]
    c_valid = clusters[mask]

    # 至少需要两个簇 & 多于 1 个样本，否则这些指标无意义
    unique_clusters = np.unique(c_valid)
    if len(unique_clusters) < 2 or len(X_valid) < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    metrics = {}

    try:
        metrics["silhouette"] = silhouette_score(X_valid, c_valid, metric=metric)
    except Exception:
        metrics["silhouette"] = np.nan

    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(X_valid, c_valid)
    except Exception:
        metrics["calinski_harabasz"] = np.nan

    try:
        metrics["davies_bouldin"] = davies_bouldin_score(X_valid, c_valid)
    except Exception:
        metrics["davies_bouldin"] = np.nan

    return metrics


def compute_external_metrics(
    true_labels: np.ndarray,
    clusters: np.ndarray,
) -> Dict[str, float]:
    """
    计算利用已有标签的外部指标：
      - normalized_mutual_information (NMI)
      - adjusted_rand_index (ARI)

    同样会过滤 cluster = -1 的噪音样本。
    """
    mask = clusters != -1
    labels_valid = true_labels[mask]
    c_valid = clusters[mask]

    # 至少需要两个簇 & 两个不同标签
    if len(np.unique(c_valid)) < 2 or len(np.unique(labels_valid)) < 2:
        return {
            "nmi": np.nan,
            "ari": np.nan,
        }

    metrics = {}
    try:
        metrics["nmi"] = normalized_mutual_info_score(labels_valid, c_valid)
    except Exception:
        metrics["nmi"] = np.nan

    try:
        metrics["ari"] = adjusted_rand_score(labels_valid, c_valid)
    except Exception:
        metrics["ari"] = np.nan

    return metrics



# ---------------------------
# 切分视角构造
# ---------------------------

def build_views(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    构造不同的切分视角（布尔 mask）：
      - all: 全数据
      - hate_only: label != "non-hate"
      - label_{lab}: 每个标签单独一份

    返回：
      - dict: view_name -> boolean mask array (len = N)
    """
    N = len(df)
    masks = {}

    # 1. 全体样本
    masks["all"] = np.ones(N, dtype=bool)

    # 2. hate-only
    masks["hate_only"] = df["label"] != "non-hate"

    # 3. 每个标签单独一份
    unique_labels = sorted(df["label"].unique())
    for lab in unique_labels:
        name = f"label_{lab}"
        masks[name] = df["label"] == lab

    return masks


# ---------------------------
# 主流程
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sentence-level hate speech clustering with BGE embeddings."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to sentence-level dataset (.json or .jsonl).")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory to save outputs (CSVs and figures).")

    parser.add_argument("--model_path", default="models/base/bge-large-zh-v1.5",
                        help="Local path to BGE model, e.g., models/base/bge-large-zh-v1.5")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding.")
    parser.add_argument("--no_normalize", action="store_true", help="If set, do NOT L2-normalize embeddings.")

    parser.add_argument("--dr_method", choices=["none", "umap", "tsne"], default="umap",
                        help="Dimensionality reduction method.")
    parser.add_argument("--cluster_method", choices=["kmeans", "hdbscan"], default="kmeans",
                        help="Clustering method.")

    # KMeans 参数
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for KMeans.")

    # HDBSCAN 参数
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=15,
                        help="min_cluster_size for HDBSCAN.")
    parser.add_argument("--hdbscan_min_samples", type=int, default=5,
                        help="min_samples for HDBSCAN.")

    # UMAP 参数
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                        help="n_neighbors for UMAP.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="min_dist for UMAP.")

    # t-SNE 参数
    parser.add_argument("--tsne_perplexity", type=float, default=30.0,
                        help="perplexity for t-SNE.")
    parser.add_argument("--tsne_n_iter", type=int, default=1000,
                        help="n_iter for t-SNE.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 读取数据
    print(f"[INFO] Loading dataset from {args.input} ...")
    df = load_dataset(args.input)
    print(f"[INFO] Dataset size: {len(df)} samples.")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")

    # 2. 加载本地 BGE 模型
    print(f"[INFO] Loading embedding model from {args.model_path} ...")
    model = SentenceTransformer(args.model_path)

    # 3. 计算全文本 embedding
    print("[INFO] Computing embeddings ...")
    embeddings = compute_embeddings(
        model,
        df["content"].tolist(),
        batch_size=args.batch_size,
        normalize=not args.no_normalize
    )
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # 4. 构造不同视角（all, hate_only, label_xxx）
    views = build_views(df)

    # 5. 对每个视角执行 降维 + 聚类 + 可视化 + 保存结果
    for view_name, mask in views.items():
        idx = np.where(mask)[0]
        if len(idx) < 5:
            print(f"[WARN] View '{view_name}' has only {len(idx)} samples, skip.")
            continue

        print(f"\n[INFO] Processing view '{view_name}' with {len(idx)} samples ...")

        X = embeddings[idx]
        df_view = df.iloc[idx].copy().reset_index(drop=True)

        # 5.1 降维
        coords = None
        if args.dr_method != "none":
            print(f"[INFO]  - Reducing dimension using {args.dr_method} ...")
            coords = reduce_dimension(
                X,
                method=args.dr_method,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
                tsne_perplexity=args.tsne_perplexity,
                tsne_n_iter=args.tsne_n_iter,
            )
            df_view["x"] = coords[:, 0]
            df_view["y"] = coords[:, 1]
        else:
            print("[INFO]  - Skipping dimensionality reduction.")

        # 5.2 聚类
        print(f"[INFO]  - Clustering using {args.cluster_method} ...")
        clusters = run_clustering(
            X,
            method=args.cluster_method,
            n_clusters=args.n_clusters,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
        )
        df_view["cluster"] = clusters.astype(int)

        # 5.2.1 计算内部聚类指标（在当前 X 空间中）
        internal_metrics = compute_internal_metrics(
            X,
            clusters,
            metric="euclidean",  # 如果你想用 cosine，可以改为 "cosine"
        )
        print(f"[INFO]  - Internal metrics (view='{view_name}'):")
        print(f"           silhouette       = {internal_metrics['silhouette']:.4f}")
        print(f"           calinski_harabasz= {internal_metrics['calinski_harabasz']:.4f}")
        print(f"           davies_bouldin   = {internal_metrics['davies_bouldin']:.4f}")

        # 5.2.2 计算外部指标（与已有标签的一致性）
        label_array = df_view["label"].values
        external_metrics = compute_external_metrics(label_array, clusters)
        print(f"[INFO]  - External metrics (view='{view_name}'):")
        print(f"           NMI              = {external_metrics['nmi']:.4f}")
        print(f"           ARI              = {external_metrics['ari']:.4f}")

        metrics_for_plot = {}
        metrics_for_plot.update(internal_metrics)
        metrics_for_plot.update(external_metrics)

        metrics_path = os.path.join(
            args.output_dir,
            f"{view_name}_{args.cluster_method}_{args.dr_method}_metrics.json",
        )
        all_metrics = {
            "internal": internal_metrics,
            "external": external_metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"[INFO]  - Saved metrics to: {metrics_path}")

        # 5.3 打印简单统计（簇大小、簇内标签分布）
        print(f"[INFO]  - Cluster distribution in view '{view_name}':")
        print(df_view["cluster"].value_counts().sort_index())

        # 按簇与标签联合统计也存一份 CSV
        crosstab = pd.crosstab(df_view["cluster"], df_view["label"])
        crosstab_path = os.path.join(
            args.output_dir,
            f"{view_name}_{args.cluster_method}_{args.dr_method}_cluster_label_crosstab.csv",
        )
        crosstab.to_csv(crosstab_path, encoding="utf-8-sig")
        print(f"[INFO]  - Saved cluster-label crosstab to: {crosstab_path}")

        # 5.4 保存视角结果 CSV
        csv_path = os.path.join(
            args.output_dir,
            f"{view_name}_{args.cluster_method}_{args.dr_method}_results.csv",
        )
        df_view.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO]  - Saved view results to: {csv_path}")

        # 5.5 若有 2D 坐标，则画散点图（按 label / 按 cluster 着色）
        if coords is not None:
            # 按标签上色的分布图（不展示指标）
            fig_path_label = os.path.join(
                args.output_dir,
                f"{view_name}_{args.cluster_method}_{args.dr_method}_by_label.png",
            )
            plot_scatter(
                df_view,
                x_col="x",
                y_col="y",
                color_col="label",
                title=f"{view_name} - colored by label",
                save_path=fig_path_label,
                metrics=metrics_for_plot,  # 或者直接不写 metrics 参数
            )

            # 按簇上色的分布图（展示聚类指标）
            fig_path_cluster = os.path.join(
                args.output_dir,
                f"{view_name}_{args.cluster_method}_{args.dr_method}_by_cluster.png",
            )
            plot_scatter(
                df_view,
                x_col="x",
                y_col="y",
                color_col="cluster",
                title=f"{view_name} - colored by cluster ({args.cluster_method})",
                save_path=fig_path_cluster,
                metrics=metrics_for_plot,  # 把指标贴到图里
            )
            print(f"[INFO]  - Saved figure (colored by cluster) to: {fig_path_cluster}")
        else:
            print(f"[INFO]  - No 2D coordinates for view '{view_name}', skip plotting.")

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()
