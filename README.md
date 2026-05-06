# Bandit learning in matching markets — reproduction starter

目标：复现 *Bandit Learning in Matching Markets: Utilitarian and Rawlsian Perspectives* 类工作（epoch Explore-Then-Commit、双侧效用、稳定匹配与福利）。

- 论文主页：<https://arxiv.org/abs/2412.00301>  
- 若你本地 PDF 文件名是 `2409_...`，请以 PDF **首页的 arXiv 编号**为准；若与上面不一致，改一下本 README 里的链接即可。

## 当前仓库里有什么

| 模块 | 作用 |
|------|------|
| `bandit_matching/deferred_acceptance.py` | 已知偏好时 Gale–Shapley（提出方一侧最优） |
| `bandit_matching/stable_bruteforce.py` | 小规模市场规模下枚举稳定匹配（用于 utilitarian / Rawlsian 对照） |
| `bandit_matching/toy_run.py` | 随机效用 + bandit 噪声的一次玩具实验（可跑通） |

后续可把论文中的 **Epoch ETC** 按小节/算法编号逐段接进 `bandit_matching/`。

## 环境

```bash
cd bandit_in_marketing
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m bandit_matching.toy_run
```

## 建议的复现顺序

1. 对照 PDF 记下：\(N, K\)、奖励噪声假设、冲突时收益（论文中为 0）、ETC 的 epoch 长度等。  
2. 在 `toy_run.py` 里把参数改成与实验节一致。  
3. 实现论文中的 ETC 变体，用 `stable_bruteforce` 或论文给出的“已知效用下的最优稳定匹配”子程序作 oracle 对照 regret。

若作者后续放出官方代码，可把本仓库当作实验脚手架，再逐步替换核心算法实现。
