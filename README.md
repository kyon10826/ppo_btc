# ppo_btc — PPOを用いたBTC取引戦略

深層強化学習（PPO）を用いて Bybit BTCUSDT 無期限先物（1時間足）の自動売買戦略を構築し、Walk-Forward Optimization（WFO）で検証するプロジェクトです。リスク調整型コンポジット報酬と Calmar 比基準のハイパーパラメータ選択により、ドローダウンに耐性のある戦略の獲得を目指しています。

> 出典: 今井智貴・今奏太・DAI HANG (2025-09-13) 「PPOを用いたBTC取引戦略」
> 初学者向けの解説は [`note_article.md`](./note_article.md) を参照してください。

---

## 目次

1. [概要](#概要)
2. [主な特徴](#主な特徴)
3. [ディレクトリ構成](#ディレクトリ構成)
4. [必要環境と依存ライブラリ](#必要環境と依存ライブラリ)
5. [クイックスタート](#クイックスタート)
6. [システムアーキテクチャ](#システムアーキテクチャ)
   - [データパイプライン](#データパイプライン)
   - [49特徴量の内訳](#49特徴量の内訳)
   - [取引環境 RiskAwareCryptoTradingEnv](#取引環境-riskawarecryptotradingenv)
   - [コンポジット報酬関数](#コンポジット報酬関数)
   - [PPOエージェント](#ppoエージェント)
   - [4つのハイパーパラメータセット](#4つのハイパーパラメータセット)
   - [Walk-Forward Optimization](#walk-forward-optimization)
7. [評価指標](#評価指標)
8. [結果](#結果)
9. [リーク防止の設計](#リーク防止の設計)
10. [既知の制限と今後の改善](#既知の制限と今後の改善)
11. [v9 改良点（旧版との差分）](#v9-改良点旧版との差分)
12. [ライセンスと免責](#ライセンスと免責)

---

## 概要

- **銘柄**: Bybit BTCUSDT Perpetual（USDT建て無期限先物）
- **足**: 1時間足
- **期間**: 2020-09-01 〜 2025-08-31（5年、43,824本）
- **手法**: PPO（Stable-Baselines3）+ 5分割 Walk-Forward Optimization
- **行動**: 連続値 -1（フルショート）〜 +1（フルロング）
- **選択基準**: Calmar 比（年率リターン / 最大ドローダウン）
- **比較対象**: Buy & Hold 戦略

## 主な特徴

- **49 個の特徴量**: リターン、ボラティリティ、トレンド、Order Flow Imbalance、市場レジーム検出など、すべて無次元化・定常性重視で設計
- **リスク調整コンポジット報酬**: 7 項目を組み合わせ、下落とドローダウンを重点的に罰する報酬関数
- **5分割 WFO × 4 ハイパラセット = 16 回の独立 PPO 学習**を回し、平均 Calmar 比で最良設定を自動選定
- **データリーク防止の三重設計**: 特徴量 `shift(1)` / Scaler は訓練 fit・検証 transform のみ / 厳格な時系列分割
- **動的 timesteps**: 学習データ行数に応じて学習回数を自動調整

## ディレクトリ構成

```
ppo_btc/
├── README.md                                          # 本ファイル（技術詳細）
├── note_article.md                                    # note掲載用・初学者向けチュートリアル
├── data/
│   ├── bybit_btcusdt_perp_1h_20200901_20250831.pkl   # 学習データ（pickle）
│   └── bybit_btcusdt_perp_1h_20200901_20250831.csv   # 同上（CSV）
├── finrl_btcusdt_drl_walkforward.ipynb               # 旧版（Sharpe比ベース）
└── finrl_btcusdt_drl_walkforward_v9_improved.ipynb   # v9 改良版（Calmar比ベース）★メイン
```

## 必要環境と依存ライブラリ

### 実行環境
- Python 3.10 以上
- GPU 推奨（CPU でも動作可、ただし数時間〜）
- メモリ 8GB 以上
- Google Colab（GPU ランタイム）対応

### 主要ライブラリ

| 用途 | パッケージ | バージョン |
|---|---|---|
| 強化学習アルゴリズム | `stable-baselines3` | 2.3.2 |
| 強化学習環境 API | `gymnasium` | 0.29.1 |
| データ処理 | `pandas` | >=2.0 |
| 数値計算 | `numpy` | >=1.23 |
| テクニカル指標 | `ta` | 0.11.0 |
| 前処理（スケーラ） | `scikit-learn` | 任意 |
| リスク指標 | `empyrical-reloaded` | 任意 |
| 可視化 | `matplotlib`, `plotly` | 任意 |
| 深層学習バックエンド | `torch` | SB3 依存 |

## クイックスタート

### 1. 依存関係のインストール

```bash
pip install -U pip setuptools wheel
pip install gymnasium==0.29.1 stable-baselines3==2.3.2
pip install "numpy>=1.23" "pandas>=2.0" matplotlib plotly scikit-learn
pip install ta==0.11.0 --use-pep517
pip install empyrical-reloaded
```

### 2. データパスの設定

`finrl_btcusdt_drl_walkforward_v9_improved.ipynb` のセル 5 で `DATA_PATH` をローカル環境のパスに調整してください（リポジトリ同梱の `data/bybit_btcusdt_perp_1h_20200901_20250831.pkl`）。

### 3. ノートブックを実行

```bash
jupyter notebook finrl_btcusdt_drl_walkforward_v9_improved.ipynb
```

全セルを順に実行すると、データ読込 → 特徴量生成 → WFO → 最終学習 → テスト期間バックテスト → 可視化までが自動で走ります。Colab 利用時はノートブック内の Drive マウント/Upload コードが分岐します。

### 4. 取引コストの切替

セル 5 の `TAKER_FEE` を変更してください（例: `0.0006` で 0.06% を試す）。

---

## システムアーキテクチャ

### データパイプライン

```
生 OHLCV (43,824 行)
    │  load_and_preprocess_data
    ▼
正規化・ソート済みデータ
    │  generate_enhanced_features_v9
    ▼
49 特徴量 + OHLCV (43,624 行、warmup 200 行を dropna)
    │  split_data(train_ratio=5/6)
    ▼
Train (36,353 行, 2020-09-09 〜 2024-11-02)
Test  ( 7,271 行, 2024-11-02 〜 2025-08-31)
    │  create_walk_forward_splits(n_splits=5)
    ▼
4 つの (train_fold, val_fold) ペア（ローリング拡張窓）
```

### 49 特徴量の内訳

すべての特徴量は学習時に `shift(1)` で 1 期ずらされ、未来情報のリークを防いでいます。

| カテゴリ | 数 | 主な特徴量 |
|---|---|---|
| リターン系 | 6 | `ret_1h`, `ret_4h`, `ret_12h`, `ret_24h`, `ret_48h`, `ret_168h` |
| 累積リターン系（v9 新規） | 2 | `cumret_24h`, `cumret_168h` |
| ボラティリティ系 | 7 | `vol_24h`, `vol_72h`, `vol_168h`, `vol_ratio_24_72`, `vol_ratio_24_168`, `realized_vol_24h`, `realized_vol_168h` |
| 市場レジーム系（v9 新規） | 3 | `trend_strength`, `adx`, `regime_trending` |
| Order Flow 系（v9 新規） | 4 | `buying_pressure`, `selling_pressure`, `order_flow_imbalance`, `volume_weighted_flow` |
| モメンタム/オシレータ系 | 5 | `rsi_7`, `rsi_14`, `rsi_30`, `stoch_k`, `stoch_d` |
| MACD 系 | 2 | `macd_diff_normalized`, `macd_signal_normalized` |
| バンド系 | 2 | `bb_position`, `bb_width` |
| ATR 系 | 1 | `atr_normalized` |
| 出来高系 | 4 | `volume_ratio_24h`, `volume_ratio_168h`, `volume_trend`, `obv_normalized` |
| 価格位置系 | 3 | `hl_ratio`, `hc_ratio`, `lc_ratio` |
| SMA 乖離系 | 3 | `deviation_sma20`, `deviation_sma50`, `deviation_sma200` |
| 価格形状系 | 2 | `price_skew`, `price_momentum` |
| 時間系 | 4 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| ボラレジーム系 | 1 | `vol_regime` |
| **合計** | **49** | |

#### 設計思想

- **無次元化**: 価格そのもの（1万〜12万ドルの範囲を取る）は使わず、比率・変化率・正規化値に統一
- **定常性**: RSI/Stoch を 0〜1 に、トレンド強度を 0〜1 のスコアに変換
- **リーク防止**: 関数末尾で全特徴量を `shift(1)`、その後 `dropna()` で warmup 期間（最長 sma200 の 200 本）を除去

### 取引環境 `RiskAwareCryptoTradingEnv`

Gymnasium 互換のカスタム環境です。

| 要素 | 仕様 |
|---|---|
| 観測空間 | `Box(shape=(53,), low=-inf, high=inf)`<br>＝ 49 市場特徴量 + [`position`, `pnl_clipped`, `drawdown`, `win_rate`] |
| 行動空間 | `Box(low=-1, high=+1, shape=(1,))` 連続値ターゲットポジション |
| 遷移 | 価格次足へ進む（`next_close` または `next_open` で約定） |
| 報酬 | 7 項目のコンポジット報酬（次節） |
| 終了条件 | (1) データ終端 / (2) NAV が初期資産の 10% 未満（破産判定で `reward=-1`） |
| スケーリング | `RobustScaler`、訓練 fold で `fit`、検証/テストは `transform` のみ |

#### `step()` の処理順序（要旨）

```python
1. target_position = clip(action, -1, +1)
2. price_return = (next_price - current_price) / current_price
3. portfolio_return = self.position × price_return × leverage - trade_cost
4. nav, peak_nav, drawdown を更新
5. recent_returns, wins, win_rate を更新
6. 7 項目のコンポジット報酬を計算
7. self.position = target_position  # ★ポジション更新は損益計算の後
8. current_step += 1
```

> 注: `portfolio_return` は更新前の `self.position`（前回決定したポジション）で計算されます。新しい行動は次のステップから損益に反映される、保守的な設計です。

### コンポジット報酬関数

```
reward = base_reward
       + downside_penalty
       + consistency_bonus
       + recovery_bonus
       + momentum_bonus
       + trade_penalty
       + dd_penalty
```

| 項 | 条件・式 | 役割 |
|---|---|---|
| `base_reward` | `portfolio_return × w_base` | 基本リターン |
| `downside_penalty` | `ret<0` のとき `-w_downside × ret² × 10` | 損失を 2 乗で重く罰する |
| `consistency_bonus` | 直近 12 本以上 かつ 勝率>0.5 で正の値 | 勝率の安定 |
| `recovery_bonus` | DD<-5% 中の `ret>0` で正の値 | ドローダウンからの回復 |
| `momentum_bonus` | 直近 3 本連続プラスで正の値 | 連続利益 |
| `trade_penalty` | `\|Δposition\|>0.5` で `-0.001 × Δposition` | 急激な反転を抑制 |
| `dd_penalty` | DD<-15% で `-0.02 × \|DD\|` | 大ドローダウンを忌避 |

デフォルト重み: `w_base=1.0, w_downside=0.3, w_consistency=0.2, w_recovery=0.1, w_momentum=0.1`

### PPOエージェント

| 要素 | 仕様 |
|---|---|
| アルゴリズム | PPO（On-policy、Actor-Critic、Clipped Surrogate Objective） |
| 実装 | `stable_baselines3.PPO('MlpPolicy', ...)` |
| ネットワーク | Actor / Critic 完全分離型 MLP（`net_arch={'pi': [...], 'vf': [...]}`） |
| 活性化 | Tanh（SB3 デフォルト） |
| 行動分布 | 対角共分散正規分布（連続行動） |
| 推論 | 学習中はサンプリング、評価時は `deterministic=True`（平均値） |

### 4 つのハイパーパラメータセット

WFO で同時並行で競わせる 4 つの哲学:

| 名称 | net_arch | learning_rate | gamma | clip_range | ent_coef | キャラクター |
|---|---|---|---|---|---|---|
| `DeepNet-v9-Large` | [1024, 1024] | 2e-5 | 0.995 | 0.15 | 0.008 | 大型脳・最長視点・最小探索 |
| `DeepNet-v9-ExtraLarge` | [1024, 512, 512] | 1e-5 | 0.99 | 0.10 | 0.010 | 超大型・超慎重 |
| `Adaptive-v9` | [512, 512, 256] | 5e-5 | 0.98 | 0.20 | 0.015 | 中型・積極学習（epochs=20） |
| `Conservative-v9` | [256, 256] | 1e-4 | 0.95 | 0.30 | 0.020 | 小型・敏速・短期視点 |

全セット共通: `n_steps / batch_size = 8`（minibatch 数を統一）、`max_grad_norm`, `gae_lambda` などは各設定で微調整。

### Walk-Forward Optimization

```
4 ハイパラセット × 4 fold = 16 回の独立 PPO 学習

           fold1       fold2       fold3       fold4    平均
HP1     ┌─Calmar─┬─Calmar─┬─Calmar─┬─Calmar─┐  avg ─┐
HP2     │  ...   │  ...   │  ...   │  ...   │  avg  │ ← 最大値が
HP3     │  ...   │  ...   │  ...   │  ...   │  avg  │   勝者
HP4     └─...─ ──┴───  ───┴─────  ─┴─────  ─┘  avg ─┘
```

各 fold で:

1. `train_fold` で `Scaler.fit` → `train_env` 構築
2. `train_env.get_scaler()` を `val_env` に渡す（fit せず transform のみ）
3. `train_ppo_model()` で PPO 学習（timesteps は `max(50000, len(train_fold) × 10)` で動的）
4. `evaluate_model_v9()` で検証 Calmar/Sortino/Sharpe/Return/MaxDD/Trades を計算
5. 結果を `WFOResult` に記録

最終的に各 HP の平均 Calmar 比を比較し、最大のものを `best_hyperparameters` として採用します。

その後、勝者 HP で `train_df` 全体を再学習（`max(200000, len(train_df) × 15) ≈ 545,000` timesteps）し、`test_df`（7,271 行）で最終バックテスト。

## 評価指標

`evaluate_model_v9()` および `backtest_strategy()` で計算:

| 指標 | 式 |
|---|---|
| Sharpe 比 | `√8760 × mean(returns) / std(returns)` |
| Sortino 比 | `√8760 × mean(returns) / std(負リターンのみ)` |
| Calmar 比 | `annual_return / |max_drawdown|` |
| Max Drawdown | `min((nav - cummax(nav)) / cummax(nav))` |
| Profit Factor | `sum(正リターン) / |sum(負リターン)|` |
| Win Rate | `count(returns > 0) / total` |

注: `ANNUALIZATION = 8760 = 24 × 365`（1 時間足の年率換算係数）

## 結果

テスト期間: 2024-11-02 〜 2025-08-31（7,271 時間 ≈ 302 日）

### 手数料なし

| Metric | DRL Strategy (v9) | Buy & Hold |
|---|---|---|
| Total Return (%) | **111.89** | 55.33 |
| Annual Return (%) | **147.17** | 69.98 |
| Sharpe Ratio | **2.34** | 1.35 |
| Sortino Ratio | **2.96** | 1.75 |
| **Calmar Ratio** | **9.03** | 2.26 |
| Max Drawdown (%) | **-16.29** | -30.98 |
| Win Rate (%) | 51.20 | — |
| Profit Factor | 1.09 | — |
| Number of Trades | 3,425 | 1 |
| Final NAV | 2.12 | 1.55 |

→ DRL 戦略が Buy & Hold を全指標で圧倒。選ばれた HP は `DeepNet-v9-Large`。

### 手数料 0.06%

| Metric | DRL Strategy (v9) | Buy & Hold |
|---|---|---|
| Total Return (%) | -9.86 | 55.23 |
| Calmar Ratio | -0.26 | 2.25 |
| Max Drawdown (%) | -45.08 | -30.98 |
| Number of Trades | 511 | 1 |
| Final NAV | 0.90 | 1.55 |

→ コスト込みでは DRL は B&H に敗北。3,425 → 511 と取引回数が大幅減るも、報酬関数がコストを考慮していないため戦略自体は最適化されておらず、累積コストに沈む。

## リーク防止の設計

このプロジェクトでは時系列リークを 3 段階で防いでいます。

1. **特徴量レベル**: `generate_enhanced_features_v9` の最後で全特徴量を `shift(1)` 化し、時刻 *t* の判断には *t-1* 以前の情報しか使えない構造を強制
2. **スケーリングレベル**: `RobustScaler` を訓練 fold だけで `fit` し、検証・テストには `transform` のみ適用。検証データの統計量は学習に一切混入しない
3. **データ分割レベル**: WFO は常に「過去 → 未来」のローリング拡張窓。テストデータ（直近 1/6）は全工程を通じて一度も学習・選択に使われない

## 既知の制限と今後の改善

- **取引コスト未考慮**: 報酬関数・HP 選択基準が手数料を反映していないため、コスト込みでは戦略が崩壊
- **取引回転の過多**: OFI などの短期特徴がポジション反転を頻発させる → コスト累積
- **Funding Rate 未モデル化**: 無期限先物特有の保有コストが反映されていない
- **レバレッジ・ロスカット未モデル化**: `LEVERAGE = 1.0` 固定、強制ロスカットや維持証拠金の制約なし
- **単一銘柄**: BTCUSDT のみ。複数銘柄ポートフォリオは未対応
- **適応学習率の名のみ**: スライドは「適応スケジュール」と記載するが、コード上は固定値（適応性は WFO で複数 LR を試す部分から来る）

### 改善方向の候補

- 報酬に `-α × |Δposition|` の取引コスト項を追加し、取引回転を直接ペナルティ化
- HP 選択基準を「コスト込み Calmar 比」に変更
- ポジション保持期間に下限を設ける（例: 4 時間以上）
- Funding Rate 履歴を取得し報酬・状態に組込み
- Recurrent Policy（LSTM）の導入

## v9 改良点（旧版との差分）

| 観点 | 旧版（`finrl_btcusdt_drl_walkforward.ipynb`） | v9 版 |
|---|---|---|
| 環境クラス | `ImprovedCryptoTradingEnv` | `RiskAwareCryptoTradingEnv` |
| 観測次元 | 特徴量 + 2 (position, pnl) | 特徴量 + 4 (+ drawdown, win_rate) |
| 報酬 | `portfolio_return + 0.01 × sharpe_24h` | 7 項目コンポジット |
| HP 選択基準 | 平均 Sharpe 最大 | 平均 Calmar 最大 |
| ネットワーク | [256/128/64] | [1024/512/256] |
| 特徴量 | 定常化重視（少なめ） | + レジーム検出 + Order Flow + 累積リターン |

## ライセンスと免責

本リポジトリのコード・解説は教育目的のみに提供されています。

- **投資助言ではありません**。本戦略を実取引に用いて生じた損失について、作者は一切の責任を負いません。
- 過去のバックテスト性能は将来の成果を保証しません。
- 暗号資産取引には元本を超える損失が生じる可能性があります。

---

**参考文献・出典**

- 元プロジェクト: 今井智貴・今奏太・DAI HANG (2025-09-13) 「PPOを用いたBTC取引戦略」
- アルゴリズム: J. Schulman et al. (2017) *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- 実装: A. Raffin et al. *Stable-Baselines3*
- データ提供: Bybit Public Data
