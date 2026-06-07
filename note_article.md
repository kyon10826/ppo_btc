# 【完全ガイド】PPO（深層強化学習）でビットコイン自動売買AIをゼロから自作する

> 対象読者: 「機械学習で投資戦略を作ってみたい」と思っているけれど、金融もコーディングもまだ駆け出し、という方。専門用語はそのつど噛み砕いて説明します。
>
> 所要時間: 通読 60〜90 分、手を動かして実装するなら週末ふたつ程度。

---

## はじめに：このノートで手に入るもの

この記事は、5年分のビットコイン価格データに対して **深層強化学習（PPO）でトレード AI を学習させ、Buy & Hold（ただ買って持ち続ける戦略）と勝負させる** プロジェクトを、ゼロから自分で組み立てるためのチュートリアルです。

実際に手を動かして組み立てると、こんな図のような結果が出ます。

```
[テスト期間 約10ヶ月]
Buy & Hold:   +55%  /  最大下落 -31%
DRL 戦略 :  +112%  /  最大下落 -16%   ← 手数料0%なら圧勝！
```

…ただし、**手数料を 0.06% に変えるだけで結果はマイナスに転落**します。この「圧勝と惨敗」の両方を体験し、なぜそうなるのかを理解することが、この記事の本当のゴールです。**結果より、設計の良し悪しと落とし穴を学ぶための題材**として読んでください。

完成形のコードは公開リポジトリ ([`ppo_btc`](https://github.com/)) にあります。手元に置いて読み進めると理解が深まります。

---

## 第 1 部：背景知識（必要な人だけ）

### 1.1 強化学習を「ゲームをプレイするAI」として理解する

強化学習（Reinforcement Learning, RL）は、**試行錯誤しながら点数（報酬）を最大化する方法を学ぶ AI** の一族です。チェスや囲碁で人間を超えた AlphaGo、家庭用ゲームを攻略するDQNなどがこの仲間。

登場人物はたった 2 人:

- **エージェント**: プレイヤー。状況を見て、行動を選び、上手くなりたい AI
- **環境**: ゲーム盤＋審判。プレイヤーに状況を見せ、行動の結果を返し、点数を採点する

ループはシンプル:

```
   ┌──────────────────────┐
   │       環境          │
   └──┬─────────────▲────┘
  状況 │             │ 行動
      │     報酬     │
   ┌──▼─────────────┴───┐
   │     エージェント    │
   └──────────────────────┘
```

これを **トレードに当てはめる** と:

| ゲーム | トレード |
|---|---|
| 状況（盤面） | 価格・出来高・指標などの市場データ |
| 行動 | 買う/売る/様子見/ポジション量 |
| 報酬 | 儲け、または手数料を引いた損益 |
| プレイヤー | トレードAI（ニューラルネット） |

つまり、**価格データという"ゲーム"をプレイして報酬（儲け）を最大化するように学ぶ AI** を作ろう、というのが本プロジェクトです。

### 1.2 ビットコイン無期限先物とは（一般人向け）

扱うデータは **「Bybit BTCUSDT Perpetual」の 1 時間足**。長い名前ですが分解すると:

- **Bybit**: 仮想通貨の大手取引所
- **BTC**: ビットコイン（売買の対象）
- **USDT**: テザー。ほぼドルと同じステーブルコイン
- **BTCUSDT**: 「1 ビットコイン = 何 USDT か」の価格
- **Perpetual**: **無期限先物**

「無期限先物」は、**ビットコイン本体を持たずに値段の上下に賭けられる契約** です。普通の「買って持っておく」と違って:

1. **本体は買わない**: 財布にビットコインは入らない。値段の動きだけを追う
2. **上がる方にも下がる方にも賭けられる**: 「下がる」と思えばショート（空売り）で稼げる
3. **期限がない**: いつまでも続けられる

例えるなら、「ビットコインの値段が上がるか下がるかに賭けるゲームの、1 時間ごとの値段記録」を扱うイメージです。

### 1.3 なぜ仮想通貨 × 強化学習なのか

- 24 時間 365 日動くので、データに **休場や時間外がない** = きれいな連続データが取れる
- ボラティリティが大きく **学習の餌として情報量が多い**
- 1 銘柄で完結するので最初の題材として扱いやすい

---

## 第 2 部：プロジェクトの全体像

### 2.1 ゴールと評価方法

**ゴール**: PPO というアルゴリズムで「-1（フルショート）〜 +1（フルロング）」を 1 時間ごとに決める AI を作り、**Buy & Hold より良い戦略**を見つける。

「良い戦略」をどう測るか? このプロジェクトは **Calmar 比** を主役に据えています。

```
Calmar 比 = 年率リターン ÷ 最大ドローダウン（の絶対値）
```

「**痛みの最大値（最大の含み損）あたり、年で何%儲けたか**」。高いほどよい。

- 戦略 A: 年 20% / 最大下落 10% → Calmar 2.0
- 戦略 B: 年 30% / 最大下落 30% → Calmar 1.0
- → リターンが高くても、下落が深ければ Calmar は悪い

リターンだけでなく **耐えやすさ** を評価する指標です。

### 2.2 全体のフロー

```
[1] データを集める (Bybit 5年分)
        ↓
[2] 49個の特徴量を作る
        ↓
[3] 取引環境(ゲーム盤)を作る
        ↓
[4] PPOエージェント(プレイヤー)を用意
        ↓
[5] Walk-Forward Optimizationで4×4=16回学習
        ↓
[6] 一番強かったハイパーパラメータで本番学習
        ↓
[7] テスト期間(直近10ヶ月)でバックテスト
        ↓
[8] Buy & Holdと比較
```

これを順に作っていきます。

---

## 第 3 部：実装ステップ

### Step 1: 環境セットアップ

Python 3.10+ と pip があれば OK。Google Colab でも実行できます。

```bash
pip install -U pip setuptools wheel
pip install gymnasium==0.29.1 stable-baselines3==2.3.2
pip install "numpy>=1.23" "pandas>=2.0" matplotlib plotly scikit-learn
pip install ta==0.11.0 --use-pep517
pip install empyrical-reloaded
```

最初に乱数シードを固定しておくと結果の再現性が上がります。

```python
import random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
```

### Step 2: データを集める

このプロジェクトは Bybit BTCUSDT Perpetual の **1 時間足、2020-09-01 〜 2025-08-31**（5 年 = 43,824 本）を使います。

```python
import pandas as pd
df = pd.read_pickle('data/bybit_btcusdt_perp_1h_20200901_20250831.pkl')
df = df.set_index('datetime').sort_index()
df.columns = df.columns.str.lower()
```

データには以下の 5 列があれば十分:

| 列 | 意味 |
|---|---|
| open | その 1 時間の最初の取引価格 |
| high | その 1 時間の最高値 |
| low | その 1 時間の最安値 |
| close | その 1 時間の最後の取引価格 |
| volume | その 1 時間の出来高 |

これを **OHLCV データ** と呼びます。Bybit, Binance, ByBit など、ほとんどの取引所が同じ形式で配信しているので、別の銘柄でも置き換えられます。

**ポイント**: 必ず時刻順にソートし、欠損や時刻の飛びがないか確認しましょう。

```python
# 1時間刻みで連続しているか
diffs = df.index.to_series().diff().dropna()
print('1h以外の隙間:', (diffs != pd.Timedelta(hours=1)).sum())
```

### Step 3: 49 個の特徴量を作る

AI が「市場のいまの状態」を読み解くための入力です。**生の価格はそのまま使わず、必ず比率・変化率に変換** します。

なぜか? ビットコインは 5 年で 1.2 万ドル → 10.8 万ドルと約 12 倍動いており、生価格を入力にすると「価格が 1 万ドルの時の挙動」と「10 万ドルの時の挙動」が AI には別物に見えてしまうからです。

#### 例: リターン系特徴量

```python
for h in [1, 4, 12, 24, 48, 168]:
    df[f'ret_{h}h'] = df['close'].pct_change(h)
```

「1 時間前と比べて何%動いたか」「4 時間前と比べて何%か」「1 週間前 (168h) と比べて何%か」を入れます。これなら価格水準に関係なく、**いつでも同じ意味を持つ数字** になります。

#### 例: 市場レジーム検出

```python
import ta
sma_20  = ta.trend.sma_indicator(df['close'], 20)
sma_50  = ta.trend.sma_indicator(df['close'], 50)
sma_200 = ta.trend.sma_indicator(df['close'], 200)

df['trend_strength'] = (
    (df['close'] > sma_20).astype(float)
  + (df['close'] > sma_50).astype(float)
  + (df['close'] > sma_200).astype(float)
) / 3.0  # 0, 0.33, 0.66, 1.0 のいずれか
```

3 本の移動平均すべてより上なら 1.0、すべてより下なら 0.0。**今がトレンドの強い時期か弱い時期かを 0〜1 で表す** スコアです。

#### 例: Order Flow Imbalance（買い圧力 vs 売り圧力）

```python
df['buying_pressure']  = (df['close'] - df['low'])  / (df['high'] - df['low'])
df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
df['order_flow_imbalance'] = df['buying_pressure'] - df['selling_pressure']
```

終値が高値寄りなら買いが強かった証拠、安値寄りなら売りが強かった証拠。**ローソク足 1 本のミクロ構造** から相場の勢いを推定する指標です。

このような特徴量を全 14 カテゴリ・**合計 49 個** 作ります。詳細は README とリポジトリのノートブックを参照してください。

#### 最重要：「未来情報リーク」を防ぐ

特徴量計算が終わったら、**全特徴量を 1 期ずらす** ことを絶対に忘れないでください。

```python
feature_cols = [c for c in df.columns if c not in ['open','high','low','close','volume']]
for col in feature_cols:
    df[col] = df[col].shift(1)
df = df.dropna()
```

なぜか? `close` を使った特徴量は **その 1 時間が終わらないと確定しません**。確定値を「時刻 *t* の判断材料」に使ってしまうと、「未来を覗き見て取引する」ズルになり、バックテストはまったく当てになりません。

`shift(1)` をするだけで、「**時刻 *t* の AI は、*t-1* 以前で確定した情報しか見られない**」という制約が機械的に保証されます。

### Step 4: 取引環境（ゲーム盤）を作る

ここからが強化学習らしいパートです。OpenAI Gym（現 Gymnasium）の規約に従い、**自作の「環境クラス」** を用意します。最低限必要なメソッドは 2 つ:

- `reset()`: ゲームを初期状態に戻し、最初の観測を返す
- `step(action)`: 1 手進めて (新しい観測, 報酬, 終了か, 補足情報) を返す

```python
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from sklearn.preprocessing import RobustScaler

class RiskAwareCryptoTradingEnv(gym.Env):
    def __init__(self, df, feature_cols, taker_fee=0.0,
                 initial_capital=1.0, leverage=1.0,
                 scaler=None, fit_scaler=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.taker_fee = taker_fee
        self.initial_capital = initial_capital
        self.leverage = leverage

        # スケーリング（重要：訓練でfit、検証ではtransformのみ）
        if fit_scaler:
            self.scaler = RobustScaler()
            self.features_scaled = self.scaler.fit_transform(df[feature_cols].values)
        else:
            self.scaler = scaler
            self.features_scaled = scaler.transform(df[feature_cols].values)

        # 行動空間: -1〜+1 の連続値
        self.action_space = spaces.Box(low=-1, high=+1, shape=(1,), dtype=np.float32)
        # 観測空間: 特徴量49 + 口座状態4 = 53
        n = len(feature_cols) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.nav = self.initial_capital
        self.peak_nav = self.initial_capital
        self.drawdown = 0.0
        self.nav_history = [self.nav]
        self.position_history = [self.position]
        self.trades = []
        self.recent_returns = deque(maxlen=24)
        self.wins = deque(maxlen=24)
        self.win_rate = 0.5
        return self._get_observation(), {}

    def _get_observation(self):
        features = self.features_scaled[self.current_step]
        account_state = np.array([
            self.position,
            np.clip((self.nav - self.initial_capital) / self.initial_capital, -1, 1),
            self.drawdown,
            self.win_rate
        ])
        return np.concatenate([features, account_state]).astype(np.float32)

    def step(self, action):
        target_position = float(np.clip(action[0], -1, +1))

        # 価格変化率を計算
        current_price = self.df.iloc[self.current_step]['close']
        next_price    = self.df.iloc[self.current_step + 1]['close']
        price_return  = (next_price - current_price) / current_price

        # 取引コストと損益
        position_change = abs(target_position - self.position)
        trade_cost      = self.taker_fee * position_change
        portfolio_return = self.position * price_return * self.leverage - trade_cost

        # 資産・最高値・ドローダウンの更新
        self.nav *= (1 + portfolio_return)
        if self.nav > self.peak_nav:
            self.peak_nav = self.nav
        self.drawdown = (self.nav - self.peak_nav) / self.peak_nav

        # 勝敗の更新
        self.recent_returns.append(portfolio_return)
        self.wins.append(1 if portfolio_return > 0 else 0)
        self.win_rate = sum(self.wins) / max(1, len(self.wins))

        # ★ コンポジット報酬（後述）
        reward = self._compute_reward(portfolio_return, position_change)

        # ポジション更新（損益計算後）
        self.position = target_position

        # 履歴記録
        self.nav_history.append(self.nav)
        self.position_history.append(self.position)
        self.current_step += 1

        done = self.current_step >= len(self.df) - 2
        if self.nav < self.initial_capital * 0.1:   # 90%損失で破産
            done = True
            reward = -1.0
        return self._get_observation(), reward, done, False, {}
```

#### コードの肝: 損益の計算

```python
portfolio_return = self.position × price_return × leverage − trade_cost
```

**ポジション × 価格変化率**。これが先物取引の本質です。

- ロング（position > 0）で価格が上がれば → プラス × プラス = プラス（儲け）
- ショート（position < 0）で価格が下がれば → マイナス × マイナス = プラス（儲け）

下がるときも稼げるのは、ポジションの符号がマイナスだから、というそれだけの話です。

### Step 5: 報酬関数を設計する

ここが本プロジェクト最大の工夫です。**儲け（損益）をそのまま報酬にすると、AI は一発逆転狙いの大博打を学んでしまいます**。代わりに、リスクを意識した **コンポジット報酬** を作ります。

```python
def _compute_reward(self, portfolio_return, position_change):
    w_base, w_downside, w_consistency, w_recovery, w_momentum = 1.0, 0.3, 0.2, 0.1, 0.1

    # ① 基本報酬
    base_reward = portfolio_return * w_base

    # ② 下落ペナルティ（負リターンを2乗で重く罰する）
    downside_penalty = 0
    if portfolio_return < 0:
        downside_penalty = -w_downside * (portfolio_return ** 2) * 10

    # ③ 一貫性ボーナス（勝率>0.5でプラス）
    consistency_bonus = 0
    if len(self.wins) >= 12:
        consistency_bonus = w_consistency * max(0, self.win_rate - 0.5) * 0.1

    # ④ 回復力ボーナス（DD<-5%中のプラスを褒める）
    recovery_bonus = 0
    if self.drawdown < -0.05 and portfolio_return > 0:
        recovery_bonus = w_recovery * portfolio_return * abs(self.drawdown) * 2

    # ⑤ モメンタムボーナス（3連勝で褒める）
    momentum_bonus = 0
    if len(self.recent_returns) >= 3 and all(r > 0 for r in list(self.recent_returns)[-3:]):
        momentum_bonus = w_momentum * np.mean(list(self.recent_returns)[-3:]) * 2

    # ⑥ 過度な取引ペナルティ
    trade_penalty = -0.001 * position_change if position_change > 0.5 else 0

    # ⑦ 大ドローダウンへの追加ペナルティ
    dd_penalty = -0.02 * abs(self.drawdown) if self.drawdown < -0.15 else 0

    return (base_reward + downside_penalty + consistency_bonus
            + recovery_bonus + momentum_bonus + trade_penalty + dd_penalty)
```

#### 設計の意図

| 項 | 何を促すか |
|---|---|
| ① base | 単純に儲かったら正の報酬 |
| ② downside | **負を 2 乗で罰する**ことで「大損は絶対避けろ」を強く伝える |
| ③ consistency | チマチマ勝ち続けることを評価 |
| ④ recovery | 含み損中の挽回を強く評価 |
| ⑤ momentum | 連勝を褒める（モメンタムに乗る） |
| ⑥ trade | 派手なポジション変更を抑制 |
| ⑦ dd | 大ドローダウンをさらに罰する |

これを最大化するように学習することで、AI は **「リターンを稼ぐ」より「深い谷を作らない」を優先**するように育ちます。これが Calmar 比志向の戦略を作る肝です。

### Step 6: PPO エージェントを学習させる

エージェント本体は **Stable-Baselines3 ライブラリ** が提供してくれるので、自分で書くのは環境と報酬だけです。

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 訓練データで環境を作る
train_env = RiskAwareCryptoTradingEnv(
    df=train_df, feature_cols=feature_cols,
    taker_fee=0.0, fit_scaler=True       # 訓練でScaler を fit
)
vec_env = DummyVecEnv([lambda: train_env])

# PPOモデルを作って学習
model = PPO(
    'MlpPolicy', vec_env,
    learning_rate=2e-5,
    n_steps=4096,
    batch_size=512,
    n_epochs=15,
    gamma=0.995,             # 約200時間先まで見据える
    gae_lambda=0.98,
    clip_range=0.15,         # 一度に方針を15%以上変えない
    ent_coef=0.008,          # 探索ボーナス
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={'net_arch': [dict(pi=[1024,1024], vf=[1024,1024])]},
    seed=SEED, device='auto'
)
model.learn(total_timesteps=200_000)
```

#### PPO の頭の中（イメージ）

PPO は **2 つのニューラルネット** を持っています。

- **Actor**: 「今どう動くべきか」を出す本番の意思決定者（ポジションを出力）
- **Critic**: 「今の状況はどれくらい有望か」を見積もるコーチ役

学習は、

1. **経験を集める**: 今の方針で環境を 4,096 ステップ動かし、(状態, 行動, 報酬) を記録
2. **アドバンテージ計算**: 各行動が「予想よりどれだけ良かったか」を計算
3. **脳を更新**: 良かった行動の確率を上げる（ただし `clip_range` で変化を制限）
4. ループ

を繰り返します。`clip_range` の存在が PPO の名前の由来で、**一度の更新で方針が壊れないように調整**する仕組みです。

### Step 7: Walk-Forward Optimization で検証する

「過去で学んで未来でテスト」を時間方向にずらしながら繰り返すのが **Walk-Forward Optimization（WFO）** です。

#### データ分割

```python
n = len(df_features)
train_size = int(n * 5/6)
train_df = df_features.iloc[:train_size]
test_df  = df_features.iloc[train_size:]    # ← 最後まで一度も触らない
```

train を 5 分割し、ローリング拡張窓で 4 つの (train_fold, val_fold) を作ります:

```
Split 1: 学習 [...10ヶ月]                / 検証 [次の10ヶ月]
Split 2: 学習 [.........20ヶ月]          / 検証 [次の10ヶ月]
Split 3: 学習 [..............30ヶ月]     / 検証 [次の10ヶ月]
Split 4: 学習 [.................40ヶ月]  / 検証 [次の10ヶ月]
```

各 fold で **訓練 fold だけで Scaler を fit** し、検証 fold には `transform` のみ適用します（リーク防止の鍵）。

#### 4 つのハイパーパラメータを競わせる

性格の違う 4 設定（大型・超大型・中型アクティブ・小型敏速）を全 fold で学習。合計 **4 × 4 = 16 回の独立 PPO 学習** を回します。

各 fold の検証 Calmar 比を記録し、最後に **平均 Calmar 比が最大の設定** を採用:

```python
avg_calmars = {hp: np.mean([r.val_calmar for r in results]) for hp, results in all_results.items()}
best_hp_name = max(avg_calmars, key=avg_calmars.get)
```

### Step 8: 本番学習＆テスト期間でバックテスト

勝った設定で、訓練データ全体 (`train_df`) を再学習。学習時間（timesteps）を増やして、本気で学ばせます。

```python
final_model = PPO('MlpPolicy', vec_env, **best_hyperparameters, ...)
final_model.learn(total_timesteps=545_000)  # 動的に計算（行数×15）
```

そして、**学習に一度も使っていないテスト期間** (直近 10 ヶ月) で動作確認:

```python
test_env = RiskAwareCryptoTradingEnv(
    df=test_df, feature_cols=feature_cols,
    taker_fee=0.0,
    scaler=train_env.get_scaler(),  # 訓練の Scaler を流用
    fit_scaler=False                 # ← 絶対 fit しない
)
obs, _ = test_env.reset()
done = False
while not done:
    action, _ = final_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
```

`deterministic=True` は「乱数を使わず、Actor の平均出力をそのまま使う」モードです。本番は確定的に動かします。

### Step 9: Buy & Hold と比較する

```python
def buy_and_hold(test_df, initial=1.0, fee=0.0):
    prices = test_df['close'].values
    btc = initial * (1 - fee) / prices[0]
    nav_history = btc * prices
    return nav_history

bh_nav = buy_and_hold(test_df)
```

「最初の足で全力で買って、最後の足まで持ち続けた場合の資産推移」を計算し、DRL 戦略の `nav_history` と並べてグラフ化します。

```python
import matplotlib.pyplot as plt
plt.plot((test_env.nav_history - 1) * 100, label='DRL Strategy')
plt.plot((bh_nav - 1) * 100, label='Buy & Hold', linestyle='--')
plt.legend(); plt.ylabel('Cumulative Return (%)'); plt.show()
```

---

## 第 4 部：結果と教訓

### 4.1 数字で見る結果

テスト期間（2024-11-02 〜 2025-08-31、約 10 ヶ月）の結果:

#### 手数料 0%

| 指標 | DRL 戦略 | Buy & Hold |
|---|---|---|
| Total Return | **+111.9%** | +55.3% |
| Calmar 比 | **9.03** | 2.26 |
| 最大 DD | **-16.3%** | -30.9% |
| Sharpe 比 | 2.34 | 1.35 |
| Final NAV | 2.12 | 1.55 |

→ **すべての指標で DRL が圧勝**。特にドローダウンを半減させた点が際立ちます。

#### 手数料 0.06%（実際の Bybit 取引手数料に近い水準）

| 指標 | DRL 戦略 | Buy & Hold |
|---|---|---|
| Total Return | **-9.9%** | +55.2% |
| Calmar 比 | **-0.26** | 2.25 |
| 最大 DD | **-45.1%** | -30.9% |
| Final NAV | 0.90 | 1.55 |

→ **同じ AI が、手数料の数字を変えただけで惨敗**。

### 4.2 教訓 1: 取引コストを侮るな

手数料なし版では **3,425 回** の取引、手数料あり版でも **511 回**。10 ヶ月で 511 回でも、1 回 0.06% × 511 = 30% 相当のコストになります。

問題の本質は、**報酬関数も HP 選択も「コストゼロを前提に設計されている」** こと。AI は儲かるパターンを見つけているのに、コストを差し引くとマイナスになる。これは現実の運用で最もハマる落とし穴です。

**改善策**: 報酬関数の `trade_penalty` を強くする、もしくは HP 選択を「コスト込み Calmar」に変更する。

### 4.3 教訓 2: リークを徹底的に避けろ

「過去のバックテストでは儲かったのに、本番に出したら全然ダメ」という話の 9 割は **データリーク** が原因です。本プロジェクトは 3 段階で防いでいます:

1. **特徴量**: `shift(1)` で未来を覗かない
2. **Scaler**: 訓練でだけ fit、検証は transform のみ
3. **データ分割**: 検証は常に訓練の "未来"、テストは全工程で一度も触らない

「**評価が良すぎたら、まずリークを疑え**」と覚えておいてください。

### 4.4 教訓 3: 報酬設計が 9 割

PPO 本体は SB3 が提供してくれるので、自分が触るのは **環境（特に報酬）と特徴量** だけです。
- 損益をそのまま報酬にすると博打を学ぶ
- ドローダウンを罰すると慎重になる
- 連勝を褒めるとモメンタムを掴む
- これらの **重み（`w_base`, `w_downside`...）が戦略の性格を決める**

報酬関数は、AI への「お手本となる価値観」の指定書。ここに自分の投資哲学を込めるイメージで設計しましょう。

---

## 第 5 部：自分でアレンジするヒント

### 5.1 拡張アイデア

| やってみたいこと | 改造ポイント |
|---|---|
| 別の銘柄 | データを差し替えるだけ（OHLCV さえあればOK） |
| 4 時間足や日足 | データの取得粒度を変える。`ANNUALIZATION` 係数も合わせて変更 |
| 複数銘柄ポートフォリオ | 行動空間を `Box(-1,+1, shape=(n_assets,))` に拡張 |
| Funding Rate 込み | データに funding_rate 列を追加し、損益計算と特徴量に反映 |
| LSTM ポリシー | SB3 の `RecurrentPPO`（sb3-contrib）に切り替え |
| 取引コスト織込み | 報酬関数の `trade_penalty` 強化 ＋ HP 選択を「コスト込み Calmar」に |

### 5.2 ハマりやすい落とし穴

1. **`shift(1)` を忘れる**: バックテストが嘘の利益を出す
2. **Scaler を検証データで fit してしまう**: リークで実力以上の数字が出る
3. **`deterministic=False` のままバックテスト**: 結果が再現せず、本番でも不安定
4. **報酬 ≠ 損益を混同**: AI は報酬を最大化するのであって損益ではない
5. **手数料を後から足す**: 戦略全体が手数料前提で歪んでいるので、後付けではダメ
6. **WFO の検証 fold で再学習**: 検証データの statistic を学習に使ったら検証の意味がない
7. **ライブラリのバージョン不一致**: SB3 と Gymnasium の組合せは README の通りに固定するのが無難

### 5.3 さらに学ぶための参考リンク

- **PPO 原著**: Schulman et al. (2017) "Proximal Policy Optimization Algorithms" arXiv:1707.06347
- **Stable-Baselines3 公式**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **書籍**: 『深層強化学習による金融取引』『強化学習』(Sutton & Barto 邦訳)

---

## おわりに

ここまで読んでいただきありがとうございました。

このプロジェクトの最大の価値は、**「うまくいく方法」を学ぶことではなく、「うまくいきそうに見えて本当はそうではない」設計の罠を、自分の手で再現できる** ことだと思います。

特に **手数料を 0% → 0.06% に変えると圧勝が惨敗に変わる** という体験は、机上の論文を 10 本読むより腹落ちします。「実運用に出す前に、何を考慮しなければいけないか」を骨身で理解できるはずです。

ぜひ手を動かしてみてください。改造の方向性は無数にあります。コードは [GitHub リポジトリ](https://github.com/) に公開されているので、フォークして自分の戦略を育ててみてください。

---

**免責事項**

本記事およびプロジェクトのコードは教育目的のみに提供されており、投資助言ではありません。本戦略を実取引に使用して生じた損失について筆者は一切の責任を負いません。暗号資産取引には元本を超える損失が生じる可能性があります。
