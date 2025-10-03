# VocalTrax コード概要

## 全体像
VocalTrax は、人間の声帯・声道の物理モデルを JAX/Flax で記述し、観測した音声のスペクトログラムと一致するようにモデルパラメータを最適化するアーティキュラトリ合成システムです。`vocaltrax/synthesize.py` がエントリポイントで、Hydra による設定、音声読み込み、物理モデルの初期化、Optax による最適化ループをまとめて実行します。

## `synthesize.py` の処理フロー
- Hydra 設定 (`conf/config.yaml` とサブ設定) を読み込み、`utils.hydra.print_config` で内容を表示します。
- 乱数シードは `utils.random.PRNGKey` で管理し、JAX の PRNGKey を逐次分割します。
- 入力音声 (`cfg.general.target`) を読み込み、フレーム長に合わせてゼロパディングした後、`utils.misc.frameaudio` でフレーミングします。
- CREPE でフレームごとの基本周波数を推定し、信頼度 0.5 未満をゼロにマスクした上で `VocalTract` に渡します。
- `VocalTract` モデルを初期化し、Optax のオプティマイザ (既定は AdamW) を構築します。モデル適用は `jax.jit` で事前にコンパイルされます。
- 損失は `utils.audio` が生成する特徴量 (既定はメルスペクトログラム) に `utils.misc.preloss_log` を適用した後、`utils.misc.mse` でターゲットとの差を計算します。
- 反復ごとに `jax.value_and_grad` で勾配を取得し、Optax でパラメータ更新・`jnp.clip` による [0,1] への射影を行います。`cfg.general.smooth_every` ごとに Savitzky–Golay フィルタで時間方向の平滑化を挿入します。
- `cfg.general.log_every` ごとに合成音を `soundfile.write` で `logs/` 以下へ保存し、最終的に最適化後パラメータを `params.json` に書き出します。

## 声道モジュール
### `VocalTract`
- `PhysicalTract` が生成する直径系列と、フレームごとの張力 (`tenses`) を学習パラメータとして保持します。
- 基本周波数 `f0` と張力は `utils.misc.upsample_frames` でフレーム内補間し、`glottis_make_waveform` を通じて声門波形を生成します。
- 得られた励起信号と声道直径を `process_diams` に渡し、波の前進・後退成分をシミュレーションして最終的な音圧波形を返します。

### `PhysicalTract`
- 44 セグメントの基準直径に対して、以下のサブモジュールの適用結果を組み合わせて声道断面直径を決定します。
  - `Tongue`: 舌の位置 (`tongue_idx`) と舌量 (`tongue_diam`) を 0–1 正規化されたパラメータとして持ち、所定区間の直径を余弦カーブで再形成します。
  - `ThroatConstriction`: 喉部 (所定 index 以下) の直径を収縮させる単一パラメータのゲートです。
  - `LipConstriction`: 口唇側 (所定 index 以上) を同様に調整します。
- これらのモジュールはフレームごとに `jax.vmap` で並列適用され、すべて Flax `Module` として学習可能なパラメータを持ちます。

### `process_diams`
- 直径から面積を求め、隣接セグメントの反射係数を計算して時間方向に展開します。
- `jax.lax.scan` でステップを畳み込み、グロッタル端と口唇端でそれぞれ定められた反射係数 (`glottal_reflection`, `lip_reflection`) を適用します。
- 出力は 2 ステップ分のサンプルをまとめた後に間引かれ、最終的な音圧波形として返されます。

## 声門モデル (`glottis.py`)
- Liljencrants–Fant (LF) モデルに基づき、張力 (`tenseness`) から波形パラメータを `setup_lf` で計算します。
- 補間された `f0` に基づき位相を蓄積し、フレームごとに LF 波形と小さな白色雑音 (吸気音) を生成します。
- `jax.lax.scan` でフレームごとに処理し、連結した励起信号を返します。

## 特徴量と損失 (`utils/audio.py`, `utils/misc.py`)
- `utils/audio` は Audax のスペクトログラム実装をラップし、最大 100,000 サンプル単位に分割してメモリフットプリントを抑えています。
- メルスペクトログラムやマルチスケールメルなど複数の特徴関数に対応しており、Hydra の設定で切り替え可能です。
- `utils.misc` にはフレーミング (`frameaudio`)、フレーム補間 (`upsample_frames`)、正規化解除 (`unnormalize_params`)、損失関数などがまとまっています。

## 設定 (`conf/`)
- `conf/config.yaml` が基本設定で、`spectrogram/`, `preloss/`, `optimizer/` ディレクトリに個別のプリセットを持ちます。
- 実行時に `python synthesize.py general.iters=1000 ...` のように CLI から個別パラメータを上書き可能です。
- `general` セクションではログ保存先、入力ファイル、フレーム・サンプリング設定、最適化ハイパーパラメータ、スペクトログラム条件などを制御します。

## ログと成果物
- 実行時は `logs/<ターゲット名>/<optimizer>/<spectrogram>/<preloss>/<日時>/` 以下に WAV と `params.json` が保存されます。
- `params.json` は Flax/JAX のパラメータ PyTree を Python リストに変換したものです。再現には JAX 側で読み戻した後 `VocalTract.apply` に渡します。

## 参考事項
- コード全体が JAX/Flax を前提としており、GPU/TPU を利用する場合は Hydra 設定の `system.device` を変更する必要があります。
- CREPE による F0 推定は完全ではないため、信頼度マスクによって無声区間をゼロ周波数として扱います。必要に応じて別の F0 推定器に差し替えられる構造になっています。
- 自作のターゲット音声を使う際は、Hydra の `general.target` を自分のファイルに向け、サンプリングレートやフレーミング条件を適切に調整してください。
