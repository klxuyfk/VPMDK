# VPMDK サーバーモード実装指示書 (v1)

作成日: 2026-07-16 / 対象リポジトリ: vpmdk (klxuyfk/vpmdk)

## 0. 背景と目的

現在の `vpmdk` は 1 回の起動につき 1 計算を実行する。起動のたびに
`_build_calculator_from_tags()` が MLP モデル(CHGNet / MACE / MatterSim /
MatGL / fairchem / ORB ...)を VRAM にロードし直すため、小さな構造を大量に
処理するワークフロー(結晶構造探索、スクリーニング、データセット生成など)
ではモデルロードが実行時間の 2〜4 倍を占める。

本機能は **モデルを VRAM に常駐させたサーバーを CLI から起動し、別コマンド
で計算を何度でも投げ、最後に CLI から停止する** 汎用機能である。

**重要な設計方針: これは VPMDK として完結した一般機能である。**
特定の下流ソフトウェア(LSPEX 等)のための専用機能ではない。仕様・命名・
ドキュメントはすべて「たくさんの計算を捌きたい一般ユーザー」を主語として
書くこと。下流からの利用像は付録 A に参考情報として載せるが、vpmdk 本体に
下流固有の概念を持ち込んではならない。

## 1. 絶対条件(互換性契約)

以下は交渉不可の要件。実装がこれを満たさない場合は設計から見直すこと。

1. **既存 CLI の完全後方互換。** `vpmdk` および `vpmdk --dir DIR` の挙動は
   1 バイトも変えない。サブコマンドを追加しても、第一引数がサブコマンド名
   でない従来呼び出しはそのまま従来経路に入る。これを固定する回帰テストを
   追加すること。
2. **出力等価性。** 同一の入力ディレクトリ・同一のモデル/デバイスに対して、
   `vpmdk run --dir D`(サーバー経由)は `vpmdk --dir D`(ワンショット)と
   **同一の出力ファイル群を同一の内容で** 生成する(OUTCAR / OSZICAR /
   CONTCAR / vasprun.xml、および BCAR オプションで有効化される energy CSV /
   LAMMPS trajectory / CHGCAR / XDATCAR 等)。浮動小数点の表記も含めて
   一致させるため、実装は「同じコードパスを通す」ことで達成する(§4 参照)。
3. **成功マーカー。** ワンショット成功時に stdout へ出る
   `Calculation completed.` は、`vpmdk run` 成功時にもクライアントの stdout
   に必ず出す(既存のスクリプト・下流ツールがこの行を成功判定に使える)。
4. **失敗の隔離。** 1 リクエストの失敗(例外・OOM・不正入力)はサーバーを
   殺さない。失敗はそのリクエストのクライアントにのみ返り、サーバーは次の
   リクエストを受け付け続ける。
5. **モデル常駐の単位は「1 サーバー = 1 バックエンド構成」。**
   1 つのサーバーは 1 つの (MLP, MODEL, DEVICE) 構成だけを保持する。
   複数モデル・複数 GPU を扱いたいユーザーは複数サーバーを別ソケットで
   立てる。サーバー内でのモデル切り替え・マルチテナントは v1 の非目標。

## 2. CLI 仕様

argparse にサブコマンドを追加する。サブコマンド名は `serve` / `run` /
`status` / `stop` の 4 つ。

### 2.1 `vpmdk serve` — サーバー起動(モデル常駐)

```
vpmdk serve [--dir DIR] [--bcar PATH] [--socket PATH]
            [--idle-timeout SEC] [--daemon] [--log-file PATH]
```

- バックエンド構成の解決: `--bcar PATH` があればそれを、なければ
  `--dir`(既定 `.`)の `BCAR` を読む。BCAR が見つからなければ既定構成
  (MLP=CHGNET)で起動してよいが、その旨を明示的にログする。
- 起動時に **一度だけ** calculator を構築し(既存の
  `_build_calculator_from_tags`)、以後プロセス生存中は保持する。
  DEVICE タグ(未指定なら既存の `_resolve_device` の自動検出)と、起動時の
  `CUDA_VISIBLE_DEVICES` がそのままデバイス配置を決める。サーバー自身は
  GPU 管理をしない(複数 GPU はユーザーが `CUDA_VISIBLE_DEVICES=n vpmdk
  serve --socket ...` を GPU ごとに起動して使い分ける)。
- **モデルロードが完了してからソケットを bind する。** ソケットファイルの
  出現がそのまま readiness シグナルになる(クライアントは `vpmdk status`
  が応答するまでポーリングすれば良い)。
- 既定はフォアグラウンド実行(シェルの `&`、tmux、systemd、ジョブスクリプト
  に馴染む)。`--daemon` は double-fork + pidfile(ソケットと同じディレクトリ
  に `<name>.pid`)+ `--log-file` 既定 `<socket>.log` のデーモン化。v1 では
  POSIX のみサポートで良い。
- `--idle-timeout SEC`: 最後のリクエスト完了から SEC 秒間新規リクエストが
  無ければ自動終了(VRAM を握った孤児サーバー対策)。既定 0 = 無効。
- 起動時にソケットパスが既に存在する場合: 接続を試み、生きたサーバーが
  応答したら「already running」でエラー終了(exit 1)。応答が無ければ
  stale socket とみなして unlink して起動を続ける。

### 2.2 `vpmdk run` — 常駐サーバーに 1 計算を投げる

```
vpmdk run [--dir DIR] [--socket PATH] [--timeout SEC]
```

- `--dir`(既定 `.`)をワンショットの `vpmdk --dir DIR` と同じ意味で解釈
  する。クライアントはディレクトリの **絶対パス** をサーバーに送るだけで、
  入力の解釈・実行・出力書き込みはすべてサーバー側プロセスが行う。
- ブロッキング実行: 計算が終わる(または失敗する)までクライアントは
  待つ。サーバーがビジーなら FIFO で並ぶ(§3.2)。
- `--timeout SEC`(既定 0 = 無制限): 超過したらクライアントは exit 4 で
  抜ける。v1 ではサーバー側のジョブは中断しなくて良い(中断はベストエフォート
  で、実装が重ければ「クライアントが去ってもジョブは完走する」と文書化)。
- サーバー無応答・接続不能は **即エラー(exit 3)**。ワンショット実行への
  暗黙フォールバックはしない(設定ミスや死んだサーバーを隠蔽し、silent に
  モデルロード代を払い続けることになるため)。
- ソケットパスの解決順序(`run` / `status` / `stop` 共通):
  `--socket` フラグ > 環境変数 `VPMDK_SOCKET` > 既定パス。
  既定パスは `${XDG_RUNTIME_DIR:-/tmp}/vpmdk-<uid>/default.sock`。
  既定ディレクトリはモード 0700 で作成する。

### 2.3 `vpmdk status` — 生存確認と状態表示

```
vpmdk status [--socket PATH] [--json]
```

- 応答(§3.3 の status レスポンス)を人間可読で表示。`--json` で生 JSON。
- サーバー生存なら exit 0、接続不能なら exit 3。スクリプトの readiness
  ポーリングはこれを使う。

### 2.4 `vpmdk stop` — サーバー停止

```
vpmdk stop [--socket PATH] [--force] [--timeout SEC]
```

- 既定は graceful: 実行中のジョブがあれば完走を待ってから終了。
  `--force` は即時終了(実行中ジョブのクライアントにはエラーが返る)。
- 停止完了(ソケット消滅)まで待って exit 0。`--timeout`(既定 60s)超過は
  exit 4。
- サーバーは終了時に必ずソケットファイルと pidfile を削除する(シグナル
  ハンドラ: SIGTERM / SIGINT でも graceful shutdown + クリーンアップ)。

### 2.5 終了コード(`run` / `status` / `stop` 共通)

| code | 意味 |
|---|---|
| 0 | 成功 |
| 1 | 入力が不正(POSCAR/INCAR/BCAR の解析失敗、NEB レイアウト/幾何の不整合、未対応 INCAR 設定 等)。ワンショット `vpmdk --dir` と同じく非リトライ対象 |
| 2 | 計算が**実行中に**サーバー側で失敗した(計算中の例外・OOM・非収束エラー等)。リトライで解消しうる |
| 3 | サーバーに接続できない / 応答しない / 実行中に接続が切れた |
| 4 | クライアント側タイムアウト |
| 5 | バックエンド構成の不一致(§3.4) |

exit 1(入力不正)と exit 2(計算失敗)は明確に分ける: 入力不正はユーザーが入力を直すまで恒久的に失敗する非リトライ対象であり、
サーバーは `code="input_error"` の `done` イベントで返す(ワンショットは exit 1 で終了)。計算中の例外は `code="calculation_error"`
(exit 2)で、リトライ機構(付録 A)の再試行対象となりうる。

## 3. プロトコル仕様

### 3.1 トランスポート

- **Unix ドメインソケット + 改行区切り JSON(NDJSON)。1 接続 = 1 リクエスト。**
  クライアントが接続 → リクエスト 1 行送信 → サーバーがイベント行を
  ストリーム → `done` イベントで双方クローズ。
- POSIX 専用で良い(GPU 計算の実態は Linux)。Windows ではサーバー系
  サブコマンドは明確なエラーメッセージで拒否する。
- 認証はファイルシステムパーミッションに委ねる(同一ユーザー前提)。
  ソケットの親ディレクトリは 0700。この前提(ソケットにアクセスできる者は
  サーバーのユーザー権限で任意ディレクトリに計算出力を書かせられる)を
  ドキュメントに明記する。

### 3.2 実行モデル

- サーバーは **直列実行**(ワーカー 1 本、FIFO キュー)。GPU 上の MLP 推論は
  実質デバイス直列であり、並列受付はメモリと複雑さを増やすだけ。複数
  クライアントが同時に `run` してきた場合は受付順に処理する。
- accept ループと実行ワーカーは分離し、実行中でも `status` / `stop` には
  即応答できること。

### 3.3 メッセージ

リクエスト(クライアント → サーバー、1 行):

```json
{"op": "run", "version": 1, "workdir": "/abs/path/to/calc_dir"}
{"op": "status", "version": 1}
{"op": "stop", "version": 1, "force": false}
```

`run` はオプションで `"umask": <int 0..0o777>` を送ってよい。サーバーは
その値を当該ジョブの実行中だけプロセス umask に適用し、出力アーティファクトの
モードをワンショット実行と一致させる(送られなければ従来通りサーバー側の
umask)。型・範囲が不正な `umask` はプロトコルエラー。

レスポンス(サーバー → クライアント、イベント行のストリーム):

```json
{"event": "accepted", "queue_position": 0}
{"event": "log", "line": "Note: KPOINTS detected but not used in MLP calculations."}
{"event": "heartbeat", "elapsed_s": 30.0}
{"event": "done", "ok": true, "elapsed_s": 12.3}
{"event": "done", "ok": false, "error": "OOM ...", "traceback": "..."}
```

- `log` イベント: サーバー側で当該リクエスト実行中に発生する既存の
  print/警告類をクライアントへ中継し、クライアントはそのまま stdout に
  出す(§1-3 の成功マーカーもこの経路で流れる)。実装はワンショット経路の
  stdout をリクエストスコープでキャプチャする形でよい。
  計算中の stderr(サードパーティの警告類)は `"stream": "stderr"` を付けた
  `log` イベントとして同様に中継し、クライアントは自分の stderr に書き戻す
  (§1.2 の両ストリーム一致のため)。`stream` なしの `log` は従来通り stdout。
  古いクライアントは `stream` キーを無視して stdout に出すだけなので、
  行が失われることはない。
- `heartbeat`: 実行中 30 秒ごとに送る。クライアントはこれで「長い計算」と
  「ハング/死亡」を区別できる(ソケット無音が heartbeat 間隔を大きく
  超えたら異常と判断できる)。
- `status` への応答:

```json
{"event": "status", "state": "idle", "backend": {"mlp": "MACE", "model": "/path/model", "device": "cuda"},
 "jobs_completed": 42, "jobs_failed": 1, "queue_length": 0,
 "uptime_s": 3600.5, "pid": 12345, "vpmdk_version": "0.1.0", "protocol": 1}
```

- 未知の `op` / `version` 不一致にはエラーイベントを返して接続を閉じる
  (サーバーは落ちない)。

### 3.4 リクエスト側 BCAR の扱い

ワークディレクトリに BCAR が置かれている場合:

- **バックエンド系タグ(MLP / NNP / MODEL / DEVICE)が常駐構成と食い違って
  いたら、そのリクエストをエラーで拒否する**(クライアント exit 5、差分を
  メッセージに列挙)。黙って常駐モデルで計算すると「意図と違うモデルの
  結果」が静かに混入するため、明示エラーが正しい。
- バックエンド系以外のタグ(energy CSV / LAMMPS trajectory / CHGCAR 出力
  等の実行オプション)はリクエストごとにそのまま尊重する(ワンショットと
  同じ挙動)。

## 4. 実装指針(リファクタリング)

### 4.1 ワンショット経路の関数抽出(最重要)

現在 `cli.main()` に直書きされている「workdir を受けて入力を解釈し、
single-point / relaxation / MD / NEB / force-constants にディスパッチして
出力を書く」一連の処理を、**再利用可能な関数に抽出する**:

```python
def run_workdir(workdir: str, *, calculator=None) -> None:
    """ワンショット CLI とサーバーの共通実行経路。
    calculator=None ならこの場で構築(従来挙動)、渡されたら再利用。"""
```

- ワンショット CLI は `run_workdir(dir)`、サーバーは
  `run_workdir(dir, calculator=resident_calc)` を呼ぶ。
  **これが §1-2(出力等価性)を構造的に保証する唯一の方法である。**
  サーバー用に実行ロジックを複製することは禁止。
- 既存の公開 API(`api.single_point` / `relax` / `md`)が既に
  `calculator=` 注入を受け付けている設計と整合する。

### 4.2 リクエスト間の状態隔離

`run_workdir` が触るプロセスグローバル状態を棚卸しし、リクエストスコープに
閉じ込めること。現行コードで判明しているもの:

- `_working_directory(...)` / `_active_pseudo_scf_settings(...)` /
  `_active_vasp_input_paths(...)` — 既にコンテキストマネージャなので、
  そのまま `run_workdir` 内に含まれていれば良い。
- 環境変数 `_CHARGE_ENV_BASE_DIR_VAR` の set/restore — 現在 `cli.main` に
  あるので、抽出時に `run_workdir` 内へ移す。
- **ASE calculator の結果キャッシュ**: リクエスト間で前の構造の結果が
  漏れないよう、各リクエスト開始時に `calculator.reset()`(または同等の
  results クリア)を行う。atoms オブジェクトはリクエストごとに新規に
  読むので共有されないが、calculator 側のキャッシュは明示的に切る。
- モデルの重み自体はリクエスト間で共有してよい(それが本機能の目的)。
  推論は `torch.no_grad` 相当の既存経路のままで、勾配や optimizer 状態が
  蓄積しないことをテストで確認する(§6)。

### 4.3 失敗の隔離と OOM

- リクエスト実行は try/except で包み、例外は `done(ok=false)` +
  traceback としてクライアントに返す。サーバーは継続。
- CUDA OOM を捕捉したら `torch.cuda.empty_cache()` を試みてから継続する
  (torch が import されているバックエンドの場合のみ、ベストエフォート)。
- 連続失敗してもサーバーは自殺しない(ユーザーが `status` の
  `jobs_failed` で気付ける)。ただし失敗はサーバーログに traceback 込みで
  必ず残す。

### 4.4 モジュール構成

- 新規 `src/vpmdk_core/server.py`(Server: ソケット、キュー、ワーカー、
  ライフサイクル)と `src/vpmdk_core/client.py`(Client: connect / run /
  status / stop、CLI から独立して Python API としても使える)を作る。
- CLI サブコマンドはこの 2 モジュールの薄い皮とする(README の
  「Choose Your Entry Point」= CLI と Python API の二本立て構成に合わせる)。
- 依存追加は標準ライブラリのみで実装できるはず(socket / json /
  threading / signal)。新規サードパーティ依存を足さないこと。

## 5. ドキュメント要件

- README に「Server mode」節を追加(起動 → 複数 run → status → stop の
  最短例。複数 GPU の例: `CUDA_VISIBLE_DEVICES=0 vpmdk serve --socket
  /tmp/vpmdk-gpu0.sock &` を GPU ごとに)。
- `docs/` にプロトコル(§3)とセキュリティ前提(同一ユーザー、0700)を
  記した長文ドキュメントを追加。
- `examples/` に「1 サーバー + ディレクトリ群をシェルループで処理」の
  実例を追加。
- CHANGELOG / リリースノート: `feat:` コミット規約に従う。

## 6. テスト要件

AGENTS.md の方針(ユニットは決定的・モック、実バックエンドは
`@pytest.mark.integration`)に従う。

ユニット(スタブ calculator で実行、モデルロード不要):

1. 後方互換: `vpmdk --dir` / 引数なし `vpmdk` が従来通り動く(サブコマンド
   追加後も)。
2. 等価性: 同じ workdir をワンショットとサーバー経由で実行し、
   OUTCAR / OSZICAR / CONTCAR / vasprun.xml がバイト一致する。
3. ライフサイクル: serve → status(idle)→ run → status(jobs_completed=1)
   → stop(graceful)でソケット・pidfile が消える。
4. 直列キュー: 2 クライアント同時 run で FIFO 実行され、両方正しい結果を
   受け取る(スタブ計算に sleep を仕込み、実行が重ならないことを検証)。
5. 失敗の隔離: 例外を投げるスタブで run → クライアントは exit 2 +
   traceback 受領、サーバーは次の run を正常処理。
6. BCAR 不一致: 常駐 MLP=CHGNET のサーバーに MLP=MACE の workdir を投げる
   → exit 5、差分がメッセージに含まれる。非バックエンドタグ
   (VPMDK_WRITE_ENERGY_CSV 等)の差異は拒否されない。
7. stale socket: 死んだソケットファイルが残っていても serve が起動できる。
   生きたサーバーがいる場合は「already running」で exit 1。
8. idle timeout: `--idle-timeout 1` で無リクエスト 1 秒後に自動終了。
9. 接続不能: サーバー不在で run → exit 3(フォールバック実行しない)。
10. 状態隔離: 同一サーバーに構造 A → B → A の順で投げ、1 回目と 3 回目の
    A の結果が一致する(キャッシュ・状態漏れがない)。
11. クライアントタイムアウト: `--timeout` 超過で exit 4。

インテグレーション(`@pytest.mark.integration`、実モデル):

12. 実バックエンド(例: CHGNet 小規模)で serve → 同一構造を 3 回 run し、
    2 回目以降の壁時計時間が 1 回目より大幅に短い(モデルロード償却の実証)
    + ワンショット実行との数値一致。

## 7. 非目標(v1 では実装しない。判断に迷ったらここに戻る)

- 複数モデルの同時常駐・リクエストごとのモデル切替
- バッチ推論エンドポイント(複数構造の一括 forward。将来 v2 候補として
  設計を閉ざさない程度に意識すれば良い)
- TCP / リモートホスト対応、認証
- ジョブの永続化・サーバー再起動をまたぐキュー
- サーバー内での GPU スケジューリング(複数 GPU は複数サーバーで表現)
- Windows サポート

## 8. 受け入れチェックリスト

- [ ] `vpmdk --dir` の挙動が 1 バイトも変わっていない(回帰テストで固定)
- [ ] `vpmdk run --dir D` と `vpmdk --dir D` の出力ファイルがバイト一致
- [ ] 成功時にクライアント stdout に `Calculation completed.` が出る
- [ ] serve は モデルロード完了後にソケットを作る(readiness = ソケット出現)
- [ ] 1 リクエストの失敗でサーバーが落ちない
- [ ] stop / SIGTERM / SIGINT のいずれでもソケットと pidfile が残らない
- [ ] 終了コードが §2.5 の表どおり
- [ ] 新規サードパーティ依存ゼロ
- [ ] README / docs / examples / CHANGELOG 更新
- [ ] `pytest -m "not integration"` 全緑

## 9. 補遺(2026-07-17): クライアント薄化 — 実測に基づく改修依頼

実 CHGNet(CPU)での統合計測により、v1 実装に対する重要な改善点が
判明した。**優先度: 高**(本機能の価値提案そのものに関わる)。

### 症状

- `vpmdk run` の壁時計 ~3.3 秒に対し、サーバー側の計算は一瞬。
  ワンショット `vpmdk --dir` は ~3.45 秒 — **常駐の利得がほぼ消えている**。
- `python -X importtime` での実測: 存在しないソケットへの
  `vpmdk status`(即 exit 3)ですら torch / chgnet / mace / e3nn /
  pymatgen / ase を import(重量モジュール record 1,564 件、累積 ~26 秒分、
  壁時計 ~3 秒)。

### 原因

ルートの `vpmdk.py`(互換シム)が **argv を見る前に**
`importlib.import_module("vpmdk_core")` を実行し、`vpmdk_core/__init__.py`
がトップレベルでバックエンド群(→ torch 一式)を import しているため。
`client.py` 自体は socket/json しか必要としない(§4.4 の設計意図どおり)
のに、そこへの import 経路が ML スタックを引きずり込む。

### 改修内容

1. シム/CLI のエントリで、`vpmdk_core` ロードの**前に** argv 先頭を判定し、
   `run` / `status` / `stop` はパッケージ `__init__` を経由せずに
   `vpmdk_core.client` を直接ロードする(または client を
   トップレベル軽量モジュールに移す)。
2. client モジュールの import は標準ライブラリのみに限定し、
   回帰テストで固定する(client サブコマンド実行後の `sys.modules` に
   torch / e3nn が存在しないこと)。
3. 受け入れ基準: `time vpmdk status` < 0.3 s、`vpmdk run` の
   オーバーヘッド ≈ ソケット往復のみ。終了コード契約(§2.5)は不変。

### 効果

クライアント固定費(~3 s/回)は呼び出し回数に比例して積み上がる
(下流の構造探索では緩和1ステージ=1回の `vpmdk run`。例: 30 個体 ×
20 世代 × 2 ステージ = 1,200 回 ≈ 1 時間の純 import 待ち)。薄化により
小モデルでも常駐の利得(モデルロード償却)がそのまま壁時計に現れる。

## 付録 A(参考情報): 下流ツールからの利用像

vpmdk 本体の仕様には影響しないが、この機能の代表的な消費者として結晶構造
探索フレームワーク(LSPEX)が想定している使い方を記す。仕様が「一般ユーザー
向けとして自然か」を判断する際の具体例として参照のこと。

- 探索フレームワークは run 開始時に GPU ごとに
  `CUDA_VISIBLE_DEVICES=<g> vpmdk serve --socket <path_g> --idle-timeout 3600`
  をサブプロセスとして起動し、`vpmdk status` が応答するまで待つ。
- 各構造の緩和は、計算ディレクトリ(VASP 形式の POSCAR/INCAR/BCAR)を
  用意した上で環境変数 `VPMDK_SOCKET=<path_g>` を付けて `vpmdk run` を
  実行する形で行う(= 従来 `vpmdk` を指定していた起動コマンドを
  `vpmdk run` に置き換えるだけ)。§1-2 の等価性と §1-3 のマーカーが
  この置き換え可能性を保証する。
- 終了・異常終了時に `vpmdk stop` を呼ぶ。呼び損ねた場合の保険が
  `--idle-timeout`(孤児サーバーが VRAM を握り続けない)。
- 終了コード(§2.5)は失敗時リトライ機構の判定に直接使われる。
