# RL Lab 1 — TD learning for 2048

以 `2048_sample.cpp` 為基礎，兩個提交用 `.cpp` 均可獨立編譯，不依賴其他原始碼檔。

## 編譯

```powershell
New-Item -ItemType Directory -Force build, models, logs
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic td_state.cpp -o build/td_state.exe
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic td_afterstate.cpp -o build/td_afterstate.exe
```

Linux/macOS 可使用相同 g++ 選項，自行調整執行檔副檔名與建立目錄指令。

## 訓練

每個版本至少 100,000 局。以下固定 seed，分別儲存權重及每局 CSV，每 10,000 局覆寫 checkpoint：

```powershell
./build/td_afterstate.exe --episodes 100000 --alpha 0.1 --seed 42 --stats 1000 --save models/td_afterstate.bin --log logs/td_afterstate_train.csv --checkpoint-every 10000
./build/td_state.exe --episodes 100000 --alpha 0.1 --seed 42 --stats 1000 --save models/td_state.bin --log logs/td_state_train.csv --checkpoint-every 10000
```

可用 `--load` 載入同版本權重繼續訓練。`--episodes` 是這次額外執行的局數，CSV 局數由 1 重新計算；checkpoint 不儲存 RNG 狀態，因此續訓不等同於不中斷訓練。`--log` 會覆寫指定檔案，續訓應使用新紀錄檔名。未指定 `--save` 時，訓練會儲存至目前目錄的 `td_state.bin` 或 `td_afterstate.bin`。

每個版本的四個 6-tuple LUT 共占 256 MiB，另有少量棋盤、軌跡與程序開銷。權重檔也約 256 MiB。權重格式保留 sample 的原生二進位欄位，應在相同架構／工具鏈環境使用；檔頭會檢查 TD 版本，兩版權重不可混用。

## 評估與 Demo

```powershell
./build/td_afterstate.exe --eval --load models/td_afterstate.bin --episodes 1000 --seed 2026 --log logs/td_afterstate_eval.csv
./build/td_state.exe --eval --load models/td_state.bin --episodes 1000 --seed 2026 --log logs/td_state_eval.csv
```

`--eval` 必須提供 `--load`，預設 1,000 局，不更新或儲存權重。程式持續玩到無合法動作；最大方塊至少 2048 即計為達成。終端輸出的 `mean`、`win2048` 是最近一個統計區間；最後 `summary` 是本次全部局數。CSV 欄位為 `episode,score,max_tile,moves,elapsed_seconds`，可用每 1,000 局平均分數繪製兩條訓練曲線。

## 演算法

- `s`：移動前棋盤；`s'`：合併後、popup 前棋盤；`s''`：popup 後下一個棋盤。
- 棋盤仍採 sample 的 64-bit 表示，每格 4 bits，儲存 tile 的指數。新增 2 與 4 的機率分別是 0.7、0.3，空格位置均勻抽樣。
- 使用 sample 的四組 6-tuple：`012345`、`456789`、`012456`、`45689a`；每組使用 8 種旋轉／鏡射，共 32 個 active features。估值為各 LUT 查值總和，每次更新各 active feature 加上 `alpha * TD-error / 32`。對稱位置若查到同一權重，保留每次出現的梯度貢獻。
- TD-state：選擇最大 `r + sum(P(s'') * V(s''))` 的合法動作，枚舉每個空格的 2、4 popup。訓練使用實際取樣的下一個 state，target 為 `r + V(s'')`；若下一個 state 終局，其價值明確設為 0。
- TD-after-state：選擇最大 `r + V(s')` 的合法動作。反向更新時，依**當時的權重**在實際的 `s''` 重新選擇 greedy action，target 為 `r_next + V(s'_next)`；無合法下一步時 target 為 0。不是直接使用軌跡中的舊估值或固定舊動作。
- 每局結束後由最後一步向前更新，與文件的 modified backward training 一致；`alpha=0.1`，公式沒有額外折扣。
- 保留 sample 的棋盤容量限制：可表示最大 tile 32768。若候選移動將生成 65536，程式明確報錯停止，避免 4-bit 溢位靜默破壞棋盤。

## 驗證

需有 Python 3 與 PATH 中的 g++（或以環境變數 `CXX` 指定編譯器）：

```powershell
python tests/run_tests.py
```

測試涵蓋 tuple 索引、對稱與重複索引梯度、popup 機率與終局價值、兩版動作排序、反向 TD-target、目前權重重新選擇動作、權重存取、checkpoint、固定 seed 評估重現性，以及不合法參數／模型檔拒絕。測試會編譯到 `build/`，並移除自己產生的暫存權重與 CSV。

## 遊戲 GUI

已新增 `scripts/play_2048.py`，使用原本 C++ 引擎載入本次訓練權重，僅推論，不修改模型或提交用的兩份 `.cpp`。

```powershell
g++ -std=c++11 -O3 -DGUI_TD_STATE scripts/gui_engine.cpp -o build/gui_td_state.exe
g++ -std=c++11 -O3 scripts/gui_engine.cpp -o build/gui_td_afterstate.exe
python scripts/play_2048.py
```

預設 TD-state，可切換 TD-after-state。方向鍵／WASD 手動操作，AI 單步查看模型決策，自動遊玩可暫停並調整速度；固定 seed 可重現棋局。介面顯示四方向估值、總分、步數與最大方塊結果。`python scripts/play_2048.py --self-test` 可驗證基本互動流程。兩個引擎在 seed 2026 的第一局，分數、步數及最大方塊皆已與正式評估紀錄逐項比對一致。

## 繳交

### 報告編譯

報告已改用 XeLaTeX。可編輯來源為 `output/latex/RL_Lab1_Report.tex`，公式使用 amsmath，TD-backup 圖使用 TikZ。執行 `python scripts/build_report.py` 會編譯兩次，檢查缺字與溢出，再更新 `output/pdf/RL_Lab1_Report.pdf`。腳本會尋找 PATH 或本機 MiKTeX，也可用 `--xelatex PATH` 指定編譯器。

截止：2026/09/27 23:59；Demo：2026/10/06。

提交 `td_state.cpp`、`td_afterstate.cpp` 和實驗報告 PDF，壓縮為 `RL_LAB1_StudentId_Name.zip`，**不包含權重**。權重需另外保留供 Demo 載入。兩版完整 100k 訓練、最終評估與報告已完成，報告位於 `output/pdf/RL_Lab1_Report.pdf`，結果位於 `logs/`。
