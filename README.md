# facedataML-analysis

## 專案簡介

本專案使用 AT&T Face Dataset 進行人臉辨識實驗，重點在比較不同分類模型在原始高維影像特徵與 PCA 降維特徵下的準確率、訓練時間與維度取捨。

資料集中每張人臉影像已整理為 64x64 灰階像素，共 4096 個特徵欄位，並以 `target` 欄位表示人物類別。Notebook 會將資料切分為訓練集與測試集，完成標準化、PCA 解釋變異分析，再比較多種監督式分類模型。

## 專案內容

```text
.
├── att_combined.ipynb
├── requirements.txt
└── data/
    └── face_data.csv
```

- `att_combined.ipynb`：主要分析 notebook，包含資料展示、PCA、模型訓練、GridSearchCV 與結果視覺化。
- `data/face_data.csv`：AT&T 人臉資料，包含 4096 個像素特徵與 `target` 類別欄位。
- `requirements.txt`：執行 notebook 所需的 Python 套件。

## 分析流程

1. 讀取 `data/face_data.csv`，確認人臉影像可還原為 64x64 灰階圖。
2. 使用 `train_test_split` 將資料切為訓練集與測試集，測試比例為 30%。
3. 使用 `StandardScaler` 對特徵做標準化。
4. 使用 PCA 分析累積解釋變異比例，並建立 70%、80%、95% 三組降維資料。
5. 比較 Logistic Regression、SVM 與 MLP/NN 在不同資料版本下的分類表現。
6. 使用 GridSearchCV 搜尋超參數，並以測試準確率、訓練時間、混淆矩陣與 loss curve 進行評估。

## 模型與實驗重點

本作品主要比較下列模型與設定：

- Logistic Regression / LogisticRegressionCV
- Support Vector Machine，包括 linear kernel 與 RBF kernel
- Neural Network / MLPClassifier，包括 ReLU 與 tanh activation
- PCA 70%、PCA 80%、PCA 95% 與原始標準化特徵的差異

實驗觀察顯示，線性模型與 linear SVM 在此高維人臉資料上表現穩定；PCA 80% 與 PCA 95% 通常能在維持接近原始資料準確率的同時明顯降低訓練成本。PCA 70% 的壓縮比例最高，但準確率下降較明顯。

## 主要結果摘要

Notebook 中的代表性測試結果如下：

| 資料版本 | 模型 | 測試準確率 |
| --- | --- | --- |
| 原始資料 | LogisticRegressionCV | 0.9667 |
| 原始資料 | GridSearchCV + Logistic Regression | 0.9667 |
| 原始資料 | SVM linear | 0.9667 |
| PCA 80% | Logistic Regression | 0.9583 |
| PCA 80% | SVC | 0.9417 |
| PCA 95% | GridSearchCV + Logistic Regression | 0.9667 |
| PCA 95% | SVM linear | 0.9667 |
| PCA 95% | SVM RBF | 0.8250 |
| PCA 95% | NN tanh | 0.9083 |

整體而言，PCA 95% 能保留主要辨識資訊，並讓部分模型的訓練時間大幅下降；若以效率和準確率平衡來看，PCA 80% 到 PCA 95% 是較實用的區間。

## 執行方式

建議使用 Python 3.11 或相容版本建立虛擬環境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

啟動 Jupyter Notebook：

```powershell
jupyter notebook att_combined.ipynb
```

或使用 JupyterLab：

```powershell
jupyter lab
```

## 注意事項

- Notebook 會自動處理從專案根目錄或 `data` 目錄執行時的資料路徑定位。
- 請確認 `data/face_data.csv` 存在，否則 notebook 無法讀取資料。
- 部分 GridSearchCV 與 MLP 訓練步驟需要較長時間，執行時間會依 CPU 與環境不同而改變。

## 技術棧

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook
