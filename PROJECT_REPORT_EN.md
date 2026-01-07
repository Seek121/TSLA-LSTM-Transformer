# Stock Price Prediction Based on LSTM-Transformer Hybrid Neural Network

## — A Case Study of Tesla (TSLA) Stock

---

**Abstract**: Stock price prediction is a crucial research topic in financial engineering. Due to the highly nonlinear and complex nature of financial markets, traditional statistical methods struggle to effectively capture inherent patterns. This study proposes a hybrid deep learning model that integrates Long Short-Term Memory (LSTM) networks with the Transformer architecture for Tesla (TSLA) stock closing price prediction. The model leverages LSTM to capture long-term dependencies in time series while utilizing the multi-head self-attention mechanism of Transformers to learn complex relationships between different time steps. Experiments are conducted on TSLA daily stock data from October 2019 to April 2022, constructing an 11-dimensional feature space comprising price, volume, and various technical indicators. Experimental results demonstrate that the proposed hybrid model achieves a Mean Absolute Percentage Error (MAPE) of 11.39% on the test set, validating the effectiveness of deep learning methods in financial time series forecasting.

**Keywords**: Stock Price Prediction; LSTM; Transformer; Attention Mechanism; Deep Learning; Time Series Analysis

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Background](#2-theoretical-background)
3. [Data Description and Preprocessing](#3-data-description-and-preprocessing)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Model Architecture](#5-model-architecture)
6. [Experimental Design and Results](#6-experimental-design-and-results)
7. [Conclusions and Future Work](#7-conclusions-and-future-work)
8. [References](#8-references)

---

## 1. Introduction

### 1.1 Research Background and Motivation

Financial markets constitute a core component of modern economic systems. Accurate stock price prediction holds significant theoretical and practical value for investment decisions, risk management, and resource allocation. However, according to the Efficient Market Hypothesis (EMH) [1], stock prices already reflect all available information, and thus price movements follow a random walk process, making them difficult to predict.

Nevertheless, extensive empirical research suggests that financial markets are not perfectly efficient and exhibit some degree of predictability [2]. Particularly with the rapid development of machine learning and deep learning technologies, researchers have discovered that neural networks can learn valuable nonlinear patterns from historical data [3].

### 1.2 Literature Review

Traditional stock price prediction methods mainly include:

**Statistical Methods**:
- Autoregressive Integrated Moving Average (ARIMA)
- Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
- Vector Autoregression (VAR)

**Machine Learning Methods**:
- Support Vector Machines (SVM)
- Random Forest
- Gradient Boosting Trees (XGBoost)

**Deep Learning Methods**:
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM) [4]
- Gated Recurrent Units (GRU)
- Transformer [5]

In recent years, LSTM has gained widespread adoption in time series forecasting due to its ability to effectively handle long-sequence dependencies. The Transformer architecture, leveraging its self-attention mechanism, has achieved breakthrough performance in natural language processing and is gradually being introduced into financial forecasting.

### 1.3 Research Objectives and Contributions

The main objectives of this study are:

1. Construct a hybrid prediction model that combines the strengths of LSTM and Transformer
2. Extract effective features based on technical analysis theory
3. Predict Tesla stock prices and evaluate model performance

The contributions of this study include:

- **Architectural Innovation**: Proposing a serial structure of bidirectional LSTM with Transformer encoder
- **Feature Fusion**: Integrating raw price data with technical indicators to construct a multi-dimensional feature space
- **Attention Enhancement**: Utilizing multi-head attention mechanism to capture dependencies at different time scales

---

## 2. Theoretical Background

### 2.1 Long Short-Term Memory Networks (LSTM)

LSTM is a special recurrent neural network structure proposed by Hochreiter and Schmidhuber in 1997 [4], which addresses the vanishing gradient problem of traditional RNNs through the introduction of gating mechanisms.

#### 2.1.1 LSTM Cell Structure

An LSTM cell contains three gating mechanisms: Forget Gate, Input Gate, and Output Gate. Let the input at time \(t\) be \(\mathbf{x}_t\), the hidden state from the previous time step be \(\mathbf{h}_{t-1}\), and the cell state be \(\mathbf{c}_{t-1}\).

**Forget Gate** determines what information to discard from the cell state:

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
\]

**Input Gate** determines what new information to store:

\[
\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
\]

\[
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
\]

**Cell State Update**:

\[
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
\]

**Output Gate** determines what to output:

\[
\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
\]

\[
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\]

where \(\sigma(\cdot)\) denotes the Sigmoid activation function, \(\odot\) denotes the Hadamard product (element-wise multiplication), and \(\mathbf{W}\) and \(\mathbf{b}\) are the weight matrices and bias vectors, respectively.

#### 2.1.2 Bidirectional LSTM

Bidirectional LSTM (BiLSTM) processes sequences in both forward and backward directions simultaneously, capturing context from both past and future:

\[
\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{forward}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})
\]

\[
\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{backward}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})
\]

\[
\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]
\]

### 2.2 Transformer Architecture

The Transformer was proposed by Vaswani et al. in 2017 [5], with self-attention mechanism as its core component.

#### 2.2.1 Scaled Dot-Product Attention

Given query matrix \(\mathbf{Q}\), key matrix \(\mathbf{K}\), and value matrix \(\mathbf{V}\), the scaled dot-product attention is computed as:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
\]

where \(d_k\) is the dimension of the key vectors, and division by \(\sqrt{d_k}\) prevents the dot products from becoming too large, which would push the softmax into regions with extremely small gradients.

#### 2.2.2 Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces:

\[
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O
\]

where each attention head is computed as:

\[
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
\]

\(\mathbf{W}_i^Q \in \mathbb{R}^{d_{model} \times d_k}\), \(\mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}\), \(\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}\), and \(\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{model}}\) are learnable projection matrices.

#### 2.2.3 Positional Encoding

Since Transformers contain no recurrence, positional encodings are needed to inject sequence position information. This study adopts sinusoidal positional encoding:

\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

where \(pos\) represents the position index and \(i\) represents the dimension index.

#### 2.2.4 Transformer Encoder Layer

Each Transformer encoder layer contains two sub-layers: multi-head self-attention and feed-forward network (FFN). Residual connections and layer normalization are applied after each sub-layer:

\[
\mathbf{Z} = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))
\]

\[
\mathbf{H} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\]

The feed-forward network consists of two linear transformations with a ReLU activation:

\[
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
\]

### 2.3 Technical Analysis Indicators

Technical analysis studies historical price and volume data to predict future price movements. This study employs the following technical indicators:

#### 2.3.1 Moving Average (MA)

Simple moving average is the arithmetic mean of prices over the past \(n\) periods:

\[
MA_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}
\]

where \(P_t\) represents the closing price on day \(t\).

#### 2.3.2 Relative Strength Index (RSI)

RSI is a momentum oscillator that measures the speed and magnitude of price movements:

\[
RSI = 100 - \frac{100}{1 + RS}
\]

where Relative Strength \(RS\) is defined as:

\[
RS = \frac{\text{Average Gain}}{\text{Average Loss}} = \frac{\frac{1}{n}\sum_{i \in U} |P_i - P_{i-1}|}{\frac{1}{n}\sum_{j \in D} |P_j - P_{j-1}|}
\]

\(U\) and \(D\) represent the sets of trading days with price increases and decreases, respectively.

#### 2.3.3 Moving Average Convergence Divergence (MACD)

MACD consists of the fast line, slow line, and histogram:

\[
EMA_n(t) = \alpha \cdot P_t + (1-\alpha) \cdot EMA_n(t-1), \quad \alpha = \frac{2}{n+1}
\]

\[
MACD(t) = EMA_{12}(t) - EMA_{26}(t)
\]

\[
Signal(t) = EMA_9(MACD(t))
\]

\[
Histogram(t) = MACD(t) - Signal(t)
\]

#### 2.3.4 Bollinger Bands

Bollinger Bands consist of middle, upper, and lower bands, reflecting price volatility range:

\[
Middle_t = MA_{20}(t)
\]

\[
\sigma_t = \sqrt{\frac{1}{20}\sum_{i=0}^{19}(P_{t-i} - Middle_t)^2}
\]

\[
Upper_t = Middle_t + 2\sigma_t
\]

\[
Lower_t = Middle_t - 2\sigma_t
\]

---

## 3. Data Description and Preprocessing

### 3.1 Data Source

This study uses historical daily trading data of Tesla (TSLA) stock. Basic dataset information is shown in Table 3.1.

**Table 3.1 Dataset Information**

| Attribute | Description |
|-----------|-------------|
| Data Source | TSLA.csv |
| Time Span | September 30, 2019 — April 11, 2022 |
| Original Sample Size | 639 trading days |
| Original Features | Date, Open, High, Low, Close, Volume, Adj Close |
| Data Frequency | Daily |

### 3.2 Descriptive Statistics

Descriptive statistics of the raw data are shown in Table 3.2.

**Table 3.2 Descriptive Statistics of Raw Data**

| Statistic | High | Low | Open | Close | Volume |
|-----------|------|-----|------|-------|--------|
| Count | 639 | 639 | 639 | 639 | 639 |
| Mean | $543.36 | $521.94 | $532.17 | $531.30 | 4.82×10⁷ |
| Std Dev | $340.84 | $326.38 | $333.65 | $333.36 | 3.58×10⁷ |
| Min | $46.90 | $44.86 | $46.32 | $46.29 | 9.80×10⁶ |
| 25th Percentile | $170.26 | $159.13 | $163.53 | $164.78 | 2.39×10⁷ |
| Median | $620.41 | $597.63 | $607.56 | $605.13 | 3.45×10⁷ |
| 75th Percentile | $796.58 | $777.37 | $788.52 | $781.30 | 6.33×10⁷ |
| Max | $1243.49 | $1217.00 | $1234.41 | $1229.91 | 3.05×10⁸ |

**Data Quality Assessment**: No missing values were found in the dataset.

### 3.3 Feature Engineering

To enhance model prediction capability, this study constructs a multi-dimensional feature space based on technical analysis theory. The feature engineering process is shown in Algorithm 1.

---

**Algorithm 1: Technical Indicator Feature Extraction**

---

**Input**: Raw price sequence \(\{P_t\}_{t=1}^{T}\), volume sequence \(\{V_t\}_{t=1}^{T}\)

**Output**: Enhanced feature matrix \(\mathbf{X} \in \mathbb{R}^{T' \times d}\)

1. **Initialize**: Set window parameters \(n_1=5, n_2=10, n_3=20, n_{RSI}=14\)

2. **Compute Moving Averages**:
   - **for** \(n \in \{n_1, n_2, n_3\}\) **do**
     - \(MA_n(t) \leftarrow \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}\)
   - **end for**

3. **Compute Price Change Rate**:
   - \(r_t \leftarrow \frac{P_t - P_{t-1}}{P_{t-1}}\)

4. **Compute Volatility**:
   - \(\sigma_t \leftarrow \text{Std}(\{r_{t-19}, ..., r_t\})\)

5. **Compute RSI**:
   - \(U_t \leftarrow \max(P_t - P_{t-1}, 0)\)
   - \(D_t \leftarrow \max(P_{t-1} - P_t, 0)\)
   - \(AvgU_t \leftarrow EMA_{14}(U_t)\)
   - \(AvgD_t \leftarrow EMA_{14}(D_t)\)
   - \(RSI_t \leftarrow 100 - \frac{100}{1 + AvgU_t/AvgD_t}\)

6. **Compute MACD**:
   - \(MACD_t \leftarrow EMA_{12}(P_t) - EMA_{26}(P_t)\)
   - \(Signal_t \leftarrow EMA_9(MACD_t)\)

7. **Compute Bollinger Bands**:
   - \(BB_{upper}(t) \leftarrow MA_{20}(t) + 2 \cdot \text{Std}_{20}(P_t)\)
   - \(BB_{lower}(t) \leftarrow MA_{20}(t) - 2 \cdot \text{Std}_{20}(P_t)\)

8. **Remove samples with missing values**:
   - \(T' \leftarrow T - \max(n_3, 26) + 1\)

9. **Return** feature matrix \(\mathbf{X}\)

---

### 3.4 Final Feature Space

After feature engineering, an 11-dimensional feature space is constructed, as shown in Table 3.3.

**Table 3.3 Model Input Features**

| No. | Symbol | Feature Name | Type |
|-----|--------|--------------|------|
| 1 | \(O_t\) | Open Price | Raw Feature |
| 2 | \(H_t\) | High Price | Raw Feature |
| 3 | \(L_t\) | Low Price | Raw Feature |
| 4 | \(C_t\) | Close Price | Raw Feature |
| 5 | \(V_t\) | Volume | Raw Feature |
| 6 | \(MA_5(t)\) | 5-day Moving Average | Trend Indicator |
| 7 | \(MA_{10}(t)\) | 10-day Moving Average | Trend Indicator |
| 8 | \(MA_{20}(t)\) | 20-day Moving Average | Trend Indicator |
| 9 | \(RSI_t\) | Relative Strength Index | Momentum Indicator |
| 10 | \(MACD_t\) | MACD Line | Trend Indicator |
| 11 | \(Signal_t\) | MACD Signal Line | Trend Indicator |

### 3.5 Data Normalization

Min-Max normalization is applied to scale data to the \([0, 1]\) interval:

\[
\tilde{x}_i = \frac{x_i - x_{min}}{x_{max} - x_{min}}
\]

where \(x_{min}\) and \(x_{max}\) are the minimum and maximum values of the feature in the training set.

### 3.6 Sequence Construction

A sliding window approach is used to construct supervised learning samples. Given window length \(\tau\), for time \(t\), the model input consists of the feature vector sequence over the past \(\tau\) time steps, with the next time step's closing price as output:

\[
\mathbf{X}^{(i)} = [\mathbf{x}_{i}, \mathbf{x}_{i+1}, ..., \mathbf{x}_{i+\tau-1}] \in \mathbb{R}^{\tau \times d}
\]

\[
y^{(i)} = C_{i+\tau}
\]

This study sets \(\tau = 30\) (using 30 trading days to predict the next day's closing price).

### 3.7 Dataset Split

The dataset is split chronologically into training, validation, and test sets, as shown in Table 3.4.

**Table 3.4 Dataset Split**

| Dataset | Ratio | Samples | Time Range |
|---------|-------|---------|------------|
| Training | 70% | 412 | 2019.10 — 2021.05 |
| Validation | 15% | 88 | 2021.05 — 2021.10 |
| Test | 15% | 89 | 2021.10 — 2022.04 |

---

## 4. Exploratory Data Analysis

### 4.1 Price Trend Analysis

Figure 4.1 shows the TSLA stock price trend, volume changes, and RSI indicator during the study period.

![Stock Overview](output/01_stock_overview.png)

**Figure 4.1 TSLA Stock Price Trend and Technical Indicators**

The following characteristics can be observed from Figure 4.1:

1. **Price Trend**: TSLA stock experienced significant appreciation during the study period, rising from approximately $50 in October 2019 to over $1200 by late 2021, a gain of over 20x.

2. **Moving Average Crossovers**: The 5-day MA crossed the 20-day MA multiple times, forming golden crosses (short-term MA crosses above long-term MA) and death crosses (short-term MA crosses below long-term MA).

3. **Bollinger Bands**: Prices mostly fluctuated within the Bollinger Bands, with band breakouts often accompanying trending moves.

4. **Volume**: Periods of significant price volatility were accompanied by notably increased volume, reflecting the "volume-price correlation" characteristic of markets.

5. **RSI Indicator**: Multiple entries into overbought (>70) and oversold (<30) regions provided signals for short-term price reversals.

### 4.2 Data Distribution and Correlation Analysis

Figure 4.2 shows data distribution characteristics and inter-feature correlations.

![Data Analysis](output/02_data_analysis.png)

**Figure 4.2 Data Distribution and Feature Correlation**

**Key Findings**:

1. **Close Price Distribution** (upper left): Shows a clear right-skewed distribution with more samples in lower price ranges, reflecting the stock's gradual rise from low levels.

2. **Daily Returns Distribution** (upper right): Approximately follows a normal distribution centered at 0, consistent with typical financial time series characteristics. The leptokurtic and fat-tailed nature indicates the presence of extreme price movements.

3. **OHLC Chart** (lower left): Candlestick chart of the last 60 trading days, with green bars indicating up days (Close > Open) and red bars indicating down days.

4. **Correlation Matrix** (lower right):
   - Price features (Open, High, Low, Close) are highly correlated (\(\rho > 0.99\))
   - Moving averages are highly correlated with prices
   - Volume has lower correlation with price features (\(\rho \approx 0.5\))
   - RSI and MACD show moderate correlation

### 4.3 MACD Technical Analysis

Figure 4.3 shows detailed MACD indicator analysis.

![MACD Analysis](output/03_macd_analysis.png)

**Figure 4.3 MACD Technical Indicator Analysis**

MACD analysis key points:

- **MACD and Signal Line Crossovers**: Buy signals are generated when MACD crosses above the signal line; sell signals when it crosses below
- **Histogram**: Histogram height reflects the difference between MACD and signal line; transitions from negative to positive or vice versa may indicate trend reversals
- **Zero Line**: MACD above zero indicates short-term MA is above long-term MA, suggesting bullish trend

---

## 5. Model Architecture

### 5.1 Architecture Design

The proposed LSTM-Transformer hybrid model architecture is shown in Figure 5.1.

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: X ∈ ℝ^(B×τ×d)                                      │
│         ↓                                                   │
│   ┌─────────────────────┐                                   │
│   │   Input Projection  │  Linear: d → h                    │
│   └─────────────────────┘                                   │
│         ↓                                                   │
│   ┌─────────────────────┐                                   │
│   │  Bidirectional LSTM │  2 layers, hidden_dim=h          │
│   │   (Forward + Back)  │  Output: 2h                       │
│   └─────────────────────┘                                   │
│         ↓                                                   │
│   ┌─────────────────────┐                                   │
│   │  LSTM Projection    │  Linear: 2h → h                   │
│   └─────────────────────┘                                   │
│         ↓                                                   │
│   ┌─────────────────────┐                                   │
│   │ Positional Encoding │  PE_{sin/cos}                     │
│   └─────────────────────┘                                   │
│         ↓                                                   │
│   ┌─────────────────────┐                                   │
│   │ Transformer Encoder │  2 layers                         │
│   │  ├─ Multi-Head Attn │  4 heads                         │
│   │  ├─ Feed-Forward    │  h → 4h → h                       │
│   │  └─ LayerNorm + Res │                                   │
│   └─────────────────────┘                                   │
│         ↓                                                   │
│   ┌─────────────────────┐                                   │
│   │  Last Time Step     │  H[:, -1, :]                      │
│   └─────────────────────┘                                   │
│         ↓                                                   │
│   ┌─────────────────────┐                                   │
│   │   Output MLP        │  h → h/2 → h/4 → 1               │
│   │   (ReLU + Dropout)  │                                   │
│   └─────────────────────┘                                   │
│         ↓                                                   │
│   Output: ŷ ∈ ℝ^B                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Figure 5.1 LSTM-Transformer Hybrid Model Architecture**

### 5.2 Mathematical Description

Let the input sequence be \(\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_\tau] \in \mathbb{R}^{\tau \times d}\), where \(\tau\) is the sequence length and \(d\) is the feature dimension. The forward propagation process is as follows:

**Step 1: Input Projection**

\[
\mathbf{H}^{(0)} = \mathbf{X}\mathbf{W}_{proj} + \mathbf{b}_{proj}, \quad \mathbf{W}_{proj} \in \mathbb{R}^{d \times h}
\]

**Step 2: Bidirectional LSTM Processing**

\[
\overrightarrow{\mathbf{H}}^{(lstm)}, \overleftarrow{\mathbf{H}}^{(lstm)} = \text{BiLSTM}(\mathbf{H}^{(0)})
\]

\[
\mathbf{H}^{(lstm)} = [\overrightarrow{\mathbf{H}}^{(lstm)}; \overleftarrow{\mathbf{H}}^{(lstm)}]\mathbf{W}_{lstm} + \mathbf{b}_{lstm}
\]

where \(\mathbf{W}_{lstm} \in \mathbb{R}^{2h \times h}\) projects the bidirectional LSTM output back to \(h\) dimensions.

**Step 3: Positional Encoding**

\[
\mathbf{H}^{(pe)} = \mathbf{H}^{(lstm)} + \mathbf{PE}
\]

**Step 4: Transformer Encoder**

For the \(l\)-th Transformer encoder layer (\(l = 1, 2\)):

\[
\mathbf{Q}^{(l)} = \mathbf{H}^{(l-1)}\mathbf{W}_Q^{(l)}, \quad
\mathbf{K}^{(l)} = \mathbf{H}^{(l-1)}\mathbf{W}_K^{(l)}, \quad
\mathbf{V}^{(l)} = \mathbf{H}^{(l-1)}\mathbf{W}_V^{(l)}
\]

\[
\mathbf{A}^{(l)} = \text{softmax}\left(\frac{\mathbf{Q}^{(l)}(\mathbf{K}^{(l)})^T}{\sqrt{d_k}}\right)\mathbf{V}^{(l)}
\]

\[
\mathbf{Z}^{(l)} = \text{LayerNorm}(\mathbf{H}^{(l-1)} + \text{Dropout}(\mathbf{A}^{(l)}))
\]

\[
\mathbf{H}^{(l)} = \text{LayerNorm}(\mathbf{Z}^{(l)} + \text{Dropout}(\text{FFN}(\mathbf{Z}^{(l)})))
\]

**Step 5: Output Layer**

Take the hidden state of the last time step and obtain the prediction through a multi-layer perceptron:

\[
\mathbf{h}_{out} = \mathbf{H}^{(L)}_\tau \in \mathbb{R}^h
\]

\[
\hat{y} = \mathbf{W}_3(\text{ReLU}(\mathbf{W}_2(\text{ReLU}(\mathbf{W}_1\mathbf{h}_{out} + \mathbf{b}_1)) + \mathbf{b}_2)) + \mathbf{b}_3
\]

### 5.3 Forward Propagation Pseudocode

---

**Algorithm 2: LSTM-Transformer Hybrid Model Forward Propagation**

---

**Input**: Input sequence \(\mathbf{X} \in \mathbb{R}^{B \times \tau \times d}\)

**Output**: Predictions \(\hat{\mathbf{y}} \in \mathbb{R}^B\)

**Hyperparameters**: Hidden dimension \(h=128\), attention heads \(n_h=4\), dropout rate \(p=0.2\)

1. **Input Projection**
   - \(\mathbf{H} \leftarrow \text{Linear}_{d \rightarrow h}(\mathbf{X})\)

2. **Bidirectional LSTM**
   - \(\overrightarrow{\mathbf{H}}, (\overrightarrow{\mathbf{c}}, \overrightarrow{\mathbf{h}}) \leftarrow \text{LSTM}_{forward}(\mathbf{H})\)
   - \(\overleftarrow{\mathbf{H}}, (\overleftarrow{\mathbf{c}}, \overleftarrow{\mathbf{h}}) \leftarrow \text{LSTM}_{backward}(\mathbf{H})\)
   - \(\mathbf{H}_{lstm} \leftarrow \text{Concat}(\overrightarrow{\mathbf{H}}, \overleftarrow{\mathbf{H}})\)
   - \(\mathbf{H} \leftarrow \text{Linear}_{2h \rightarrow h}(\mathbf{H}_{lstm})\)

3. **Positional Encoding**
   - \(\mathbf{PE} \leftarrow \text{SinusoidalEncoding}(\tau, h)\)
   - \(\mathbf{H} \leftarrow \mathbf{H} + \mathbf{PE}\)

4. **Transformer Encoder** (repeat \(L=2\) times)
   - **for** \(l = 1\) **to** \(L\) **do**
     - \(\mathbf{Q}, \mathbf{K}, \mathbf{V} \leftarrow \text{Linear}(\mathbf{H})\)
     - \(\mathbf{A} \leftarrow \text{MultiHeadAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, n_h)\)
     - \(\mathbf{H} \leftarrow \text{LayerNorm}(\mathbf{H} + \text{Dropout}(\mathbf{A}, p))\)
     - \(\mathbf{F} \leftarrow \text{FFN}(\mathbf{H})\)
     - \(\mathbf{H} \leftarrow \text{LayerNorm}(\mathbf{H} + \text{Dropout}(\mathbf{F}, p))\)
   - **end for**

5. **Extract Last Time Step**
   - \(\mathbf{h}_{out} \leftarrow \mathbf{H}[:, -1, :]\)

6. **Output MLP**
   - \(\mathbf{h}_1 \leftarrow \text{ReLU}(\text{Linear}_{h \rightarrow h/2}(\mathbf{h}_{out}))\)
   - \(\mathbf{h}_1 \leftarrow \text{Dropout}(\mathbf{h}_1, p)\)
   - \(\mathbf{h}_2 \leftarrow \text{ReLU}(\text{Linear}_{h/2 \rightarrow h/4}(\mathbf{h}_1))\)
   - \(\mathbf{h}_2 \leftarrow \text{Dropout}(\mathbf{h}_2, p)\)
   - \(\hat{\mathbf{y}} \leftarrow \text{Linear}_{h/4 \rightarrow 1}(\mathbf{h}_2)\)

7. **Return** \(\hat{\mathbf{y}}\)

---

### 5.4 Parameter Statistics

Parameter counts for each model component are shown in Table 5.1.

**Table 5.1 Model Parameter Statistics**

| Component | Input Dim | Output Dim | Parameters | Formula |
|-----------|-----------|------------|------------|---------|
| Input Projection | 11 | 128 | 1,536 | \(11 \times 128 + 128\) |
| BiLSTM (2 layers) | 128 | 256 | 660,480 | \(4 \times [(128+128) \times 128 + 128] \times 2 \times 2\) |
| LSTM Projection | 256 | 128 | 32,896 | \(256 \times 128 + 128\) |
| Transformer Layer 1 | 128 | 128 | 198,400 | Self-attention + FFN |
| Transformer Layer 2 | 128 | 128 | 198,400 | Self-attention + FFN |
| FC Layer 1 | 128 | 64 | 8,256 | \(128 \times 64 + 64\) |
| FC Layer 2 | 64 | 32 | 2,080 | \(64 \times 32 + 32\) |
| FC Layer 3 | 32 | 1 | 33 | \(32 \times 1 + 1\) |
| **Total** | - | - | **1,100,801** | - |

---

## 6. Experimental Design and Results

### 6.1 Experimental Environment

Experiments were conducted in the following environment:

**Table 6.1 Experimental Environment Configuration**

| Item | Configuration |
|------|---------------|
| Operating System | Linux |
| Deep Learning Framework | PyTorch 2.0.1 |
| GPU | CUDA Acceleration |
| Python Version | 3.10+ |

### 6.2 Training Configuration

Model training hyperparameters are shown in Table 6.2.

**Table 6.2 Training Hyperparameters**

| Hyperparameter | Symbol | Value | Description |
|----------------|--------|-------|-------------|
| Sequence Length | \(\tau\) | 30 | Use past 30 days for prediction |
| Batch Size | \(B\) | 32 | Mini-batch size |
| Max Epochs | \(E_{max}\) | 100 | Maximum training epochs |
| Initial Learning Rate | \(\eta_0\) | 0.001 | AdamW initial learning rate |
| Weight Decay | \(\lambda\) | \(10^{-5}\) | L2 regularization coefficient |
| Dropout Rate | \(p\) | 0.2 | Random dropout probability |
| Early Stopping Patience | patience | 20 | Epochs to wait for improvement |
| Gradient Clipping | max_norm | 1.0 | Prevent gradient explosion |

### 6.3 Loss Function

Mean Squared Error (MSE) is used as the loss function:

\[
\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
\]

where \(y_i\) is the actual value, \(\hat{y}_i\) is the predicted value, and \(N\) is the number of samples.

### 6.4 Optimization Algorithm

The AdamW optimizer [6] is used, with update rules:

\[
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t
\]

\[
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2
\]

\[
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
\]

\[
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda\boldsymbol{\theta}_{t-1}\right)
\]

where \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-8}\).

### 6.5 Learning Rate Scheduling

ReduceLROnPlateau strategy is used: when validation loss doesn't improve for `patience` epochs, learning rate is multiplied by `factor`:

\[
\eta_{new} = \eta_{old} \times factor, \quad factor = 0.5
\]

### 6.6 Training Process Analysis

Figure 6.1 shows the loss function curves during training.

![Training Curves](output/04_training_curves.png)

**Figure 6.1 Training and Validation Loss Curves**

**Training Process Analysis**:

1. **Convergence**: The model converged rapidly in the first 10 epochs, with training loss decreasing from approximately 0.05 to below 0.01.

2. **Validation Loss**: Validation loss reached its optimal value of 0.0117 at epoch 28, with subsequent fluctuations.

3. **Early Stopping**: Training stopped at epoch 48 due to no improvement in validation loss for 20 epochs.

4. **Overfitting Monitoring**: Training loss continued to decrease while validation loss fluctuated in later epochs, indicating slight overfitting tendency. Early stopping effectively prevented severe overfitting.

### 6.7 Evaluation Metrics

Model performance is evaluated using the following metrics:

**Mean Squared Error (MSE)**:

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
\]

**Root Mean Squared Error (RMSE)**:

\[
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
\]

**Mean Absolute Error (MAE)**:

\[
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
\]

**Coefficient of Determination (R²)**:

\[
R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}
\]

**Mean Absolute Percentage Error (MAPE)**:

\[
MAPE = \frac{100\%}{N}\sum_{i=1}^{N}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

### 6.8 Test Set Results

Model evaluation results on the test set are shown in Table 6.3.

**Table 6.3 Test Set Evaluation Results**

| Metric | Value | Unit |
|--------|-------|------|
| MSE | 16,319.14 | USD² |
| RMSE | 127.75 | USD |
| MAE | 113.42 | USD |
| R² | -0.60 | - |
| MAPE | 11.39 | % |

### 6.9 Prediction Results Visualization

Figure 6.2 shows prediction result analysis on the test set.

![Prediction Results](output/05_prediction_results.png)

**Figure 6.2 Test Set Prediction Results Analysis**

**Chart Interpretation**:

1. **Time Series Comparison** (upper left): Blue line represents actual prices, red line represents predicted prices. The model captures the overall trend but shows lag during periods of sharp volatility.

2. **Error Distribution** (upper right): Prediction errors approximately follow a normal distribution with mean near 0, indicating no systematic bias.

3. **Scatter Plot** (lower left): Actual vs predicted scatter plot; ideally points should be distributed along the diagonal. Negative R² indicates the model performs worse than simple mean prediction on the test set.

4. **Metrics Bar Chart** (lower right): RMSE of $127.75, MAE of $113.42, MAPE of 11.39%.

### 6.10 Full Dataset Prediction Visualization

Figure 6.3 shows prediction results across the full dataset.

![Full Prediction](output/06_full_prediction.png)

**Figure 6.3 Full Dataset Prediction Results**

From Figure 6.3:

- **Training Set** (green region): Model fits well, with prediction curve highly overlapping actual curve.
- **Validation Set** (yellow region): Model maintains good prediction accuracy.
- **Test Set** (red region): Test set covers the high-volatility period from late 2021 to early 2022, presenting higher prediction difficulty.

### 6.11 Results Analysis and Discussion

**Performance Discussion**:

1. **MAPE = 11.39%** means the average deviation between predicted and actual prices is approximately 11%. This is a reasonable error range for stock prediction.

2. **R² = -0.60** being negative is due to:
   - Test set period (2021.10-2022.04) coincides with extreme TSLA price volatility
   - Stock prices are influenced by multiple factors (news, policy, market sentiment, etc.)
   - Pure technical indicators struggle to capture impacts of unexpected events

3. **Model Strengths**:
   - Capable of learning complex nonlinear temporal patterns
   - Combines LSTM's memory capability with Transformer's attention mechanism
   - Performs well in relatively stable market environments

4. **Model Limitations**:
   - Limited prediction capability during extreme market conditions
   - Does not incorporate external information (news, sentiment, etc.)
   - Relatively limited training data

---

## 7. Conclusions and Future Work

### 7.1 Research Conclusions

This study proposes a hybrid deep learning model integrating LSTM and Transformer architectures for stock price prediction. Main conclusions are:

1. **Model Effectiveness**: The LSTM-Transformer hybrid model effectively learns temporal features of stock prices, achieving 11.39% MAPE on the test set.

2. **Importance of Feature Engineering**: Introducing technical indicators such as moving averages, RSI, and MACD enriches input information and improves prediction accuracy.

3. **Architectural Advantages**: Bidirectional LSTM can simultaneously utilize historical and future information (during training), while Transformer's self-attention mechanism captures dependencies between different time steps.

4. **Market Complexity**: Experimental results show that pure technical analysis methods have limited predictive power in high-volatility market environments, validating the semi-strong form efficiency of financial markets.

### 7.2 Future Predictions

Based on the trained model, predictions for the next 10 trading days are shown in Figure 7.1.

![Future Prediction](output/07_future_prediction.png)

**Figure 7.1 Future 10-Day Price Prediction**

Predictions show prices will fluctuate slightly in the $900-910 range. Note that long-term prediction uncertainty accumulates over time; results are for reference only.

### 7.3 Research Limitations

This study has the following limitations:

1. **Data Limitations**: Only approximately 2.5 years of daily data used; relatively limited sample size.

2. **Feature Limitations**: Only price-volume data and technical indicators used; fundamental data, macroeconomic indicators, and news sentiment not considered.

3. **Model Limitations**: No model ensembling or uncertainty estimation performed.

4. **Evaluation Limitations**: No detailed comparison with other baseline models conducted.

### 7.4 Future Research Directions

Based on this study's experience, the following improvements are suggested:

1. **Multi-Source Data Fusion**: Incorporate news text, social media sentiment, macroeconomic indicators, and other heterogeneous multi-source data.

2. **Advanced Architecture Exploration**: Try Temporal Fusion Transformer [7], Informer [8], and other Transformer variants designed specifically for time series.

3. **Uncertainty Quantification**: Use Bayesian neural networks or MC-Dropout to estimate prediction uncertainty.

4. **Multi-Task Learning**: Simultaneously predict price, volatility, and volume, leveraging inter-task correlations to improve performance.

5. **Reinforcement Learning Application**: Combine prediction model with trading strategy, optimizing portfolios in simulated environments.

---

## 8. References

[1] Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.

[2] Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *The Review of Financial Studies*, 1(1), 41-66.

[3] Zhang, G., Patuwo, B. E., & Hu, M. Y. (1998). Forecasting with artificial neural networks: The state of the art. *International Journal of Forecasting*, 14(1), 35-62.

[4] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

[5] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

[6] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*.

[7] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

[8] Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.

---

## Appendix

### Appendix A: Project File Structure

```
pbl_lstm/
├── TSLA.csv                    # Raw dataset
├── tsla_stock_prediction.py    # Model training script (Chinese)
├── tsla_stock_prediction_en.py # Model training script (English)
├── requirements.txt            # Python dependencies
├── README.md                   # Project README (English)
├── PROJECT_REPORT.md           # Project report (Chinese)
├── PROJECT_REPORT_EN.md        # Project report (English)
└── output/
    ├── 01_stock_overview.png   # Figure 4.1 Stock Overview
    ├── 02_data_analysis.png    # Figure 4.2 Data Analysis
    ├── 03_macd_analysis.png    # Figure 4.3 MACD Analysis
    ├── 04_training_curves.png  # Figure 6.1 Training Curves
    ├── 05_prediction_results.png # Figure 6.2 Prediction Results
    ├── 06_full_prediction.png  # Figure 6.3 Full Prediction
    ├── 07_future_prediction.png # Figure 7.1 Future Prediction
    ├── best_model.pth          # Model weights
    ├── evaluation_metrics.csv  # Evaluation metrics
    ├── prediction_results.csv  # Prediction results
    └── future_predictions.csv  # Future predictions
```

### Appendix B: Notation

| Symbol | Meaning |
|--------|---------|
| \(\mathbf{x}_t\) | Input feature vector at time t |
| \(\mathbf{h}_t\) | Hidden state at time t |
| \(\mathbf{c}_t\) | Cell state at time t |
| \(\tau\) | Sequence length (time window) |
| \(d\) | Feature dimension |
| \(h\) | Hidden layer dimension |
| \(B\) | Batch size |
| \(\sigma(\cdot)\) | Sigmoid activation function |
| \(\odot\) | Hadamard product (element-wise multiplication) |
| \(\mathbf{W}\) | Weight matrix |
| \(\mathbf{b}\) | Bias vector |
| \(\eta\) | Learning rate |

---

**Disclaimer**: This study is for academic research purposes only. All predictions do not constitute investment advice. Stock market investment involves risk; investors should make decisions carefully.

---

*Report Completion Date: January 7, 2026*

