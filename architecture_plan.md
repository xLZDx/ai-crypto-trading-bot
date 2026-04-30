### **Единый Мега-План Архитектуры (Институциональный Уровень)**

#### 1️⃣ **УРОВЕНЬ 1: DATA LAYER (Слой данных и микроструктуры)**
*Задача: Перейти от анализа свечей к анализу потока событий и обеспечить чистоту данных.*

1.  **Сбор данных о микроструктуре:**
    *   **Что:** L2/L3 снимки стакана, лента сделок (trades), данные о ликвидациях и ставки финансирования.
    *   **Как:** Настроить пайплайн для сбора и хранения тиковых данных (например, через `ccxt.watch_order_book` и `watch_trades`) в QuestDB.
2.  **Инжиниринг признаков из стакана (L2/L3 Features):**
    *   **Суть:** Обогатить поток событий агрегированными признаками состояния рынка.
    *   **Реализация:**
        ```python
        # L2 Imbalance
        I = (V_bid - V_ask) / (V_bid + V_ask)
        # L2 Microprice
        P_micro = (P_ask * V_bid + P_bid * V_ask) / (V_bid + V_ask)
        # L3/Flow Order Flow Imbalance (OFI)
        OFI = delta_V_bid - delta_V_ask
        ```
3.  **Kalman Filter (Очистка цены от шума):**
    *   **Суть:** Перед расчетом признаков применить Фильтр Калмана к ряду `mid-price` для удаления шума.
    *   **Реализация:**
        ```python
        from pykalman import KalmanFilter
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                          initial_state_mean=0, initial_state_covariance=1,
                          observation_covariance=1, transition_covariance=0.01)
        state_means, _ = kf.filter(df["close"].values)
        df["price_kalman"] = state_means
        # Все последующие расчеты (returns, индикаторы) идут от price_kalman
        ```
4.  **Strict Causal Feature Audit (Анти-Утечка):**
    *   **Суть:** Провести аудит всех признаков на предмет заглядывания в будущее.
    *   **Чек-лист:**
        *   VWAP должен быть `rolling` или `intraday reset`, **не** `cumulative`.
        *   OFI должен считаться строго до текущего тика.
        *   `t1` из Triple Barrier не должен пересекаться с тестовой выборкой.

#### 2️⃣ **УРОВЕНЬ 2: ALPHA ENGINE (Движок генерации альфы)**
*Задача: Предсказать будущее состояние рынка, а не просто направление.*

5.  **Event-Time Labeling Engine (Продвинутый Таргет):**
    *   **Суть:** Заменить свечной `Triple Barrier` на систему разметки, работающую в "event time".
    *   **Реализация:**
        ```python
        # 1. Барьеры на основе волатильности, нормализованной по режиму
        df["vol_norm"] = df["atr"] / df["atr"].rolling(100).mean()
        dynamic_tp = k_tp * atr * vol_norm
        dynamic_sl = k_sl * atr * vol_norm
        
        # 2. Удаление тайм-аутов для бинарной классификации
        mask = labels != 0
        X_filtered = X[mask]
        y_filtered = (labels[mask] == 1).astype(int) # TP hit vs SL hit
        ```
6.  **Order Flow Transformer (OFT):**
    *   *Суть:* Гибридная Transformer-модель, анализирующая последовательность событий.
    *   *Архитектура:* `Event Embedding → Order Book Encoder → Temporal Transformer → Cross-Attention`.
    *   *Выход (многозадачный):* Распределение доходности (μ, σ²), вероятность движения, риск ликвидности.
7.  **Методология обучения OFT (Anti-Overfitting):**
    *   **Purged Walk-Forward CV:** Использование `mlfinlab.PurgedKFold` для устранения утечек данных.
    *   **Regime Conditioning:** Модель обучается с учетом текущего режима рынка (`model(x | regime)`).
    *   **Output Calibration:** Калибровка выходных вероятностей (`Isotonic Regression` или `Temperature Scaling`).
    *   **Microstructure Augmentation:** Внесение шума в данные для повышения робастности.
8.  **Bayesian Regime Model (Адаптивный детектор режимов):**
    *   **Суть:** Заменить GMM/HMM на `BayesianGaussianMixture` для онлайн-адаптации.
    *   **Реализация:**
        ```python
        from sklearn.mixture import BayesianGaussianMixture
        model = BayesianGaussianMixture(n_components=5, weight_concentration_prior=0.01)
        model.fit(X)
        # Online update
        model.partial_fit(new_data)
        ```

#### 3️⃣ **УРОВЕНЬ 3: EXECUTION & SIMULATION ENGINE (Движок исполнения и симуляции)**
*Задача: Исполнить ордер, минимизируя влияние на рынок, и обучать модели в реалистичной среде.*

9.  **Synthetic Adversarial Simulator:**
    *   **Суть:** Симулятор биржи с агентной моделью рынка, который генерирует данные для обучения и реагирует на действия бота.
    *   **Компоненты:**
        *   **Differentiable Matching Engine:** Исполнение ордеров через "мягкие" вероятностные функции (`softmax matching`).
        *   **Multi-Agent Self-Play:** OFT-альфа, RL-маркетмейкеры и другие агенты обучаются, конкурируя друг с другом.
10. **Joint Training (OFT + RL Execution):**
    *   **Суть:** Объединить Alpha и Execution в один обучающий контур внутри симулятора.
    *   **Цель:** `min(-E[PnL] + λ1*CVaR + λ2*ImpactCost + λ3*InventoryRisk)`.
11. **HFT-style Inventory Hedging:**
    *   **Суть:** RL-агент по исполнению активно управляет инвентарным риском.
    *   **Reward:** `R = PnL - λ * Inventory²`.
12. **Alpha Decay Model (Распад сигнала):**
    *   **Суть:** Заменить жесткий `max_hold_bars` на модель экспоненциального затухания сигнала.
    *   **Реализация:**
        ```python
        def apply_alpha_decay(signal_strength, time_in_trade, decay_rate=0.1):
            return signal_strength * np.exp(-decay_rate * time_in_trade)
        
        # В цикле: если apply_alpha_decay(...) < threshold, закрыть позицию.
        ```

#### 4️⃣ **УРОВЕНЬ 4: PORTFOLIO OPTIMIZATION (Движок портфеля)**
*Задача: Управлять риском на уровне всего портфеля, а не отдельных сделок.*

13. **CVaR Optimizer (Оптимизатор по CVaR):**
    *   **Суть:** Заменить простые правила сайзинга на полноценный портфельный оптимизатор.
    *   **Цель:** `max E[R] - λ * CVaR_α(R)`.
14. **Risk Parity & Confidence Sizing (как компонент):**
    *   **Суть:** Использовать принципы Risk Parity и уверенности модели как входные данные для CVaR-оптимизатора.
    *   **Реализация:**
        ```python
        # 1. Сила сигнала
        weights = (probabilities - 0.5) * 2
        # 2. Паритет риска (простой)
        weights = weights / asset_volatility
        # 3. Штраф за корреляцию
        penalty = 1 - returns.corr().mean().mean()
        weights *= penalty
        # 4. Нормализация
        weights /= np.sum(np.abs(weights))
        ```
15. **Dynamic Threshold Optimization:**
    *   **Суть:** Оптимизировать порог уверенности для входа в сделку, максимизируя `Sharpe` или `PnL` на валидационном сете.
    *   **Реализация:**
        ```python
        def find_best_threshold(y_true, probs, returns):
            best_thr, best_sharpe = 0.5, -np.inf
            for thr in np.linspace(0.5, 0.8, 30):
                preds = (probs > thr).astype(int)
                pnl = pd.Series(preds * returns)
                sharpe = pnl.mean() / (pnl.std() + 1e-9)
                if sharpe > best_sharpe:
                    best_sharpe, best_thr = sharpe, thr
            return best_thr
        ```