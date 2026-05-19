# VPS Чистый старт + Редизайн пайплайна данных — ФИНАЛЬНЫЙ ПЛАН v11

**Создан:** 2026-05-19  
**Статус:** ФИНАЛЬНЫЙ — подтверждён оператором, все ревью агентов учтены  
**VPS:** 5.104.81.27 (Токио, Contabo, 400 ГБ SSD, 24 ГБ RAM, Ubuntu 24.04)  
**Ветка:** `dev/vps-clean-slate` — вся работа здесь, мерж в main только после полного тестирования

---

## Фаза 0 — Хаускипинг + Git ветка

```
git checkout -b dev/vps-clean-slate
git push -u origin dev/vps-clean-slate
```

- Запустить `tests/test_dashboard.py` — зафиксировать baseline (0 failures обязательно)
- Удалить мёртвый код / debug print'ы через агента `refactor-cleaner`
- Убрать неиспользуемые импорты; обновить `CLAUDE.md`
- Проверить `.gitignore`: `data/parquet/`, `data/raw/`, `models/`, `logs/`, `.env`
- PR → main только после: все фазы + smoke-test + 0 failures + апрув code-reviewer

**Хардениг VPS (выполнить один раз на свежем VPS, до Фазы 1):**
```bash
ufw default deny incoming
ufw allow 22/tcp
ufw allow from <operator_IP> to any port 5000   # Flask дашборд — только твой IP
ufw enable
# /etc/ssh/sshd_config:
PasswordAuthentication no
PermitRootLogin prohibit-password               # вход по ключу; правильно для solo-оператора
# автоматические патчи безопасности:
apt install unattended-upgrades
dpkg-reconfigure --priority=low unattended-upgrades
# защита от брутфорса:
apt install fail2ban
```
- **Отдельные env-профили** — не переиспользовать один `config.yaml` для разных режимов. Создать отдельные профили:
  - `config/training.yaml` — высокие лимиты памяти, нет вывода ордеров, подробное логирование
  - `config/backtest.yaml` — историческая модель slippage, нет вызовов live API
  - `config/paper.yaml` — live данные, paper-sink для ордеров, полное логирование сигналов
  - `config/live.yaml` — live данные, реальное размещение ордеров, консервативные лимиты
  - Профиль выбирается через `APP_ENV=training|backtest|paper|live` при старте

---

## Фаза 1 — Исправление багов (VPS)

**Фикс A — ZMQ_BUS_KEY**
```
python -c "import secrets; print(secrets.token_hex(32))"
```
Добавить как `ZMQ_BUS_KEY=<значение>` в `/root/trading-bot/.env`. Найти grep'ом всех подписчиков шины до рестарта.

**Фикс A2 — Проверить наличие всех секретов в .env**

VPS — чистый лист. До любой фазы, обращающейся к внешнему API, убедиться что все ключи перенесены:

| Ключ | Нужен в |
|------|---------|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Фаза 1В, Фаза 8, боевая торговля |
| `BINGX_API_KEY` / `BINGX_API_SECRET` | Боевая торговля (если активен) |
| `HETZNER_API_TOKEN` | **Фаза 8 CPU тренировка** — оркестратор упадёт без него |
| `VASTAI_API_KEY` | **Фаза 8 GPU тренировка** |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Алерты зомби-серверов (Фаза 8) |

Команда проверки на VPS: `grep -E "BINANCE|BINGX|HETZNER|VASTAI|TELEGRAM" /root/trading-bot/.env | cut -d= -f1`

**Права API-ключа Binance (ОБЯЗАТЕЛЬНО — создать ключ с минимальными правами):**

Ключ должен быть создан только с необходимыми правами. Никогда не создавать неограниченный ключ для VPS:
- ✅ Включить: `Read` (информация об аккаунте, рыночные данные)
- ✅ Включить: `Spot & Margin Trading` (spot-ордера)
- ✅ Включить: `Futures` (USDT-M фьючерсы — только если стратегия фьючерсов активна)
- ❌ Отключить: `Withdrawals` — при компрометации ключа атакующий не сможет вывести средства
- ❌ Отключить: `Universal Transfer`
- Ограничить по IP: занести в whitelist только IP VPS `5.104.81.27`

**DASHBOARD_API_KEY — жёсткий аборт при отсутствии ключа:**

Дашборд должен отказать в запуске (не тихо пропустить аутентификацию) если `DASHBOARD_API_KEY` отсутствует или пустой в `.env`. Тихий fallback к неаутентифицированному режиму оставляет Flask-дашборд открытым на порту 5000 для интернета. Проверить при старте:
```python
DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY", "")
if not DASHBOARD_API_KEY:
    raise SystemExit("FATAL: DASHBOARD_API_KEY не задан в .env — отказ запускаться без аутентификации")
```

**Фикс Б — Устаревшие метки агентов**
Записать свежий `data/agent_status.json`: все `status: inactive`, `last_heartbeat: null`.

**Фикс В — WebSocket таймауты**
`src/main.py:1437` `ping_timeout=20` → `ping_timeout=60`  
`src/main.py:1438` `close_timeout=10` → `close_timeout=15`  
`systemctl restart trading-bot`

---

## Фаза 2 — Перенос 49 ГБ data/parquet на VPS

Остановить `trading-realtime`, `trading-bot`, затем:

> **ВАЖНО — отключить спящий режим Windows перед стартом rsync.** Если ноутбук уснёт на 15-й минуте, SSH-туннель rsync оборвётся и на VPS останутся частично записанные Parquet-файлы, которые тихо повреждены. Выполнить до rsync:
> ```powershell
> powercfg /change standby-timeout-ac 0   # отключить сон на AC питании
> ```
> Восстановить после передачи: `powercfg /change standby-timeout-ac 30`

```
rsync -av --progress -e "ssh -i ~/.ssh/trading_bot" \
  "D:/test 2/AI trading assistance/data/parquet/" \
  root@5.104.81.27:/root/trading-bot/data/parquet/
```
~28 мин при 30 МБ/с. Сверить количество файлов после передачи.

> **Примечание:** `-z` (сжатие) намеренно убрано. Parquet-файлы уже сжаты Snappy — добавление `-z` нагружает CPU без экономии трафика.

**Сразу после передачи — единоразовый бэкап parquet в Google Drive:**
```
rclone copy /root/trading-bot/data/parquet/ gdrive:trading-bot-backup/parquet-archive/
```
Выполняется один раз вручную. Ежедневный cron (Фаза 5) навсегда исключает `data/parquet/**`. Без этого шага VPS — единственная точка отказа для 49 ГБ.

---

## Фаза 3 — Миграция CSV.gz → Parquet, архивирование

**Валидация схемы ДО миграции:**
```python
import pyarrow.parquet as pq
existing = pq.read_schema("data/parquet/BTCUSDT/1h/yyyymm=2025-01/data_0.parquet")
migrated = pq.read_schema("<выходной_путь>")
assert existing == migrated
```
Принудительно использовать `datetime64[us]` — это канонический формат PyArrow 13+. **Не** форсировать `ns`: PyArrow 13+ пишет `us` по умолчанию, а форсирование `ns` вызывает ложные ошибки сравнения схем при каждом новом файле. Timestamp Binance с точностью до миллисекунды — `us` сохраняет полную точность без потерь.

**Миграция существующих 49 ГБ (смесь ns/us) — Вариант А (самый безопасный):**
```
1. Писать все батчи в data/parquet_us/  (оригинальный data/parquet/ не трогать)
2. Полная проверка: количество файлов + assert схемы us + проверка fingerprint
3. СТОП БОТ
4. переименовать data/parquet/    → data/parquet_backup/
5. переименовать data/parquet_us/ → data/parquet/
6. СТАРТ БОТ
7. Удалить data/parquet_backup/ после первого успешного тренировочного запуска
```
Обрабатывать батчами ~5 ГБ для возможности пакетного отката. `rename()` атомарен на Linux (один syscall) — безопасно при остановленном боте.

- Запустить `python scripts/migrate_csv_to_parquet.py` после прохождения проверки
- Переместить все `data/raw/*.csv.gz` → `data/raw_archive/`
- **`touch` каждого перемещённого файла для сброса mtime к времени прибытия** (rsync сохраняет исходный mtime; без touch cron удалит файлы сразу если исходный mtime > 7 дней):
  ```
  find /root/trading-bot/data/raw_archive/ -name "*.csv.gz" -exec touch {} +
  ```
- Cron (явный UTC):
  ```
  TZ=UTC
  0 4 * * * find /root/trading-bot/data/raw_archive/ -name "*.csv.gz" -mtime +7 -delete
  ```

**То же правило `touch` для всех будущих доставок CSV.gz через rsync** — добавить в скрипт доставки.

---

## Фаза 4 — Изменение кода: CSV.gz только как временный файл

Новый поток: скачать CSV.gz → конвертировать → `mv` в `data/raw_archive/` → автоудаление через 7 дней.

`src/data_ingestion/ohlcv_parquet_loader.py`:
- Заменить тихий `return pd.DataFrame()` на:
  ```python
  raise FileNotFoundError(f"No parquet data for {symbol}/{timeframe}. Run backfill first.")
  ```
- Удалить CSV.gz fallback из `load_funding` (~строки 83–89) отдельно
- Обновить все вызывающие места — явная обработка `FileNotFoundError`
- Оркестратор вызывает `pyarrow.parquet.read_schema()` при загрузке и сравнивает со стандартом схемы
- Скрипт миграции CSV должен по умолчанию использовать `--skip` (не перезаписывать), если Parquet-файл для раздела уже существует — тихая перезапись валидных данных деструктивна

**Провал валидации — карантин, не удалять:**

Если новый parquet не проходит проверку схемы, fingerprint или sanity-check дрифта — перемещать в `data/quarantine/` с меткой времени:
```
data/quarantine/
  20260519_143022_BTCUSDT_1h_schema_fail.parquet
  20260519_150401_ETHUSDT_4h_fingerprint_mismatch.parquet
```
Никогда не удалять автоматически — сначала ручной аудит. Данные могут быть восстановимы, или ошибка может указывать на баг в валидаторе. Добавить в исключения rclone (Фаза 5): `--exclude "data/quarantine/**"`

---

## Фаза 5 — rclone + Google Drive (ежедневная золотая копия)

`/root/trading-bot/` → `gdrive:trading-bot-backup/`, раз в день.

**Обязательные исключения:**
```
--exclude "data/parquet/**"      ← 49 ГБ — единоразово в Фазе 2, не ежедневно
--exclude "data/raw_archive/**"
--exclude "logs/**"
--exclude "*.pyc"
--exclude "__pycache__/**"
--exclude "*.lock"
--exclude "**/*.tmp"
--exclude "data/cache/**"
--exclude "data/quarantine/**"
```

1. `rclone config` на VPS — remote "gdrive" (URL для браузера)
2. Cron:
   ```
   TZ=UTC
   0 3 * * * rclone sync /root/trading-bot/ gdrive:trading-bot-backup/ \
     --exclude "data/parquet/**" --exclude "data/raw_archive/**" \
     --exclude "logs/**" --exclude "*.pyc" --exclude "__pycache__/**" \
     --exclude "*.lock" --exclude "**/*.tmp" --exclude "data/cache/**" \
     --exclude "data/quarantine/**" \
     --log-file=/root/trading-bot/logs/rclone.log
   ```
3. Убрать Windows Task Scheduler sync с локальной машины

---

## Фаза 6 — Smoke-test на синтетических данных

Выполнять ДО архивирования реального состояния.

**Требования к синтетическим данным:**
- CPU (`data/parquet_test/base_test/`, ~100 МБ): AR(1)/GBM цены, объём коррелирован с волатильностью, ≥50k строк, точная схема `OHLCV_COLS`, `datetime64[us]`, разделы `yyyymm=YYYY-MM`
- GPU (`data/parquet_test/tft_test/`, ~200 МБ): то же + длина последовательности ≥ lookback window. **Минимум 200k строк** — TFT с lookback=168 требует ~200k/168 ≈ 1190 уникальных последовательностей для стабильного обучения; 50k даёт только ~297 — недостаточно
- Ассерция `df.dtypes` перед сохранением

Прогнать dispatch из `training_rules.json`: ≥2 символа × ≥2 таймфрейма.

**Тест CPU (Hetzner CCX33):** создать → SSH → тренировка → проверить артефакты на VPS через rsync pull → сервер **УДАЛЁН** через Hetzner API.

**Тест GPU (Vast.ai RTX 4090):** арендовать → SSH → тренировка → проверить артефакты → инстанс **УНИЧТОЖЕН** через Vast.ai API.

Только после обоих — Фаза 7.

---

## Фаза 7 — Архивирование состояния тренировок

**Остановить бота.** Убедиться что нет `running` в `dashboard_jobs.json` перед копированием `models/`.

Архив в `data/training_archive/YYYY-MM-DD/`:

| Источник | Назначение |
|----------|------------|
| `models/` | `training_archive/YYYY-MM-DD/models/` |
| `data/training_runs_history.json` | `training_archive/YYYY-MM-DD/` |
| `data/training_status_report.json` | `training_archive/YYYY-MM-DD/` |
| `data/agent_status.json` | `training_archive/YYYY-MM-DD/` |
| `data/dashboard_jobs.json` | `training_archive/YYYY-MM-DD/` |
| `data/bake_off_cut_list.json` | `training_archive/YYYY-MM-DD/` |

После копирования — очистить оригиналы. Архив никогда не удаляется.  
Никогда не трогать: `data/training_rules.json`, `data/parquet/`, `.env`.

---

## Фаза 8 — Переобучение всех моделей с нуля

### Философия ретрейна — Чистый исследовательский сброс

Это НЕ "переобучить всё так же, как раньше". Исторический торговый лог (2026-04-25 → 2026-05-17) показал: скалпинг уничтожил 98% убытков (−$1090 из −$1112), а ни одна ML-стратегия не сгенерировала ни одной live-сделки. Ретрейн — это чистый исследовательский сброс с новой стратегией деплоя.

**Правило 1 — Scalping остаётся в paper/experimental.**
Scalping_ML обучается (данные уже доказывают что модель улавливает краткосрочные движения), но НЕ должна получать реальный капитал до тех пор, пока не докажет положительный EV после fees/slippage в отдельном paper-canary из ≥500 сделок и ≥30 дней. По умолчанию после ретрейна: `Scalping_ML live: false`.

**Правило 2 — Оптимизировать expectancy после комиссий, а не accuracy.**
Тренировочный loss и WF accuracy — вторичные метрики. Основная цель оптимизации для каждой модели:
```
profit_factor = gross_profit / gross_loss  (цель > 1.5)
after_fee_sharpe                           (цель > 0.5 live)
max_drawdown                               (жёсткий лимит)
```
Модель с WF Acc 52% и Sharpe 0.8 лучше, чем WF Acc 55% и Sharpe 0.1.

**Правило 3 — Валидация раздельно по режимам, не только суммарно.**
Для каждой модели запускать walk-forward отдельно для `bull / bear / chop / high_vol`. Модель, работающая только в bull — это bull-only стратегия. Деплоить её только при regime = bull, не как постоянный сигнал.

**Правило 4 — Фокус ретрейна: сначала 3 валидированные комбо.**
Не активировать все стратегии одновременно. Первый live-деплой использует только:
```
Комбо A: Trend RF (1h/4h) + Meta-Labeler фильтр + Regime Router (только TRENDING) + GARCH sizing
Комбо B: Base RF (1h)     + Meta-Labeler фильтр + GARCH sizing
Комбо C: Volatility Breakout (rule) + Regime Router + Meta фильтр
```
Всё остальное (futures short, TFT, OFT, scalping, rule-зоопарк) → только paper/canary, пока interaction matrix (Фаза 9) не покажет Sharpe > 0.

**Правило 5 — Отключить rule-зоопарк при первом live-запуске.**
После ретрейна отключить в `strategy_config.json`: `ElliottWave_ML`, `Ichimoku_Cloud`, `MACD_Divergence`, `Keltner_Breakout`, `Supertrend`, `VWAP_Reversion`, `Donchian_Breakout`, `OU_Entry`, все Ensemble-варианты. Оставить как baselines, но `live: false`. Меньше активных стратегий = чище данные атрибуции.

**Правило 6 — Телеметрия Meta-Labeler обязательна с первого дня.**
В предыдущем запуске `model_confidence = NULL` во всех 1350 сделках — Meta-Labeler якобы был live, но не оставил никакого следа ни в одной записи трейда. После ретрейна `execution_audit.jsonl` обязан логировать на каждом ордере:
```json
{
  "model_confidence": 0.73,
  "meta_passed":      true,
  "predicted_prob":   0.73,
  "threshold":        0.54,
  "strategy":         "Trend_ML",
  "model":            "trend_model_4h",
  "timeframe":        "4h",
  "regime":           "trending",
  "expected_ev":      0.0042
}
```
Без этого interaction matrix невозможна, а у Master Allocator нет сигнала.

**Правило 7 — Тестировать комбо последовательно, а не все сразу.**
Interaction matrix строится путём поэтапного добавления одного компонента и измерения Sharpe на каждом шаге. Нельзя активировать все фильтры сразу на реальном капитале — теряется возможность атрибуции P&L отдельным компонентам. Последовательность:
```
Этап 1: Trend RF (1h/4h) в одиночку              — первый live
Этап 2: Trend + Meta-Labeler                     — после Sharpe Этапа 1 > 0 (≥50 сделок)
Этап 3: Trend + Regime Router                    — после Sharpe Этапа 2 > 0 (≥50 сделок)
Этап 4: Trend + Meta + Regime                    — после Sharpe Этапа 3 > 0 (≥50 сделок)
Этап 5: Trend + Meta + Regime + GARCH sizing     — после Sharpe Этапа 4 > 0 (≥50 сделок)
```
Каждый переход этапа требует нового canary-периода (≥50 сделок, см. пороги canary в Фазе 9). Поля `meta_passed`, `regime_used`, `garch_used` в `execution_audit.jsonl` обеспечивают атрибуцию каждого этапа в карточке Interaction Matrix.

**Правило 8 — Первый live-сет (жёсткое ограничение).**
После завершения ретрейна реальный капитал могут получать только:
```
✅  Trend Momentum RF — таймфреймы 1h и 4h
✅  Meta-Labeler фильтр (только после проверки телеметрии в paper — meta_passed не NULL)
✅  Regime Router
✅  GARCH позиционирование
❌  Scalping_ML — только paper/canary (Правило 1)
❌  Futures Short — только paper/canary до доказательства положительного live Sharpe у Trend combo
❌  TFT / OFT — только paper/canary
❌  ElliottWave, rule zoo — отключить (Правило 5)
```
Если `meta_passed` остаётся NULL в первых 5 paper-сделках — остановиться и исправить инструментирование до любой live-сделки.

**Приоритет деплоя после ретрейна:**
```
1. Инструментирование телеметрии (проверить ДО любой live-сделки)
2. Meta-Labeler (лучший AUC: 0.641 — самый важный фильтр)
3. Regime-aware Trend RF (структурно совместим с crypto)
4. Volatility Breakout (rule-based, быстро валидировать)
5. Portfolio Allocator (после того как interaction matrix докажет ≥3 комбо)
6. Futures Short, TFT, OFT, Scalping — только после доказательства в canary
```

### Предпроверка места на диске (жёсткий гейт)

Перед стартом любого тренировочного запуска оркестратор обязан проверить свободное место:
```python
import shutil
free_gb = shutil.disk_usage('/root/trading-bot').free / (1024 ** 3)
if free_gb < 20:
    raise SystemError(f"Мало места: {free_gb:.1f} ГБ свободно (нужно ≥20 ГБ). Тренировка прервана.")
```

DuckDB при нехватке RAM пишет временные файлы в `data/cache/duckdb_temp`. Если диск заполнится в разгар тренировки, файл `.db` может повредиться — незавершённая транзакция, на следующий запрос вернётся мусор. Источники роста: `training_archive/`, `logs/`, `data/oos_signals/`, `data/cache/duckdb_temp/`. При 400 ГБ SSD и 49 ГБ parquet порог 20 ГБ безопасен и конкретен.

**Коннект к DuckDB обязан выставлять лимит RAM и temp-директорию (ОБЯЗАТЕЛЬНО):**
```python
con = duckdb.connect(database_path)
con.execute("SET memory_limit='18GB'")        # НЕ PRAGMA — DuckDB использует SET, а не PRAGMA
con.execute("SET temp_directory='/root/trading-bot/data/cache/duckdb_temp/'")
```
`PRAGMA` — синтаксис SQLite. DuckDB тихо игнорирует неизвестные PRAGMA, то есть лимит никогда не применялся в предыдущих версиях. Использовать `SET`.

**Паттерн singleton-соединения (ОБЯЗАТЕЛЬНО):** Не открывать несколько соединений DuckDB к одному файлу. При 3 одновременных потребителях, каждый требующий 18 ГБ на VPS с 24 ГБ, агрегированная потребность = 54 ГБ → OOM. Одно процессно-глобальное соединение в `src/data/duckdb_pool.py` (ленивый singleton), разделяемое всеми читателями. Тренировщики на Hetzner/Vast используют свои соединения на других хостах — это нормально.

### Пред-полётный чеклист (выполнить перед реальным ретрейном)

```bash
python scripts/preflight_train.py
```

Скрипт проверяет и выводит PASS/FAIL по каждому пункту:

| Проверка | Что проверяет |
|---------|---------------|
| Disk free ≥ 20 ГБ | Аналогично пред-проверке выше |
| Количество parquet файлов | Совпадает с ожидаемым после синхронизации |
| Схема валидна | `pq.read_schema()` на выборке файлов |
| Нет запущенных джобов | В `dashboard_jobs.json` нет `status: running` |
| API ключи присутствуют | Все 5 ключей в `.env` |
| Бэкап GDrive существует | `rclone lsd gdrive:trading-bot-backup/` возвращает записи |
| `training_rules.json` валиден | JSON парсится, все обязательные поля есть |
| OOS-директория доступна для записи | `touch data/oos_signals/.write_test` успешен |
| Hetzner credentials | `GET /v1/servers` возвращает 200 |
| Vast.ai credentials | `GET /api/v0/instances` возвращает 200 |

Любой FAIL → прерывание с ненулевым кодом выхода. Оркестратор обязан проверить exit code перед стартом.

### Порядок обучения

```
regime → base → trend → futures → scalping → meta → tft → oft
```

Regime (GMM, unsupervised) первым — нет зависимостей. Признаки regime **опциональны/инжектируемы** в других моделях — не жёсткая зависимость.

### OOS-сигналы для meta (обязательно)

После завершения каждой из base / trend / futures — сохранить OOS-предсказания **с run_id**:
```
data/oos_signals/<run_id>/
  base.parquet
  trend.parquet
  futures.parquet
```
`run_id` = UTC-метка времени старта тренировочного запуска (например `2026-05-20T14:00:00`). Каждый OOS-файл содержит `run_id` в пути И как колонку.

Перед стартом meta: **жёсткая проверка** — все три файла должны существовать под **одним `run_id`**. Если хоть один отсутствует ИЛИ значения `run_id` не совпадают (устаревший файл от предыдущего частичного запуска) → остановить, не продолжать.

**Зачем run_id важен:** если `base` завершился, `trend` упал и ретрейн перезапустился с `trend` — meta тихо потребит свежие OOS base + устаревшие OOS trend от предыдущего запуска. Изоляция по run_id это предотвращает.

### Протокол чекпоинтов (Pull-модель rsync)

После завершения каждой модели на Hetzner/Vast.ai:
1. Артефакты сохранены в `models/` на тренировочном сервере
2. **VPS сам тянет с тренировочного сервера** (не push — никаких приватных ключей на временном Hetzner):
   ```
   rsync -avz -e "ssh -i ~/.ssh/trading_bot" \
     root@<hetzner_ip>:/root/models/ /root/trading-bot/models/
   ```
3. Запись в `training_runs_history.json` на VPS
4. Проверить файл на VPS до старта следующей модели

### Env Manifest

Захватить при старте тренировки, сохранить как `env_manifest.json`:
```python
import importlib.metadata, torch, platform
manifest = {
    "python": platform.python_version(),
    "lightgbm": importlib.metadata.version("lightgbm"),
    "scikit-learn": importlib.metadata.version("scikit-learn"),  # НЕ "sklearn"
    "torch": torch.__version__,
    "cuda": torch.version.cuda,   # НЕ nvcc — может не быть в PATH на Vast.ai
    "pyarrow": importlib.metadata.version("pyarrow"),
    "numpy": importlib.metadata.version("numpy"),
}
```
Добавить `capture_env_manifest()` в `src/utils/env_manifest.py`, вызов из entry point оркестратора.

### Инфраструктура

**CPU** (regime, base, trend, futures, scalping, meta) — Hetzner CCX33: ~8.5ч, ~€0.85. Сервер **УДАЛЁН** в конце И в exception handler.

**GPU** (tft, oft) — Vast.ai RTX 4090: ~4-6ч, ~$1.50-2.20. Инстанс **УНИЧТОЖЕН** в конце И в exception handler.

**Защита от зомби-серверов (ОБЯЗАТЕЛЬНО):**

Удаление/уничтожение должно использовать цикл с экспоненциальным backoff. Если API Hetzner или Vast.ai вернёт 500 / таймаут, сервер продолжит работать и сжигать баланс молча.

```python
import time
for attempt in range(3):
    try:
        api.delete_server(server_id)   # или api.destroy_instance(instance_id)
        break
    except APIError:
        time.sleep(30 * (2 ** attempt))  # 30с → 60с → 120с
else:
    send_telegram_alert(
        f"КРИТИЧНО: Не удалось уничтожить {server_id} после 3 попыток. "
        f"Требуется ручное действие — проверь биллинг-дашборд немедленно."
    )
```

Применять эту обёртку И в нормальном teardown, И в exception handler.

---

## Фаза 9 — Система Champion/Challenger с бейзлайнами

### Структура хранения

```
data/baselines/
  v1_2026-05-19/
    metadata.json                (дата, git hash, dataset_hash, заметки)
    metrics.json                 (все метрики: model × TF × symbol)
    data_manifest.json           (по символу: train_start, train_end, n_bars)
    feature_schema.json          (колонки, dtypes, frac-diff d, feature_pipeline_version)
    env_manifest.json            (версии python/lightgbm/cuda/torch)
    data_fingerprint_cache.json  (кэш mtime+size для fast-path)
    model_snapshots/
  current.json                   → { "active_baseline": "v1_2026-05-19" }
```

### Dataset Fingerprint

Хешировать **логические данные**, не raw bytes (binary sha256 меняется при ре-сжатии с тем же контентом):
```python
fingerprint = {
    "schema_hash":        sha256(column_names + dtypes),
    "row_count":          total_rows,
    "timestamp_min":      earliest_bar.isoformat(),
    "timestamp_max":      latest_bar.isoformat(),
    "per_column":         {col: sha256(col_values) for col in df.columns},
    "file_count":         n_parquet_files,
    "symbol_tf_coverage": {sym: [tfs] for sym in symbols},
}
```

**Потоковый хеш (ОБЯЗАТЕЛЬНО — 48 ГБ не помещается в памяти):**
```python
import pyarrow.parquet as pq, hashlib
h = hashlib.sha256()
pf = pq.ParquetFile(path)
for batch in pf.iter_batches(batch_size=50_000):
    for col_name in batch.schema.names:
        buf = batch.column(col_name).buffers()[1]  # zero-copy Arrow buffer
        if buf:
            h.update(buf)
per_column_hash = h.hexdigest()
```
Никогда не загружать полный DataFrame — использовать `iter_batches()` с инкрементальным SHA256.

Fast-path: сравнить `{path: (mtime, size)}` сначала. Полный логический хеш только при изменении mtime или размера.

**Запись кэша должна быть атомарной:**
```python
import os, json
tmp = path + ".tmp"
with open(tmp, 'w') as f:
    json.dump(cache, f)
os.replace(tmp, path)   # атомарно на Linux — предотвращает повреждение при крэше
```
Кэш в `data_fingerprint_cache.json`.

Агент сравнения отклоняет сравнения с разными `dataset_hash`.

### Feature Pipeline Version

`feature_schema.json` обязан содержать `feature_pipeline_version` (например `"v2.1"`). Инкрементировать при любом изменении алгоритма.

### Метрики (по каждой model × TF × symbol)

**Уровень 1 — Финансовые (блокирующие):** `Sharpe`, `EV`, `Calmar`, `max_drawdown`, per-symbol Sharpe floor (ни один символ > -10%)

**Уровень 2 — ML (только информационно):** `PR-AUC`, `Precision(TP)`, `Recall(TP)`, `OOS log-loss` (meta: только log-loss)

**ML-целостность:** `avg_uniqueness` (детектирует overlap меток)

### Логика ребейзлайна (статистическая значимость)

```
УРОВЕНЬ 1 — не должны регрессировать:
  ✅ Sharpe И EV оба не хуже (оба падают → REJECT, без исключений)
  ✅ Calmar не хуже; max_drawdown не хуже
  ✅ Ни один символ не теряет > -10% Sharpe

УРОВЕНЬ 2 — статистически значимое улучшение:
  ✅ Хотя бы одна метрика Уровня 1 улучшается значимо
     Метод: Stationary Block Bootstrap (arch.bootstrap.StationaryBootstrap)
             Politis-Romano автоматическая длина блока
             N=1000 ресэмплов; нижняя граница CI > 0
  ✅ Улучшение только по финансовой метрике (Sharpe, EV или Calmar)
     win_rate в одиночку НЕ считается

УРОВЕНЬ 3 — только отображение, не блокирует:
  PR-AUC, Precision, Recall — в таблице, не влияют на ворота
```

### Стресс-тест торговых издержек (обязателен перед canary)

Бэктест с: `fees × 1.5`, `slippage × 2`, `latency spikes`. P&L уходит в минус → fragile → не продвигаем.

### Canary Deployment (пороги по типу модели)

5% аллокации капитала рядом с текущим champion. Ворота:

| Тип модели | Мин. дней | Мин. сделок | Примечание |
|------------|-----------|-------------|------------|
| Scalping | 14 | 500 | Любое из условий: `min(14 дней, 500 сделок)`. 500 даёт ~2.2σ; но 14 дней данных достаточно, если стратегия торгует медленнее ожидаемого |
| Base / Meta | 14 | 100 | Среднечастотная |
| Trend | 30 | 50 | **OR-логика:** продвигать когда `(50 сделок ИЛИ 30 дней) AND actual_trades >= 10`. ~1.4 сделки/день при $500; 50 за 14 дней невозможно |
| Futures | 30 | 50 | **OR-логика:** то же что Trend. `(50 сделок ИЛИ 30 дней) AND actual_trades >= 10` |
| Regime (GMM) | 21 | N/A | Ворота: ≥3 полных смены режима |
| TFT / OFT | 14 | 30 | Среднe-низкочастотные |

Критерии продвижения (все должны пройти):
- `|live_sharpe - backtest_sharpe| / backtest_sharpe < 0.30`
- Ни одного дневного drawdown сверх стресс-теста
- **Challenger live Sharpe ≥ Champion live Sharpe** (не только vs бэктест)

**Защита MIN_NOTIONAL:** если 5% капитала < минимального ордера биржи ($5–$10 в зависимости от пары), канарейка торгует минимальным разрешённым лотом вместо 5%. При балансе $500 это ещё не критично ($25 > $10), но если баланс упадёт до $100, `5% = $5` и часть пар отклонит ордер. Проверять `MIN_NOTIONAL` из фильтра биржи в момент ордера; фолбэк на минимальный лот без шума (канарейку не пропускать).

### Correlation-Aware Portfolio Gate

Перед исполнением сигналов:
- Попарная корреляция Pearson > 0.7 (30-дневный rolling) = коррелированный кластер
- `exposure_cap`: максимум 20% капитала в одном кластере
- `correlation_penalty`: уменьшить размер позиции при корреляции > порога
- Roadmap: перейти на rolling correlation + beta-to-BTC взвешивание

**Компонент:** `src/risk/correlation_gate.py` (новый файл)

### Rollback Playbook

Если challenger ухудшает live-производительность — откат к последнему известному хорошему бейзлайну за 2 минуты:
```bash
python -m src.governance.baseline_manager rollback --to v1_2026-05-19
systemctl restart trading-bot
```
`rollback`: восстанавливает указатель `current.json`, меняет симлинки артефактов. Проверить загрузку верной версии: `curl localhost:5000/api/strategy/full | jq '.baseline_version'`

### Лог аудита исполнения ордеров

Для каждого заполненного ордера добавлять запись в `data/execution_audit.jsonl`:
```json
{
  "ts":                    "2026-05-19T14:30:22Z",
  "signal_id":             "uuid",
  "model_version":         "v1_2026-05-19",
  "feature_snapshot_hash": "sha256-признаков-на-момент-решения",
  "strategy":              "Trend_ML",
  "model":                 "trend_model_4h",
  "timeframe":             "4h",
  "mode":                  "live",
  "predicted_prob":        0.73,
  "model_confidence":      0.73,
  "threshold":             0.60,
  "expected_ev":           0.0042,
  "meta_passed":           true,
  "regime_used":           "trending",
  "garch_used":            true,
  "actual_fill_price":     65432.10,
  "slippage":              0.00018,
  "latency_ms":            47,
  "pnl_usdt":              null
}
```
`pnl_usdt` — `null` при входе; при закрытии позиции писать вторую запись с тем же `signal_id` и `"event": "close"`. Без `feature_snapshot_hash` + `model_version` невозможно отличить ошибку модели от ошибки исполнения. Без `meta_passed` + `regime_used` + `garch_used` карточка Interaction Matrix не сможет атрибутировать P&L по комбо.

**Дисциплина записи `execution_audit.jsonl` (WAL-паттерн):**
- Открывать только в **append-only** режиме (`open(path, 'a')`) — никогда не перезаписывать
- Вызывать `f.flush(); os.fsync(f.fileno())` после каждой записи — гарантирует сохранность при краше VPS
- Ротация при превышении 100 МБ: переименовать в `execution_audit_YYYYMMDD.jsonl`, открыть новый файл
- Частичная последняя строка (краш посреди записи) — обнаруживается и пропускается при чтении; все предыдущие записи JSONL остаются целыми

### Карточка Interaction Matrix (дашборд)

Цель: видеть P&L по каждому комбо фильтров прямо в Analytics-вкладке без ручного grep по логам.

**Backend endpoint — `GET /api/analytics/interaction_matrix`** (новый, `src/dashboard/app.py`):
- Читает `data/execution_audit.jsonl` построчно; пропускает сломанную последнюю строку (crash-safe)
- Возвращает `{"ok": true, "has_data": false, "rows": []}` если файл отсутствует или пуст
- Принимает `?mode=live|paper|all` (по умолчанию: `all`)
- Группирует **close**-записи по `(meta_passed, regime_used != null, garch_used)` булевым значениям
- На группу: `n_trades`, `win_rate`, `avg_pnl_usdt`, `trade_sharpe = mean(pnl_usdt) / std(pnl_usdt)` (null при n < 2)

**Пять канонических комбо:**

| # | Название | meta_passed | regime_used | garch_used |
|---|----------|-------------|-------------|------------|
| 1 | Trend only | false | false | false |
| 2 | Trend + Meta | true | false | false |
| 3 | Trend + Regime | false | true | false |
| 4 | Trend + Meta + Regime | true | true | false |
| 5 | Trend + Meta + Regime + GARCH | true | true | true |

**Frontend карточка** (`src/dashboard/templates/index.html`, вкладка Analytics):
- id карточки: `card-analytics-interaction-matrix`
- Автозагрузка при клике на вкладку Analytics (добавить вызов `loadInteractionMatrix()` рядом с `anLoad()`)
- JS функция: `loadInteractionMatrix()`
- Колонки: Комбо | Сделок | Win% | Средний P&L (USDT) | Trade Sharpe
- Цвет: Sharpe > 0.5 → зелёный, 0–0.5 → жёлтый, < 0 → красный, null → серый `—`
- Переключатель режима: Live / Paper / All
- Состояние-заглушка: *"Ожидаю первых live-сделок с `meta_passed` / `regime_used` / `garch_used` в `execution_audit.jsonl`"*

### Компоненты для разработки / расширения

| Компонент | Путь | Примечание |
|-----------|------|------------|
| Менеджер бейзлайнов | `src/governance/baseline_manager.py` | Новый — единый файл; `PromotionPolicy` + `rollback()` как методы класса |
| Dataset fingerprinter | `src/utils/dataset_fingerprint.py` | Новый — логический хеш + кэш |
| Env manifest | `src/utils/env_manifest.py` | Новый — `capture_env_manifest()` |
| Агент сравнения | `src/agents/model_comparison_agent.py` | Новый — supervised цикл |
| Стресс-тестер | `src/utils/trading_cost_stress_test.py` | Новый |
| Correlation gate | `src/risk/correlation_gate.py` | Новый |
| Карточка дашборда | Вкладка Аналитика | Таблица + canary статус + кнопка |
| Endpoint продвижения | **Переиспользовать** `POST /api/analytics/baseline` (`app.py:6418`) | Не создавать дубликат |
| Endpoint Interaction Matrix | `GET /api/analytics/interaction_matrix` (`app.py`) | Новый — читает `execution_audit.jsonl`, группирует по комбо |
| Карточка Interaction Matrix | Вкладка Аналитика, `card-analytics-interaction-matrix` (`index.html`) | Новая — автозагрузка при клике на вкладку |

---

## Фаза 10 — Риск-контроли: Kill-Switch, Liquidity Filter, Outage Mode

### Kill-Switch (расширить существующий `src/risk/kill_switch.py`)

Уже реализовано в `KillSwitchConfig` (строки 46–52):
- `drawdown_pct_threshold: 0.08` (8%) ← уже есть
- `latency_p99_ms_threshold: 500.0` мс ← уже есть
- `daily_loss_R_multiple` прокси для потерь ← уже есть

**Что добавить:**
- Поле `slippage_pct_threshold` в `KillSwitchConfig`
- Соответствующая проверка в `_iter_triggers` после строки 209
- Вызывающий код передаёт `slippage_pct` в словаре метрик

### Exchange Outage Mode (расширить `src/main.py`)

Уже реализовано: проверка подключения при запуске с экспоненциальным backoff (5с→80с), реконнект WS с backoff (макс 60с).

**Чего НЕТ — HIGH приоритет:**
- Нет runtime-флага `ws_connected`, который проверяет путь исполнения ордеров
- Во время реконнекта WS торговый цикл продолжает работать — может открыть позиции на устаревших данных

**Что добавить:**
- Булев флаг `ws_connected`: `False` при дисконнекте (~строка 1479), `True` при реконнекте
- Pre-trade gate: `ws_connected == False` → блокировать новые позиции; закрытие/риск-редукция разрешены
- **Сверка состояния при каждом реконнекте** (перед переключением `ws_connected = True`):

  За 60 секунд дисконнекта биржа могла исполнить TP/SL, заполнить ордер или цена могла сделать гэп. Слепое подключение = торговля на устаревшем локальном состоянии.

  ```python
  # При реконнекте — ДО ws_connected = True:
  futures_positions = client.futures_position_information()   # GET /fapi/v2/positionRisk
  spot_open_orders  = client.get_open_orders()                # GET /api/v3/openOrders
  spot_account      = client.get_account()                    # GET /api/v3/account

  # Сверяем расхождения:
  #   Позиция закрыта на бирже, но открыта локально  → пометить закрытой, зафиксировать P&L
  #   Ордер исполнен на бирже, но pending локально   → обновить статус, скорректировать инвентарь
  #   Несоответствие баланса                         → перезаписать локальный кэш значением биржи
  # ЗАТЕМ: ws_connected = True
  ```

### PreTradeGate — Единая проверка + паттерн двух замков (ОБЯЗАТЕЛЬНО)

Все условия безопасности (kill-switch, ws_connected, warmup_complete, SAFE_MODE, NaN/Inf-защита) обязаны проверяться в **одном унифицированном** вызове `PreTradeGate.check(ctx)` — НЕ разбросаны по отдельным if-проверкам в разных местах кода (классический баг пропущенного пути).

**Требуются два отдельных замка:**

```python
trading_lock = threading.Lock()   # сериализует только размещение ордеров
flag_lock    = threading.Lock()   # сериализует только мутации флагов

# Путь размещения ордера:
with trading_lock:
    gate = PreTradeGate.check(ctx)   # читает флаги под flag_lock внутри
    if not gate.allow:
        log(gate.reason); return
    exchange.create_order(...)

# Путь мутации флагов (WebSocket-поток, kill-switch, переключение SAFE_MODE):
with flag_lock:
    ws_connected = False   # или SAFE_MODE = "read_only", и т.д.
```

**Зачем два замка (не один):** Один `trading_lock` для и размещения ордеров, и записи флагов WebSocket → дедлок. WebSocket I/O-поток держит замок в ожидании сетевого ответа; торговый цикл заблокирован в ожидании замка. С раздельными замками `trading_lock` оборачивает только критическую секцию размещения, `flag_lock` — только запись флагов.

**Защита от TOCTOU:** `PreTradeGate.check()` читает все флаги атомарно под `flag_lock` внутри. Вызывающий код никогда не читает флаги отдельно до вызова `check()` — это создаёт окно, где флаг может измениться между чтением и ордером.

### Гейт размера позиции (новый)

Жёсткие лимиты, применяемые при генерации ордера до отправки на биржу:

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| `max_risk_per_trade` | 0.25–0.5% bankroll | При $500: $1.25–$2.50 на сделку |
| `max_daily_risk` | 2% bankroll | При $500: максимум $10/день |
| `max_open_positions` | N (настраивается по стратегии) | Предотвращает коррелированную перегрузку |

Если сигнал превысит любой лимит → **уменьшить до лимита, не пропускать сделку**. Логировать корректировку. Если уменьшение даст ордер ниже MIN_NOTIONAL → пропустить сделку с логированием причины.

**Компонент:** добавить `PositionSizingGate` в `src/risk/position_sizing.py` (новый файл или расширение существующего).

### Read-Only Safe Mode

Добавить операционный флаг `SAFE_MODE`. При `SAFE_MODE=read_only`:
- Бот получает рыночные данные штатно
- Вычисляет сигналы и признаки штатно
- Пишет в `execution_audit.jsonl` как paper-трейды (метка `"mode": "paper"`)
- **НЕ отправляет ордера на биржу**

Условия автоматического включения:
- После любого deploy новой модели (оставаться в `read_only` 30 минут)
- После реконнекта с аномальным state diff (reconciliation нашёл закрытые позиции или несоответствие баланса)
- После drift-алерта из Фазы 11
- После сброса kill-switch (ручной ре-энейбл → стартует в `read_only`, оператор явно повышает до `live`)

**Компонент:** добавить `SAFE_MODE` env var + гейт в путь размещения ордеров в `src/main.py`.

### Требование прогрева модели (Model Warmup)

После рестарта бота или перезагрузки модели — блокировать ордера до завершения прогрева:

| Индикатор | Минимум баров |
|-----------|---------------|
| RSI (14) | 14 баров |
| EMA (период N) | 3×N баров (для стабилизации) |
| Rolling windows признаков | max lookback по всем активным признакам |
| Состояние Regime GMM | 1 полный цикл предсказания |

**Последовательность загрузки данных при старте (7 шагов — обязательный порядок):**
```
1. Загрузить исторические закрытые бары из Parquet
2. Отбросить неполный последний бар (текущая незакрытая свеча)
3. Убедиться что все timestamp — UTC (нет naive datetime)
4. Убедиться в актуальности Parquet — самый свежий бар не старше max lookback окна
5. Загрузить недостающие бары через REST до текущей минуты
5.5. Проверить непрерывность — нет пробела И нет перекрытия между хвостом Parquet и барами REST
     (пробел = отсутствуют бары; перекрытие = бар REST дублирует бар Parquet → двойной счёт)
6. Включить WebSocket для live-обновлений свечей
7. Установить warmup_complete = True после накопления max_required_bars
```

Только после шага 7 `PreTradeGate.check()` разрешает размещение ордеров. Шаги 1–5.5 выполняются синхронно при старте до подключения WebSocket.

Реализация: флаг `warmup_complete: bool = False` при старте, `True` после загрузки `max_required_bars`. Pre-trade гейт проверяет флаг наряду с `ws_connected`.

NaN/Inf в признаках во время прогрева → блокировка ордера без ошибки (ожидаемое поведение при инициализации).

### Защита от NaN / Inf

Перед каждым вызовом инференса модели:
```python
assert np.isfinite(features).all(), (
    f"Не-конечные значения в векторе признаков: "
    f"{features.columns[~np.isfinite(features).all()].tolist()}"
)
```
LightGBM НЕ падает на NaN — он тихо применяет внутренний обработчик NaN, который может давать непредсказуемые предсказания. Одно плохое значение funding rate, деление на ноль в признаке или испорченная parquet-партиция незаметно доходит до live-сигнала.

При срабатывании ассерта: логировать и пропускать сигнал (не крашить бота).

### Мониторинг дрейфа часов (Clock Drift)

Подписанные запросы Binance используют timestamp сервера. Если часы VPS уходят относительно биржи:
- Запросы отклоняются с `INVALID_TIMESTAMP` (порог: дефолтный recvWindow 5000мс)
- Выравнивание свечей смещается → неправильное приписывание бара
- Funding timestamps смещаются → неправильный учёт P&L по фандингу

Проверка при старте бота и каждые 5 минут:
```python
exchange_time_ms = client.get_server_time()["serverTime"]
local_time_ms    = int(time.time() * 1000)
drift_ms         = abs(local_time_ms - exchange_time_ms)
if drift_ms > 500:
    alert(f"Дрейф часов {drift_ms}мс — синхронизировать NTP немедленно")
```
Исправление: `chronyc makestep` (Ubuntu 24.04 — `ntpdate` устарел и не установлен по умолчанию; `chronyc makestep` принудительно синхронизирует время через работающий демон chronyd без риска конфликта). Добавить в предстартовую проверку бота.

### Нормализация точности биржи (Exchange Precision)

Перед каждым размещением ордера нормализовать количество и цену до требований биржи:
```python
# Рекомендовано: встроенная точность CCXT (обрабатывает крайние случаи автоматически)
qty   = exchange.amount_to_precision(symbol, raw_qty)
price = exchange.price_to_precision(symbol, raw_price)

# Альтернатива при прямом Binance API — использовать Decimal во избежание бинарных ошибок float:
from decimal import Decimal, ROUND_DOWN
step = Decimal(str(step_size))
qty  = float((Decimal(str(raw_qty)) / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step)
# ВНИМАНИЕ: math.floor(0.123 / 0.001) = 122.999... = 122 из-за неточности float. Никогда не использовать math.floor для этого.
```
Без нормализации биржа молча отклоняет ордера (`LOT_SIZE` / `PRICE_FILTER`), что легко принять за сетевой сбой. Разные пары имеют разные `step_size` и `tick_size` — брать из `GET /api/v3/exchangeInfo`, кэш 24ч.

### Окна блокировки фандинга (Futures)

За 2 минуты до каждого расчёта фандинга ликвидность фьючерсов резко падает:

| UTC время | Событие фандинга |
|-----------|-----------------|
| 07:58–08:00 | Расчёт фандинга 08:00 |
| 15:58–16:00 | Расчёт фандинга 16:00 |
| 23:58–00:00 | Расчёт фандинга 00:00 |

В окна блокировки:
- **Не открывать новые фьючерсные позиции**
- Увеличить предположение по slippage в 2× для любых заполнений (если позиция уже открыта)
- Опционально: уменьшить плечо существующих позиций на 50%

**Реализация:** проверять `datetime.now(timezone.utc)` при генерации ордера, пропускать вход если в 2-минутном окне блокировки. (`datetime.utcnow()` устарел в Python 3.12+ и возвращает naive datetime — использовать `datetime.now(timezone.utc)` для timezone-aware UTC.)

### Minimum Liquidity Filter (новый, динамическая проверка)

Проверка при генерации ордера, кэш 60 секунд TTL через:
- `GET /api/v3/ticker/24hr` для volume
- `GET /api/v3/ticker/bookTicker` для spread и depth

| Метрика | Spot минимум | Futures минимум |
|---------|-------------|-----------------|
| 24ч объём (USD) | $50М | $100М |
| Bid-ask спред | ≤ 0.05% | ≤ 0.03% |
| Глубина стакана на 0.1% от mid | ≥ $50К | ≥ $50К |

Все 20 символов проходят порог по объёму. Ограничивающий фактор — спред: **SHIB, HBAR, ICP, SUI** периодически расширяются до 0.08–0.15% в UTC 02:00–06:00. Динамическая проверка это поймает; статическая при старте сессии — нет.

Фильтр **динамический** — проверка при генерации ордера, не раз при старте.

---

## Фаза 11 — Online Drift Monitoring (расширить `src/risk/drift_monitor.py`)

**Расширять существующий файл** — не создавать новый `drift_monitor_agent.py` (иначе дублирование с `src/risk/drift_monitor.py`, `drift_psi.py`, `drift_baseline.py`).

| Метрика | Порог | Примечание |
|---------|-------|------------|
| PSI — price/return признаки | 0.20 | Близко к нормальному распределению |
| PSI — volume/liquidation/funding | 0.25–0.35 | Fat-tailed; стандартный 0.2 даёт false positives |
| KL Divergence | TBD по признаку | Только в ежедневном агрегированном отчёте |
| Feature drift (mean/std) | > 3σ | Z-score по каждому признаку |
| Volatility drift | > 2σ rolling 30d | Сигнал смены рыночного режима |

**Минимальный размер выборки:** 500 баров на окно до вычисления PSI.

**Двухуровневая проверка:**
1. **Почасовая:** PSI vs rolling 24-часовое окно (НЕ статичный train distribution) — предотвращает false positives от внутридневной сезонности (открытие сессий Азия/ЕС/США, 8-часовые циклы funding rate)
2. **Ежедневная:** KL divergence vs training distribution — глубокий дрифт относительно оригинальных данных

Поток: превышение порога → обновление `data/drift_report.json` → флаг "Drift Detected" в дашборде. Оператор решает: игнорировать, расследовать или запустить ретрейн. **Никогда автоматически.**

Reference distributions сохраняются при обучении: `data/baselines/vN/train_distributions/`

---

## Ворота выполнения

```
Фаза 0   (Хаускипинг + ветка + хардениг VPS)           — одно ГО, ~45 мин
Фаза 1   (Баги A/Б/В)                                 — ~5 мин
Фаза 2   (rsync parquet + единоразовый бэкап GDrive)   — ~30 мин + ~60 мин бэкап
Фаза 3   (Миграция CSV.gz + touch + cron)              — ~20 мин
Фаза 4   (Изменения кода: FileNotFoundError, схема)    — ~30 мин + тесты
Фаза 5   (rclone ежедневный sync)                      — ~10 мин + браузер
──────────────────────────────────────────────────────────────────────────
Фаза 6   (Smoke-test: синт. данные + CPU + GPU)        — отдельное ГО
──────────────────────────────────────────────────────────────────────────
Фаза 7   (Архивирование, бот остановлен)               — отдельное ГО после 6
Фаза 8   (Ретрейн: regime первым, OOS, pull rsync)     — отдельное ГО
Фаза 9   (Бейзлайны + canary + стресс-тест)            — параллельно с 8 или после
Фаза 10  (Kill-switch слипаж + outage mode + liquidity) — параллельно
Фаза 11  (Расширить drift monitor)                     — после Фазы 9
──────────────────────────────────────────────────────────────────────────
PR ревью + мерж в main                                  — после всех фаз + 0 failures
```

---

## Правила биллинга (ОБЯЗАТЕЛЬНО)

- **Hetzner**: всегда **УДАЛЯТЬ** сервер — никогда не выключать. Удалять в exception handler. Подтверждать через API. Логировать ID при создании.
- **Vast.ai**: всегда **УНИЧТОЖАТЬ** инстанс — никогда не останавливать. Уничтожать в exception handler. Подтверждать через API.
