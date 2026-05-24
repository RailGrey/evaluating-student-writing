# Оценка сочинений: сегментация дискурсных элементов

Хашаев Раиль Муслимович

# **Постановка задачи**

Проект основан на соревновании Kaggle "Feedback Prize — Evaluating Student Writing" (https://www.kaggle.com/competitions/feedback-prize-2021/overview). Задача состоит в том, чтобы автоматически выделять в школьных argumentative essays смысловые элементы текста: Lead, Position, Claim, Counterclaim, Rebuttal, Evidence и Concluding Statement. По сути это задача сегментации текста и извлечения спанов, близкая к NER.

## **Формат входных и выходных данных**

На этапе обучения модель получает полный текст сочинения и разметку фрагментов: номер начала и конца спана в токенах/словах и тип дискурсного элемента. Данные разметки хранятся в train.csv, а сами эссе — в наборе .txt файлов. На этапе инференса на вход подаётся только сырой текст сочинения.

На выходе требуется вернуть набор фрагментов текста с их классами. Для удобства решения задачу можно сформулировать как token classification / sequence labeling с последующим восстановлением непрерывных спанов.

## **Метрики**

Для оценки качества используется несколько метрик:

1. **Официальная метрика соревнования (Competition F1)** — сравнивает предсказанные и эталонные спаны по индексам слов. Предсказание считается совпавшим с эталоном, если доля пересечения предсказания с эталоном и эталона с предсказанием обе не меньше 0.5. Если для одного эталона найдено несколько кандидатов, выбирается пара с максимальной суммой пересечений. Если спан не нашёлся в качестве пары, он считается false negative, а лишние предсказанные спаны — false positive.
2. **Token-level F1** — оценивает качество на уровне отдельных слов (токенов), а не целых спанов. Полезна для диагностики, какие именно классы путаются между собой.
3. **Span Exact Match F1** — считает спан совпавшим только при точном равенстве множеств слов. Более строгая метрика, чем Competition F1.
4. **Span Jaccard IoU** — средняя доля пересечения (Intersection over Union) для найденных пар спанов. Показывает, насколько хорошо модель подобрала границы, даже если полного совпадения нет.
5. **Macro F1** — усреднение F1 по всем классам (без учёта класса "O"). Используется как сводная метрика для каждой из вышеперечисленных.

Ожидаемые значения: простой бейзлайн (классификация предложений + TF-IDF + XGBoost) даёт Competition F1 ~0.2 (https://www.kaggle.com/code/julian3833/feedback-baseline-sentence-classifier-0-226). Лучшие решения на лидерборде соревнования достигают ~0.753 (https://www.kaggle.com/competitions/feedback-prize-2021/leaderboard).

## **Особенности данных и сложности**

- **Дисбаланс классов**: распределение дискурсных элементов неравномерно — например, Evidence встречается значительно чаще, чем Counterclaim или Rebuttal. Это требует взвешивания loss или использования специализированных техник.
- **Шум в разметке**: данные размечены экспертами, однако границы спанов могут быть субъективны; один и тот же фрагмент текста может нести несколько функций.
- **Длинные тексты**: эссе могут достигать 1000+ слов, что превышает лимит контекста многих моделей (BigBird, в отличие от BERT/RoBERTa, поддерживает последовательности до 4096 токенов).
- **Вложенность и последовательность**: дискурсные элементы часто идут в определённом порядке (Lead → Position → Claim → Evidence), но могут и перекрываться.
- **Неразмеченные фрагменты**: часть текста не аннотирована вообще (метка "O" / Other), что создаёт дополнительный шум для модели.

## **Пример входных данных**

Сырой текст эссе (первые 2 предложения):

> Have you ever wondered why we have certain rules? I believe we have rules for a good reason.

Разметка для этого эссе в train.csv:

| id | discourse_type | discourse_start | discourse_end | discourse_text | predictionstring |
|---|---|---|---|---|---|
| 0002B36B1AF6 | Lead | 0 | 353 | Have you ever wondered why we have certain rules? I believe we have rules for a good reason. | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 |

Здесь `discourse_start`/`discourse_end` — позиции символов, а `predictionstring` — индексы слов (0..21).

## **Валидация и тест**

Для внутренней проверки выделяется 10% эссе валидацией, split делается по essay_id, чтобы фрагменты одного и того же сочинения не попадали одновременно в train и valid. Это важно, потому что внутри одного эссе дискурсные элементы сильно коррелируют.

## **Датасеты**

По описанию датасета Kaggle, обучающая выборка содержит 15 594 уникальных сочинения, размеченных на 162 230 дискурсных элементов; в среднем это около 10.4 спанов на эссе. Датасет занимает 66 МБ. Датасет опубликован в 2021 году в рамках соревнования университетом штата Джорджия, подробнее про него можно прочитать в статье (https://www.sciencedirect.com/science/article/pii/S1075293522000630). Эссе написаны школьниками США 6–12 классов и сопровождаются экспертной разметкой. Ссылка на данные: https://www.kaggle.com/competitions/feedback-prize-2021/data.

# **Моделирование**

## **Бейзлайн**

В качестве простого бейзлайна задача решается как классификация предложений:

1. **Предобработка**: текст эссе разбивается на предложения при помощи библиотеки nltk (`sent_tokenize`). Это обеспечивает корректное разбиение при наличии аббревиатур, многоточий и т.д.
2. **Разметка предложений**: каждому предложению назначается метка дискурсного элемента на основе перекрытия с аннотациями (порог перекрытия по словам — 0.3). Если предложение не перекрывается ни с одним спаном, ему присваивается метка "Other".
3. **Признаки**: для каждого предложения строится TF-IDF представление (1–2 n-граммы, до 20000 признаков, с sublinear TF — `sklearn.feature_extraction.text.TfidfVectorizer`).
4. **Модель**: XGBoost (multiclass, `multi:softprob`, 300 деревьев, max_depth=3, learning_rate=0.1, subsample=0.8). Гиперпараметры настраиваются через конфиг Hydra.
5. **Валидация**: split по essay_id через `GroupShuffleSplit` (20% валидация). На валидации считаются Competition F1, Token F1, Span Exact Match F1, Span Jaccard IoU.
6. **Постобработка**: после предсказания соседние предложения одного класса объединяются в сегменты; класс "Other" отбрасывается (`merge_segments`).

## **Основная модель (NER + BigBird)**

Основное решение строится как задача NER с BIO-разметкой на уровне слов:

1. **Предобработка (BIO-разметка)**:
   - Текст разбивается на слова (по пробелам).
   - Для каждого слова определяется метка: B-{discourse_type} для первого слова спана, I-{discourse_type} для остальных слов спана, O для неразмеченных слов.
   - Токенизация через HuggingFace `AutoTokenizer` с параметром `is_split_into_words=True`, что позволяет сохранить маппинг токенов на исходные слова.
   - Padding/truncation до 1024 токенов. Для субтокенов используется `label_all_subtokens=True` — если слово размечено, все его субтокены получают ту же метку.

2. **Архитектура**:
   - Backbone: `google/bigbird-roberta-base` (HuggingFace `AutoModelForTokenClassification`).
   - BigBird выбран из-за длинных эссе (поддерживает до 4096 токенов против 512 у BERT/RoBERTa) благодаря sparse attention механизму.
   - Выходной classification head: полносвязный слой с 15 выходами (7 классов × B/I + O).
   - Dropout: 0.1 на hidden слоях для регуляризации.

3. **Обучение**:
   - **Loss function**: Cross-Entropy Loss (встроенный в `AutoModelForTokenClassification`), токены с меткой -100 (padding) игнорируются.
   - **Оптимизатор**: Adam (AdamW-style от HuggingFace), learning rate = 2.5e-5.
   - **LR scheduler**: MultiStepLR с decay в 0.1x на шагах 1000 и 2000.
   - **Gradient clipping**: max grad norm = 10.0.
   - **Early stopping**: по val_loss (patience=5, min_delta=0.001).
   - **Максимум**: 2500 шагов, batch size = 4.
   - **Мониторинг**: train_loss, val_loss, learning_rate, Competition F1 macro, Token F1 macro, Span Exact Match F1 macro, Span Jaccard IoU.

4. **Постобработка**:
   - Восстановление предсказаний на уровне слов: маппинг токенов обратно на слова (берётся первый токен каждого слова).
   - Объединение последовательных I-{class} токенов в непрерывные спаны.
   - Фильтрация спанов короче `min_span_length` (8 слов) — эвристика для отсечения шумовых предсказаний.

5. **Эксперименты и отслеживание**:
   - Все эксперименты логируются в MLflow (loss, метрики, learning rate, гипепараметры, модель).
   - Конфигурация через Hydra (YAML-конфиги для путей, модели, обучения, признаков).

# **Внедрение**

## **Формат модели и пайплайн инференса**

Финальная модель сохраняется в формате HuggingFace `transformers` (config.json + model.safetensors + tokenizer files) через `model.save_pretrained()`. Этот формат совместим с ONNX-экспортом при необходимости.

Пайплайн инференса включает:
1. **Загрузка модели**: `AutoModelForTokenClassification.from_pretrained()` + `AutoTokenizer.from_pretrained()`.
2. **Предобработка текста**: токенизация с `is_split_into_words=True`, padding до max_length=1024.
3. **Предсказание**: forward pass модели в режиме `torch.no_grad()`, argmax по logits.
4. **Постобработка**: маппинг токенов → слова, объединение в спаны, фильтрация по `min_span_length`.
5. **Формат вывода**: CSV с колонками `id`, `class`, `predictionstring` — совместимый с Kaggle-сабмишеном.

## **Ресурсы и требования**

- **Оборудование для обучения**: 1 GPU с ≥8 GB VRAM (например, NVIDIA RTX 3070/2080, Tesla T4). BigBird требует больше памяти, чем BERT-base, из-за длины последовательности.
- **Время обучения**: ~30–60 минут на полный прогон (2500 шагов, batch=4).
- **Оборудование для инференса**: CPU или GPU — модель загружается в полном виде (~1.2 GB), инференс одного эссе занимает <1 секунды на GPU.
- **Latency**: на GPU <1 с/эссе, на CPU ~2–5 с/эссе.
- **Формат развёртывания**: CLI-команда через Hydra, поддерживаются baseline и lightning-модели. В перспективе — обёртка в FastAPI для REST API.

---

# **Техническое руководство**

## **Структура проекта**

```
├── pyproject.toml              # Зависимости проекта (uv)
├── uv.lock                     # Lock-файл зависимостей
├── .python-version             # Версия Python (3.13)
├── .env                        # Переменные окружения (MLflow URI и т.д.)
├── .pre-commit-config.yaml     # Конфигурация pre-commit хуков (ruff, формат)
├── main.py                     # Точка входа Hydra (baseline train)
├── configs/
│   ├── config.yaml             # Главный конфиг (дефолты)
│   ├── experiment/
│   │   └── mlflow.yaml         # Настройки MLflow
│   ├── features/
│   │   ├── tfidf.yaml          # TF-IDF параметры (baseline)
│   │   └── tokenizer.yaml      # Токенизация (lightning)
│   ├── model/
│   │   ├── bigbird.yaml        # BigBird (lightning)
│   │   └── xgboost.yaml        # XGBoost (baseline)
│   ├── paths/
│   │   └── default.yaml        # Пути к данным и артефактам
│   ├── preprocessing/
│   │   └── sentence.yaml       # Параметры предобработки (baseline)
│   └── training/
│       └── ner.yaml            # Параметры обучения (lightning)
├── evaluating_student_writing/ # Пакет проекта
│   ├── __init__.py
│   ├── baseline/               # Baseline (XGBoost + TF-IDF)
│   │   ├── data.py             # Загрузка, TF-IDF, split
│   │   ├── train.py            # Тренировка XGBoost
│   │   ├── infer.py            # Инференс XGBoost
│   │   └── utils.py            # Разбиение на предложения, merge_segments
│   └── lightning/              # Основная модель (Lightning + BigBird)
│       ├── __init__.py
│       ├── constants.py        # BIO-метки, LABEL2ID, ID2LABEL
│       ├── data.py             # EssayDataset, BIO-разметка, загрузка
│       ├── model.py            # NERLightningModule
│       ├── train.py            # Тренировка (самостоятельный entry point)
│       ├── infer.py            # Инференс (самостоятельный entry point)
│       ├── utils.py            # predictions_to_spans, collate_fn
│       └── plot_utils.py      # Визуализация train/val loss
├── metrics/                    # Метрики оценки
│   ├── __init__.py             # evaluate() — общий entry point
│   ├── base.py                 # _match_group, _build_per_class, aggregate_averages
│   ├── competition_f1.py       # Competition F1 (метрика Kaggle)
│   ├── token_f1.py             # Token-level F1
│   ├── span_exact_match.py     # Exact Match F1
│   └── span_jaccard.py         # Span Jaccard IoU
├── data/                       # Данные (DVC)
│   ├── train.csv               # Разметка
│   ├── train/                  # .txt файлы эссе (15.6k)
│   └── test/                   # .txt файлы теста
├── models/                     # Сохранённые модели (gitignored)
│   └── baseline/
│       ├── bigbird/            # HuggingFace модель
│       ├── xgb_model.joblib
│       ├── tfidf_vectorizer.joblib
│       └── label_to_idx.joblib
├── notebooks/                  # Jupyter ноутбуки
├── scripts/                    # Вспомогательные скрипты (пусто)
├── outputs/                    # Hydra output (gitignored)
└── lightning_logs/             # Lightning logs (gitignored)
```

## **Установка и настройка (Setup)**

### Требования

- Python 3.13 (зафиксирован в `.python-version`)
- `uv` — пакетный менеджер (https://docs.astral.sh/uv/#installation)
- GPU с CUDA 12.x (рекомендуется, но не обязательно) и драйверы NVIDIA

### Шаги установки

```bash
# 1. Установить uv (если ещё не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Клонировать репозиторий
git clone https://github.com/RailGrey/evaluating-student-writing
cd evaluating-student-writing

# 3. Создать виртуальное окружение и установить зависимости
uv sync

# 4. Настроить pre-commit хуки (опционально, для разработки)
pre-commit install

# 5. Скачать данные через DVC (требуется доступ к Google Drive, напишите для запроса доступа)
dvc pull

# Без DVC: данные можно скачать вручную с Kaggle:
# https://www.kaggle.com/competitions/feedback-prize-2021/data
# После скачивания поместить train.csv в data/ и распаковать train.zip в data/train/
```

## **Обучение (Train)**

Проект поддерживает две модели: baseline (XGBoost) и основную (Lightning + BigBird).

### Baseline: Sentence Classification + XGBoost

**Запуск через `main.py` (Hydra):**

```bash
# Обучить baseline с конфигом по умолчанию
python main.py

# С переопределением параметров через Hydra CLI
python main.py \
  model=xgboost \
  features=tfidf \
  preprocessing=sentence \
  training=ner \
  model.n_estimators=500 \
  model.max_depth=6
```

**Что делает скрипт:**
1. Загружает эссе из `data/train/` и разметку из `data/train.csv`
2. Разбивает эссе на предложения (nltk) и назначает метки на основе перекрытия с аннотациями
3. Строит TF-IDF признаки и обучает XGBoost
4. Сохраняет модель, векторизатор и маппинг меток в `models/baseline/`
5. Вычисляет метрики на валидации (Competition F1, Token F1 и т.д.)

### Основная модель: Lightning + BigBird (NER)

**Запуск через модуль Lightning (отдельный entry point):**

```bash
# Обучить BigBird NER
python -m evaluating_student_writing.lightning.train

# С переопределением конфига
python -m evaluating_student_writing.lightning.train \
  model=bigbird \
  features=tokenizer \
  training=ner \
  training.max_steps=5000 \
  training.learning_rate=5e-5
```

**Что делает скрипт:**
1. Загружает эссе и разметку, делает split по essay_id (train/val = 90/10)
2. Создаёт EssayDataset с BIO-разметкой на уровне слов
3. Инициализирует NERLightningModule (BigBird + classification head)
4. Запускает обучение через PyTorch Lightning Trainer с:
   - MLflow-логированием метрик и параметров
   - Early stopping по val_loss
   - MultiStepLR scheduler
   - Gradient clipping
5. Сохраняет модель (HuggingFace format) в указанную директорию
6. Логирует всё в MLflow и генерирует графики train/val loss

### Визуализация пайплайна

```
                         ┌──────────────┐
                         │  train.csv   │
                         │  + .txt files│
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │   Split по   │
                         │  essay_id    │
                         └──┬───────┬───┘
                   ┌────────▼───┐ ┌─▼─────────┐
                   │  Train     │ │  Val       │
                   │  essays    │ │  essays    │
                   └─────┬─────┘ └─────┬──────┘
                         │              │
              ┌──────────▼──┐   ┌──────▼──────┐
              │ BIO-разметка│   │ BIO-разметка│
              │ Токенизация │   │ Токенизация │
              └──────┬─────┘   └──────┬───────┘
                     │                │
              ┌──────▼────────────────▼───────┐
              │        BigBird Model          │
              │  (AutoModelForTokenClass.)     │
              │   CrossEntropyLoss + Adam      │
              └──────────────┬────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │    Постобработка спанов       │
              │  (tokens → words → spans)    │
              │  Фильтрация min_span_length   │
              └──────────────┬────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  Оценка метрик (val)         │
              │  Competition F1, Token F1,   │
              │  Span EM F1, Span Jaccard    │
              └──────────────┬────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │   Сохранение модели +         │
              │   MLflow logging              │
              └──────────────────────────────┘
```

## **Инференс (Predict)**

### Baseline (XGBoost)

```bash
# Запустить инференс на тестовых данных
python -m evaluating_student_writing.baseline.infer

# С указанием своего пути к модели
python -m evaluating_student_writing.baseline.infer \
  paths.model_dir=models/my_model \
  paths.test_dir=data/my_test
```

### Lightning (BigBird NER)

```bash
# Запустить инференс
python -m evaluating_student_writing.lightning.infer

# С переопределением путей
python -m evaluating_student_writing.lightning.infer \
  paths.model_dir=models/my_model \
  paths.test_dir=data/my_test \
  paths.submission_path=data/my_submission.csv
```

**Что делает пайплайн инференса (Lightning):**

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Raw text│──▶│Tokenizer │──▶│ BigBird  │──▶│  Spans   │──▶│   CSV    │
│  .txt    │   │ (padding) │   │ Forward  │   │ (post-   │   │submission│
│          │   │max_len=   │   │ +argmax  │   │ process) │   │          │
│          │   │  1024     │   │          │   │min_span  │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## **Эксперименты (MLflow)**

Для отслеживания экспериментов используется MLflow:

```bash
# Запустить MLflow UI (по умолчанию на http://127.0.0.1:8080)
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 127.0.0.1 --port 8080
```

В MLflow логируются:
- Гиперпараметры (model_name, learning_rate, batch_size, max_steps, и т.д.)
- Train/val loss (по шагам и эпохам)
- Валидационные метрики (Competition F1, Token F1, Span EM F1, Span Jaccard)
- Графики train/val loss
- Git commit ID
- Сохранённая модель как артефакт

## **Разработка**

### Линтинг и форматирование

```bash
# Настроить окружение для разработки
uv sync --group dev

# pre-commit
pre-commit run --all-files
```

## **Данные**

Управление данными через DVC с Google Drive remote:

```bash
# Скачать данные
dvc pull

# После изменения данных
dvc add data/train.csv data/train/
dvc push

# Отслеживание статуса
dvc status
```

## **CLI-команды (шпаргалка)**

| Команда | Описание |
|---|---|
| `python main.py` | Baseline train (XGBoost) |
| `python -m evaluating_student_writing.baseline.infer` | Baseline predict |
| `python -m evaluating_student_writing.lightning.train` | Lightning train (BigBird NER) |
| `python -m evaluating_student_writing.lightning.infer` | Lightning predict |
| `mlflow ui` | Запустить MLflow Dashboard |
| `dvc pull` | Скачать данные |
| `pre-commit run --all-files` | Запустить pre-commit |
