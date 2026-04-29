# BaseFFT — Full Fine-Tune Qwen3-1.7B-Base на Kaggle 2×T4

> **Почему всё в блокнотах?** Проект был создан под работу на Kaggle: запускать код можно только через Notebooks - других способов запустить код Kaggle не предоставляет. Поэтому весь pipeline (EDA, тренировка, eval) оформлен как Jupyter-блокноты, а Python-модули записываются на диск через `%%writefile` прямо из ячеек.
>
> **Почему всё в fp16, а не в bf16?** GPU Tesla T4 (compute capability 7.5) **не поддерживают bf16** на уровне железа. Поэтому FSDP mixed precision и autocast настроены на `float16` для параметров и `float32` для редьюса градиентов — это компромисс между памятью и численной стабильностью на доступном железе.

---

## Задача

Превратить базовую (pretrained-only) модель `Qwen/Qwen3-1.7B-Base` в instruct-вариант через **полный fine-tune** (full fine-tune, FFT — все 1.7B параметров обучаются), используя только Kaggle-ресурсы (2×T4 16 GB, 20 GB диска `/kaggle/working`, 12-часовой лимит сессии).

В отличие от LoRA/QLoRA, FFT даёт более глубокую адаптацию модели, но требует серьёзной памяти и аккуратного управления чекпойнтами. Цель — показать, что даже на бюджетном железе (Kaggle Free Tier) можно прогнать честный SFT с замером качества.

Метрика финальной оценки — **IFEval** (instruction-following) через `lm-evaluation-harness`. Сравниваются: base ↔ N SFT-чекпойнтов на одних и тех же индексах промптов.

---

## Способ решения

**FSDP `FULL_SHARD`** на 2×T4: модель, градиенты и optimizer state шардируются между двумя GPU, что позволяет уместить 1.7B-параметрическую модель + оптимизатор в 2×16 GB VRAM. В качестве оптимизатора — `bitsandbytes.PagedAdamW8bit` (8-битные моменты Adam с paged-аллокацией для overflow в RAM).

Дополнительно:
- `gradient_checkpointing` (use_reentrant=False) — ещё снижение пика VRAM ценой ~30% времени.
- `torch.autocast(fp16)` в forward/backward.
- Mixed precision FSDP: `param_dtype=fp16, reduce_dtype=fp32` — fp32-редьюс защищает от накопления ошибок при all-reduce градиентов.
- Эффективный батч 16 = `per_device_bs=1 × world_size=2 × grad_accum=8`.
- `max_seq_len=1024` — компромисс между покрытием датасета и памятью.

Данные — `allenai/tulu-3-sft-mixture` (939K примеров). Берём strict-фильтрованный subset 50K: только примеры, которые ПОЛНОСТЬЮ помещаются в 1024 токена после применения ChatML-шаблона (никаких truncate — обрезанная инструкция != валидный SFT-пример).

---

## Этапы pipeline

### Этап 1 — `1-base-fft-eda.ipynb` (EDA + подготовка данных)

Загружаем `allenai/tulu-3-sft-mixture`, считаем распределение длин токенов после ChatML-шаблона по каждому из 19 источников, фильтруем:
1. `num_turns ≤ 4` (ограничение длины диалога).
2. `token_len ≤ 1024` после применения ChatML (strict drop, не truncate).
3. MD5-дедупликация по конкатенации всех `role:content`.
4. **Stratified sampling 50K** по `source` с сохранением пропорций после фильтрации (минимум 50 примеров на источник, чтобы не терять редкие).
5. **Stratified train/val split 95/5** по `source` → 47 499 / 2 501.

Выход публикуется как Kaggle Dataset `kotshubin/qwen-fft-data-50k` (`train.jsonl` + `val.jsonl` + `dataset_stats.json`).

ChatML-шаблон задаём вручную, потому что у `Qwen3-Base` `chat_template = None`. Шаблон дублируется во всех трёх блокнотах и должен оставаться синхронизированным.

### Этап 2 — `2-base-fft-fft.ipynb` (FSDP full fine-tune)

Тяжёлая часть. Архитектурно блокнот делает следующее:
1. Через `%%writefile` пишет 5 модулей в `/kaggle/working/src/`:
   - `utils.py` — `Logger` (rank-0 tqdm + JSON-метрики) + `set_seed`.
   - `data.py` — `ChatJsonlDataset` (ChatML + label masking: `-100` на префикс, чтобы loss считался только на последнем assistant-ответе) + `PadCollator` + `build_dataloaders` с `DistributedSampler`.
   - `model.py` — загрузка модели/токенайзера, `wrap_with_fsdp` (auto-wrap по `Qwen3DecoderLayer`), сборка optimizer+scheduler (warmup → cosine), eval-loop.
   - `checkpoint.py` — FSDP-aware save/load и финальный HF-экспорт.
   - `train_sft.py` — CONFIG + train loop + `main()` (entry point для `torchrun`).
2. Запускает: `!torchrun --nproc_per_node=2 --master_port=29500 /kaggle/working/src/train_sft.py`.
3. Cross-module импорты (`from utils import ...`) работают потому, что `torchrun` кладёт директорию скрипта в `sys.path[0]`.

Артефакты на rank 0:
- `/kaggle/working/checkpoints/ckpt-step-*.pt` — последние 2 чекпойнта (rotation `keep_last=2`).
- `/kaggle/working/SFT-final-model/` — HF-формат для eval (~3.4 GB, fp16).
- `/kaggle/working/metrics_train.json` — лог тренировки.

### Этап 3 — `3_eval_multi_local.ipynb` (локальный IFEval)

Этот блокнот запускается **локально**, не на Kaggle (Kaggle ограничивает время сессии и возможностей кастомного pip нет в нужном объёме). Скачиваем все нужные `ckpt-step-*.pt` в подпапку `./checkpoints/`, и блокнот:
1. Загружает базу `Qwen/Qwen3-1.7B-Base`, прогоняет IFEval, сохраняет метрики.
2. Для каждого `ckpt-step-*.pt`: загружает FSDP-payload (fp32) → кастит к fp16 → `model.load_state_dict(sd, strict=False)` поверх свежего HF-инстанса → экспортирует во временную HF-директорию → прогоняет IFEval.
3. Печатает pretty-table сравнение base vs все SFT-чекпойнты с дельтами.

IFEval запускается на 100 промптах из 541 (`NUM_BENCHS=100`, `SHUFFLE_SEED=42`). Чтобы дельты были корректными, monkey-patch на `lm_eval.evaluator.get_task_dict` гарантирует, что **все модели оцениваются на ОДНИХ И ТЕХ ЖЕ 100 индексах**.

---

## Технические ограничения и инженерные решения

### Kaggle-only execution
- Kaggle Free/Pro даёт ноутбуки, не CLI/контейнеры, не cron-jobs. Поэтому всё, что должно работать на их железе, оформлено как `.ipynb` с inline-`%%writefile`.
- Sessions ограничены ~12 часами — все тренировки и rotation чекпойнтов рассчитаны на возможный resume через перезапуск ноутбука с прикреплённым output-датасетом.

### T4 не поддерживает bf16
- GPU compute capability 7.5 умеет fp16/fp32, но не bf16. Все autocast и FSDP mixed-precision работают через `torch.float16`. Для устойчивости — `reduce_dtype=fp32` в FSDP MixedPrecision и fp32-master-weights в чекпойнтах.

### Дисковый бюджет 20 GB на `/kaggle/working`
- Один FSDP `FULL_STATE_DICT` чекпойнт fp32 для 1.7B-параметрической модели весит ~8 GB.
- Rotation выполняется **ДО** записи нового чекпойнта (а не после), `keep_last=2` → пик диска при write = `keep_last × 8 GB = 16 GB < 20 GB`. Если делать rotation после save, пик кратковременно достигнет `(keep_last+1) × 8 GB = 24 GB` и упадёт на ENOSPC.
- `save_final_hf` чистит `ckpt-step-*.pt` ПЕРЕД записью финальной HF-модели (~3.4 GB), потому что финальная модель важнее чекпойнтов.

### bitsandbytes PagedAdamW8bit + FSDP несовместимы по optim state
- `FSDP.full_optim_state_dict(...)` для bnb-оптимизатора падает в `_convert_all_state_info` с ошибкой "tensors on cuda:0 and cuda:1". Это известный баг.
- **Optim state намеренно НЕ сохраняется**. Resume восстанавливает model + scheduler + RNG; моменты Adam пересчитываются за несколько первых шагов. Это приемлемая цена.

### Auto-discovery входов через `rglob`
- Kaggle монтирует output-датасеты в разные пути в зависимости от способа подключения (UI vs CLI): `/kaggle/input/<slug>/...` или `/kaggle/input/notebooks/<owner>/<slug>/...`.
- Поэтому ни `train.jsonl`/`val.jsonl`, ни `ckpt-step-*.pt` не задаются хардкодом — везде `Path("/kaggle/input").rglob(...)` с выбором первого/максимального матча.

### RAM-фрагменты в локальном eval (блокнот 3)
- При загрузке `ckpt-step-*.pt`: payload (fp32, ~6.8 GB) + state_dict (fp16, ~3.4 GB) + сама модель (~3.4 GB) могут дать пик RAM ~13.6 GB.
- В коде явно расставлены `del payload; gc.collect()` и `del sd; gc.collect()` сразу после поглощения данных в модель.
- Перед загрузкой следующей модели вызывается `del model, tok` **на caller-side** (важно: `del` внутри функции освобождает только локальные ссылки), затем `_free_gpu()` с двойным `gc.collect()` (HFLM может оставить cycles).

---

## Конфигурация по умолчанию

| Параметр | Значение |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` |
| Dataset | `allenai/tulu-3-sft-mixture` → 50K subset |
| Max seq len | 1024 |
| Max turns | 4 |
| Train/val split | 47 499 / 2 501 (stratified by source) |
| GPUs | 2× Tesla T4 (16 GB) |
| FSDP | `FULL_SHARD`, auto-wrap по `Qwen3DecoderLayer` |
| Mixed precision | param=fp16, reduce=fp32, buffer=fp32 |
| Optimizer | `bnb.optim.PagedAdamW8bit`, betas=(0.9, 0.95) |
| LR | 2e-5, warmup 5% → cosine |
| Weight decay | 0.01 |
| Effective batch | 16 (`per_device=1 × world=2 × grad_accum=8`) |
| Max grad norm | 1.0 |
| Epochs | 1 |
| Gradient checkpointing | on (`use_reentrant=False`) |
| Eval | IFEval, 100 промптов, seed=42, greedy, `max_new_tokens=1024` |

---

## Запуск

### Блокноты 1 и 2 (Kaggle)
1. Создаёшь Kaggle Notebook на 2×T4.
2. Прикрепляешь HuggingFace token через User Secrets (`HF_token_super`).
3. Прогон блокнота 1 → `Save Version → Create Dataset` → `kotshubin/qwen-fft-data-50k`.
4. В блокноте 2 прикрепляешь этот датасет как input → `Save Version` (Run all) → ждёшь окончания тренировки.
5. Финальную модель и/или последний чекпойнт публикуешь как очередной dataset для следующего этапа.

### Блокнот 3 (локально)
```bash
pip install torch transformers lm-eval immutabledict langdetect nltk absl-py bitsandbytes
mkdir -p checkpoints
cp /path/to/ckpt-step-*.pt checkpoints/
jupyter notebook 3_eval_multi_local.ipynb
```
Можно положить любое количество чекпойнтов — eval сравнит их все против base в одной таблице.

---

## Структура репозитория

```
.
├── 1-base-fft-eda.ipynb       # EDA + подготовка train.jsonl/val.jsonl
├── 2-base-fft-fft.ipynb       # FSDP full fine-tune на 2×T4 (Kaggle)
├── 3_eval_multi_local.ipynb   # IFEval сравнение base vs N SFT (локально)
└── CLAUDE.md                  # инженерные заметки для будущих агентов Claude Code
```

Никакого Python-пакета, тестов или CI здесь нет — это намеренно. Pipeline живёт в трёх блокнотах, потому что таково ограничение Kaggle.
