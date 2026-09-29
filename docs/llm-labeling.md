# Разметка траектории технологий (GigaChat)

Код — [src/lctrend/modeling/dataset/llm_outcomes.py](../src/lctrend/modeling/dataset/llm_outcomes.py),
промпт `technology-trajectory-v1`, результаты — ноутбук
[02_labeling](../notebooks/02_labeling.ipynb).

## Что размечается

Один запрос на технологию. Модель получает её историю в нашем корпусе по
годам: документы к году и новые за год, независимые группы, организации,
компании, типы источников, зрелость, а также датированные заголовки
документов. Ответ по **каждому году** истории:

| Поле | Значение |
|---|---|
| `score` | 0 — в этот год слабый сигнал (ранняя, мало независимых участников, позже подтвердилась); 1 — точно не слабый сигнал (шум, угасла или уже мейнстрим) |
| `hype` | перегретость внимания относительно содержания, 0..1 |
| `maturity` | сформированность: 0 — идея, 1 — массовое внедрение |

И по траектории целиком — `verdict` (`success`, `niche`, `faded`,
`mainstream`, `junk`, `unclear`), признак `is_technology` и обоснование.
Каждый срез года получает значения этого года.

Модель видит будущее технологии, поэтому это **метки исхода** для обучения
и никогда не признаки.

## Цель обучения

`signal_llm` ([training.json](../src/lctrend/resources/training.json) → `labels`):

- 1 — оценка года ≤ `weak_max` (0,3);
- 0 — оценка ≥ `not_weak_min` (0,7);
- промежуток не размечается;
- технологии с `is_technology = false` уходят из обучения как шум
  (`noise_technologies.csv`).

## Запуск

Ключ GigaChat с именем `Разметка` берётся из пула ключей по имени (в общем
пуле он выключен, чтобы воркеры загрузки его не занимали). Журнал JSONL
делает запуск возобновляемым: повторная команда продолжает с места
остановки.

Прямо из графа, без выгрузки:

```bash
python -m lctrend.modeling.dataset.llm_outcomes --from-graph \
  --output artifacts/modeling/2026-09-29-dedup/dataset/llm_labels.csv
```

После выгрузки — разложить ответы по срезам:

```bash
python -m lctrend.modeling.dataset.llm_outcomes \
  --history artifacts/modeling/2026-09-29-dedup/dataset/history.csv \
  --output artifacts/modeling/2026-09-29-dedup/dataset/llm_labels.csv
```

`--workers` — число параллельных запросов (для личного ключа 1),
`--limit` — разметить не больше N технологий за запуск.

Ключи задаются только в `.env` (`GIGACHAT_KEYS_FILE` или
`GIGACHAT_CREDENTIALS`, см. [gigachat-keys.example.json](../gigachat-keys.example.json))
и не попадают в git и командную строку. Корневой сертификат НУЦ Минцифры —
`src/lctrend/resources/russian_trusted_root_ca_pem.crt`, TLS-проверку
отключать не нужно.

## Итоги запуска 2026-09-29-dedup

3 106 технологий, 18 235 оценок «технология × год». Вердикты: не технология
39 %, мейнстрим 22 %, ниша 21 %, состоялась 7 %, угасла 6 %, неясно 5 %.
Слабых сигналов среди размеченных лет — 8 %.

## Запись в граф

Ноутбук [05_graph](../notebooks/05_graph.ipynb) или
`python -m lctrend.modeling.labeling.graph_labels` кладёт разметку в узлы
`Technology` (`llm_verdict`, `llm_is_technology`, `llm_score`, `llm_hype`,
`llm_maturity`, `llm_rationale`, `llm_years`) рядом с вероятностью модели
(`signal_*`). Поиск в интерфейсе читает эти свойства.
