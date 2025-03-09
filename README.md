# Как запускать

## Бейзлайн решение

### Окружение

Версия Python: `Python 3.11`
Операционная система: Linux x86_64
CUDA: 12.1+

Создание окружения:

```bash
poetry install --with dev
```

### Данные

Перед началом работы необходимо загрузить данные и разместить их в папке `data`. Для этого выполните команду `poetry run download`

Минимальные требования:

1 GB VRAM - inference

6 GB VRAM - train

### Запуск скриптов

1. Обучаем модель.

```bash
poetry run train --cfg-path=<path_to_your_model_config>
```

2. Формируем `submission.csv` с использованием построенной модели.

```bash
poetry run submit --cfg-path=<path_to_your_model_config> --model-path=<path_to_your_model_weights>
```

Презентация - в файле `Presentation.pdf`
