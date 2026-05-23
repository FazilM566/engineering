# Данные для проекта

## Источник данных

Датасет рентгеновских снимков грудной клетки (4 класса):  
- COVID  
- Lung opacity  
- Normal  
- Viral Pneumonia  

Скачать архив можно, например, с Kaggle:  
[Balanced Augmented COVID‑CXR Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/balanced-augmented-covid-cxr-dataset)  

Архив содержит только папки классов, **без разделения на train/val/test**.

## Подготовка данных для локальных скриптов (PyCharm)

1. Скачайте и распакуйте архив в любую папку, например `C:/data/raw/`.

2. Из корня проекта выполните команду (один раз):
   ```bash
   uv run python -m src.data.prepare --input C:/data/raw --output data/processed
   ```
   Скрипт создаст структуру `data/processed/train`, `data/processed/val`, `data/processed/test` с разбиением 70/15/15.

3. После этого скрипты `src/train.py` и `src/evaluate.py` будут автоматически использовать `data/processed` (по умолчанию).

## Подготовка данных для ноутбука (Google Colab)

В ноутбуке данные подготавливаются независимо. Рекомендуемый порядок действий:

1. Загрузите архив на Google Drive.
2. В ноутбуке смонтируйте диск:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
4. Распакуйте архив в нужную папку, например:
```bash
import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/archive.zip', 'r') as zip_ref:
zip_ref.extractall('/content/data/raw')
```
6. **Вариант А (рекомендуемый):** скопируйте в ноутбук код из `src.data.prepare` и выполните его, указав пути к исходной и целевой папкам. Это даст такую же структуру `train/val/test`, как в локальных скриптах.
   **Вариант Б (упрощённый):** используйте прямое разбиение на `train/val/test` прямо в ноутбуке (как в исходном коде). Убедитесь, что в дальнейшем пути к данным в ноутбуке указывают на подготовленные папки.

После подготовки путь к данным в ноутбуке можно задать переменной:

DATA_PATH = "/content/data/processed"


## Примечания

- Разбиение на `train/val/test` в скрипте `prepare.py` детерминировано (seed=42). В ноутбуке для воспроизводимости также фиксируйте `random.seed(42)`.
- Если вы хотите использовать в ноутбуке уже подготовленные данные (из локальной папки) – загрузите их в Google Drive или используйте `files.upload()`.


