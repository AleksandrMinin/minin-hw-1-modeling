## kaggle-planet-modeling

Решение задачи мультилейбл классификации спутниковых изображений Амазонки.


### Датасет

Включает 40479 классифицированных изображений(256x256x3) и 17 классов.
Скачать датасет и данные можно [отсюда](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space).

### Подготовка пайплайна

1. Создание и активация окружения
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка пакетов

    В активированном окружении:
    ```
    pip install -r requirements.txt
    ```

3. Разбиваем датасет на 3 части(train_df.csv, valid_df.csv, test_df.csv):
    ```
    ROOT_PATH=/data/planet-understanding-the-amazon-from-space python train_test_split.py
    ROOT_PATH=/storage/minin/datasets/planet-from-space python train_test_split.py
    ```
    P.S. В папке ROOT_PATH необходимо, чтобы ОТДЕЛЬНО лежал файл train.csv с классифицированными изображениями из датасета,
    а также папка с тренировочными изображениями train-jpg.

4. Настройка ClearML

    a. [В своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем:
      "Settings" -> "Workspace" -> "Create new credentials"
      
    b. Появится инструкция для "LOCAL PYTHON", копируем её.
    
    с. Пишем в консоли `clearml-init` и вставляем конфиг из инструкции.
