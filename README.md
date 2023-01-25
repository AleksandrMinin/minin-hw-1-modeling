# kaggle-planet-modeling

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

    a. Обновить pip
    ```
    pip install --upgrade pip 
    ```
    b. Выполнить команду
    ```
    pip install -r requirements.txt
    ```

3. Разбиваем датасет на 3 части(train_df.csv, valid_df.csv, test_df.csv):
    ```
    ROOT_PATH=/data/planet-understanding-the-amazon-from-space python train_test_split.py
    ```
    P.S. В папке ROOT_PATH необходимо, чтобы ОТДЕЛЬНО лежал файл train.csv с классифицированными изображениями из датасета,
    а также папка с тренировочными изображениями train-jpg.

4. Настройка ClearML

    a. [В своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем:
      "Settings" -> "Workspace" -> "Create new credentials"
      
    b. Появится инструкция для "LOCAL PYTHON", копируем её.
    
    с. Пишем в консоли `clearml-init` и вставляем конфиг из инструкции.

### Обучение
Запуск тренировки c `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/planet-understanding-the-amazon-from-space nohup python train.py > log.out
```

Запуск тренировки без `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/planet-understanding-the-amazon-from-space python train.py
```

### ClearML
Метрики и конфигурации экспериментов:
1. [experiment_1](https://app.clear.ml/projects/ff3c0bfc136344e78f782c01c14f28ed/experiments/12ec3b227b8e4e6dbd584a2a8201decd/output/execution)
2. [experiment_2](https://app.clear.ml/projects/ff3c0bfc136344e78f782c01c14f28ed/experiments/18dfe6335f74420a92a0dd5a295fd80d/output/execution)
3. [experiment_3](https://app.clear.ml/projects/ff3c0bfc136344e78f782c01c14f28ed/experiments/e0090bd8815744989bc45a43b733db21/output/execution)

### DVC
#### Добавление модели в DVC
1. Инициализация DVC

    В директории проекта пишем команды:
    ```
    dvc init
    ```
    ```
    dvc remote add --default myremote ssh://91.206.15.25/home/aleksandrminin/dvc_files
    ```

    ```
    dvc remote modify myremote user aleksandrminin
    dvc config cache.type hardlink,symlink
    ```

    подробнее про типы линков можно почитать [здесь](https://dvc.org/doc/user-guide/large-dataset-optimization#file-link-types-for-the-dvc-cache).

2. Добавление модели в DVC
    
    Копируем в `weights` обученную модель
    ```
    cd weights
    dvc add model.pt
    dvc push
   ```

3. Делаем коммит с новой моделью:
    ```
    git add .
    git commit -m "add new model"
   ```

#### Загрузка лучшей модели из DVC к себе
   ```
    git pull origin main
    dvc pull
   ```

