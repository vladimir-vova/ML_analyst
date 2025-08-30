# Credit Risk Dataset Analysis
Проект для анализа датасета кредитного риска, где строится модель машинного обучения для предсказания дефолта заемщиков. Используется Gradient Boosting Classifier с оптимизацией гиперпараметров через Optuna для достижения высокой точности предсказаний. Это позволяет оценить риски и улучшить кредитные решения.

## Содержание
- [Технологии](#технологии)
- [Начало работы](#начало-работы)
- [Тестирование](#тестирование)
- [Приложение](#приложение)
- [Параметры](#параметры)
- [Contributing](#contributing)
- [To do](#to-do)
- [FAQ](#faq)
- [Источники](#источники)

## Технологии
- [Python](https://www.python.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Optuna](https://optuna.org/)
- [Matplotlib/Seaborn](https://matplotlib.org/) (для визуализации)
- [Skops](https://skops.readthedocs.io/) (для сохранения модели)

## Использование
Этот проект представляет собой Jupyter Notebook для анализа данных и обучения модели. Чтобы использовать модель для предсказаний, загрузите датасет и запустите ноутбук.

Установите необходимые библиотеки с помощью команды:
```sh
$ pip install pandas scikit-learn optuna matplotlib seaborn skops
```

Пример использования модели в коде (после обучения):
```python
import skops.io as sio
import pandas as pd

# Загрузка модели
unknown_types = sio.get_untrusted_types(file="model.skops")
model = sio.load("model.skops", trusted=unknown_types)

# Пример данных для предсказания
new_data = pd.DataFrame({
    'person_age': [30],
    'person_income': [60000],
    # ... (добавьте все признаки)
})

# Предсказание
prediction = model.predict(new_data)
print(f"Предсказание дефолта: {prediction[0]}")  # вероятность прогноза дефолта
```

## Разработка

### Требования
Для установки и запуска проекта, необходим [Python](https://www.python.org/) v3.8+ и [Jupyter Notebook](https://jupyter.org/).

### Установка зависимостей
Для установки зависимостей, выполните команду:
```sh
$ pip install -r requirements.txt
```

### Запуск Development сервера
Чтобы запустить Jupyter Notebook, выполните команду:
```sh
jupyter notebook creditRiskDataset.ipynb
```
Или используйте Google Colab для онлайн-запуска.


## Тестирование
Проект использует встроенные метрики Scikit-learn для оценки модели (accuracy, F1-score, ROC-AUC). Для запуска оценки модели выполните соответствующие ячейки в ноутбуке:
```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_test_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
```
Модель достигает ROC-AUC ~0.9485 на тестовых данных.

## Приложение
Модель сохраняется в файл `model.skops` для развертывания. 
Установка зависимостей: `pip install -r requirements.txt`
Приложение находится в директории CreditRiskApp и запускается через файл start_app.bat

## Параметры
model.skops - модель
lambdas.pkl - параметры трансформации
params/best_xgb_params.json - параметры лучшей модели 

## Contributing
Чтобы внести вклад, создайте issue с описанием бага или предложения. Для pull request: форкните репозиторий, создайте ветку, следуйте PEP8. Подробности в [Contributing.md](./CONTRIBUTING.md).

## FAQ 
### Зачем вы разработали этот проект?
Для демонстрации анализа кредитного риска с использованием ML, оптимизации модели и оценки рисков в финансах.

## To do
- [x] Добавить анализ данных и предобработку
- [x] Обучить модель с Optuna
- [ ] Добавить больше визуализаций (SHAP для интерпретации)
- [ ] Интегрировать с API для реального времени
- [ ] ...

## Источники
- Датасет: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle)
- Туториалы: Документация Scikit-learn, Optuna guides