## Проект SOC24
---
### Этот проект представляет собой интерактивный дашборд для визуализации модели Дегрута с использованием библиотек Plotly и Dash.

#### Создание окружения
Для запуска проекта необходимо создать виртуальное окружение Python версии 3.9 и установить зависимости.


```{bash}
git clone https://github.com/Mikkelando/SOC24.git
cd SOC24
```
Создание виртуального окружения
Выполните следующие команды в командной строке для создания и активации виртуального окружения:
```{bash}
python3.9 -m venv venv
source venv/bin/activate  # Для Unix / MacOS
venv\Scripts\activate      # Для Windows
```
Установка зависимостей
Установите необходимые зависимости из requirements.txt, используя следующую команду:

```{bash}
pip install -r requirements.txt
```
Запуск скрипта inference.py
Для запуска скрипта inference.py, который содержит код для генерации состояний модели и создания интерактивного дашборда, выполните следующую команду:

```{bash}
python inference.py
```
