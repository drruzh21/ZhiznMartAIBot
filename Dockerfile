# Используем официальный образ Python 3.11
FROM python:3.11-slim

# Обновляем пакеты и устанавливаем ffmpeg
RUN apt update && apt install -y ffmpeg

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл requirements.txt и устанавливаем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в контейнер
COPY . .

# Устанавливаем переменные окружения из .env с помощью python-dotenv
# Пример установки на уровне Python: это будет работать через main.py
ENV PYTHONUNBUFFERED=1

# Команда для запуска бота
CMD ["python", "main.py"]
