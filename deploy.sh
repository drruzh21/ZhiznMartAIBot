#!/bin/bash

# Обновление кода из репозитория
echo "Pulling latest code from GitHub..."
git pull origin main

# Остановка текущих контейнеров
echo "Stopping Docker containers..."
docker-compose down

# Перезапуск контейнеров и сборка образов
echo "Starting Docker containers..."
docker-compose up -d --build
