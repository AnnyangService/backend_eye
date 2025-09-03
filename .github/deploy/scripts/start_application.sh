#!/bin/bash
cd /app

echo "Docker Compose로 애플리케이션 시작 중..."

# scripts 폴더 내의 docker-compose.yml 사용
cd /app/scripts

# Docker Compose로 서비스 시작
docker-compose up -d

# 서비스가 정상적으로 시작되었는지 확인
sleep 15

# PostgreSQL 컨테이너 상태 확인
if docker-compose ps postgres | grep -q "Up"; then
  echo "PostgreSQL이 성공적으로 시작되었습니다."
else
  echo "PostgreSQL 시작에 실패했습니다."
  docker-compose logs postgres
  exit 1
fi

# Flask 앱 컨테이너 상태 확인
if docker-compose ps flask-app | grep -q "Up"; then
  echo "Flask 애플리케이션이 성공적으로 시작되었습니다."
else
  echo "Flask 애플리케이션 시작에 실패했습니다."
  docker-compose logs flask-app
  exit 1
fi

echo "모든 서비스가 성공적으로 시작되었습니다."