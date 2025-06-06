#!/bin/bash
cd /app

# Docker Compose로 애플리케이션 시작
docker compose up -d

# 컨테이너가 정상적으로 시작되었는지 확인
sleep 10
if docker ps | grep -q flask-app; then
  echo "Flask 애플리케이션이 성공적으로 시작되었습니다."
else
  echo "Flask 애플리케이션 시작에 실패했습니다."
  docker compose logs
  exit 1
fi