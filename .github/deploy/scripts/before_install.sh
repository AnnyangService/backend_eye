#!/bin/bash

# 애플리케이션 디렉토리 생성
if [ ! -d /app ]; then
  mkdir -p /app
fi

# 기존 Docker Compose 서비스 중지 및 제거
if [ -f /app/scripts/docker-compose.yml ]; then
  echo "기존 Docker Compose 서비스 중지 중..."
  cd /app/scripts
  docker-compose down --rmi all --volumes --remove-orphans || true
fi

# 개별 컨테이너도 확인해서 중지 (혹시 모를 상황 대비)
docker stop flask-app postgres || true
docker rm flask-app postgres || true

echo "기존 서비스 정리 완료"