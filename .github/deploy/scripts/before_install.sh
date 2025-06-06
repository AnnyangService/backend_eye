#!/bin/bash

# 애플리케이션 디렉토리 생성
if [ ! -d /app ]; then
  mkdir -p /app
fi

# 기존 컨테이너 중지 및 제거
docker stop flask-app || true
docker rm flask-app || true

# 기존 이미지 제거
docker rmi my-app-image || true