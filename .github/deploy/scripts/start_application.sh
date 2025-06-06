#!/bin/bash
cd /app

# Docker 컨테이너 직접 실행
docker run -d \
  --name flask-app \
  --restart unless-stopped \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  -e FLASK_CONFIG=production \
  -v /app/instance:/app/instance \
  -v /app/rule_embeddings:/app/rule_embeddings \
  my-app-image

# 컨테이너가 정상적으로 시작되었는지 확인
sleep 10
if docker ps | grep -q flask-app; then
  echo "Flask 애플리케이션이 성공적으로 시작되었습니다."
else
  echo "Flask 애플리케이션 시작에 실패했습니다."
  docker compose logs
  exit 1
fi