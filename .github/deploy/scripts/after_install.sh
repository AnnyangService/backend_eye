#!/bin/bash
cd /app

# Docker 이미지 압축 해제 및 로드
gunzip -f my-app-image.tar.gz
docker load -i my-app-image.tar

# 압축 해제된 tar 파일 정리
rm -f my-app-image.tar

# Docker Compose 파일 생성 (또는 기존 파일 사용)
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  flask-app:
    image: my-app-image
    container_name: flask-app
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - FLASK_CONFIG=production
    restart: unless-stopped
    volumes:
      - ./instance:/app/instance
      - ./rule_embeddings:/app/rule_embeddings
EOF

# 필요한 디렉토리 생성
mkdir -p instance/images rule_embeddings