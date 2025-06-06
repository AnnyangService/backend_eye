#!/bin/bash
cd /app

# Docker 이미지 압축 해제 및 로드
gunzip -f my-app-image.tar.gz
docker load -i my-app-image.tar

# 압축 해제된 tar 파일 정리
rm -f my-app-image.tar
