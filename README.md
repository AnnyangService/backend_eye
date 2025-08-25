# Backend Eye

## 설치 방법

### 1. Docker 설치

- [Docker Desktop](https://www.docker.com/products/docker-desktop) 설치
- [Docker Compose](https://docs.docker.com/compose/install/) 설치

### 2. AI 모델 설정

#### 모델 파일 준비

Step1, 2 질병여부판단 모델을 다음 경로에 배치해야 합니다:

```
app/diagnosis/models
```

지원하는 모델 파일 형식:

- `step1.pth` (권장)
- `step1` (바이너리 파일)

#### 모델 요구사항

- **모델 타입**: EfficientNet-B2
- **클래스 수**: 2 (normal, abnormal)
- **입력 크기**: 224x224
- **전처리**: ImageNet 표준 정규화

#### 환경 변수 설정

모델 경로는 환경 변수로 설정할 수 있습니다:

```bash
export STEP1_MODEL_PATH=app/diagnosis/models/step1
```

또는 `docker-compose.yml`에서:

```yaml
environment:
  - STEP1_MODEL_PATH=app/diagnosis/models/step1
```

### 3. 프로젝트 실행 (Docker)

#### 개발 환경 (Development)

```bash
# 1. AI 모델 파일이 app/diagnosis/models/step1/ 경로에 있는지 확인

# 3. 개발 환경 설정 (docker-compose.yml 파일에 이미 설정되어 있음)
# FLASK_ENV=development

# 4. Flask 앱 도커 이미지 빌드 및 컨테이너 실행
docker-compose up --build

# 5. 백그라운드에서 실행
docker-compose up -d

# 6. 로그 확인
docker-compose logs -f
```

#### 배포 환경 (Production)

```bash
# 1. AI 모델 파일이 app/diagnosis/models/step1/ 경로에 있는지 확인

# 3. 배포 환경으로 설정 변경
# docker-compose.yml 파일의 environment 섹션에서 수정:
# - "FLASK_ENV=production"

# 4. Flask 앱 도커 이미지 빌드 및 컨테이너 실행
docker-compose up --build

# 5. 백그라운드에서 실행
docker-compose up -d

# 6. 로그 확인
docker-compose logs -f
```

### 4. 환경 설정

#### 환경 구분

- **개발 환경 (Development)**

  ```bash
  export FLASK_ENV=development
  # 또는
  export FLASK_APP="app:create_app('development')"
  ```

- **배포 환경 (Production)**
  ```bash
  export FLASK_ENV=production
  # 또는
  export FLASK_APP="app:create_app('production')"
  ```

#### Docker 환경 설정

`docker-compose.yml`에서 환경 변수 설정:

```yaml
services:
  flask-app:
    environment:
      - FLASK_ENV=development # 또는 production
      - STEP1_MODEL_PATH=app/diagnosis/models/step1
      - MAX_IMAGE_SIZE=4096
      - MIN_IMAGE_SIZE=100
      - IMAGE_DOWNLOAD_TIMEOUT=30
```

#### EC2 환경 설정

1. **환경 변수로 설정**

   ```bash
   # 환경 변수 설정
   export FLASK_ENV=production
   export FLASK_APP="app:create_app('production')"
   export STEP1_MODEL_PATH=app/diagnosis/models/step1

   # 앱 실행
   flask run
   ```

2. **직접 config 지정**

   ```bash
   flask --app "app:create_app('production')" run
   ```

3. **systemd 서비스로 실행**
   `/etc/systemd/system/flask-app.service`:

   ```ini
   [Unit]
   Description=Flask Application
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/path/to/your/app
   Environment="FLASK_ENV=production"
   Environment="FLASK_APP=app:create_app('production')"
   Environment="STEP1_MODEL_PATH=app/diagnosis/models/step1"
   ExecStart=/usr/local/bin/flask run --host=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   서비스 시작:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start flask-app
   sudo systemctl enable flask-app
   ```

## AI 모델 정보

### Step1 질병여부판단 모델

- **목적**: 이미지에서 질병 여부를 판단
- **입력**: 이미지 URL
- **출력**: 정상/이상 여부 + 신뢰도

#### 모델 아키텍처

- **베이스 모델**: EfficientNet-B2
- **분류 클래스**: 2개 (normal, abnormal)
- **입력 해상도**: 224x224 RGB
- **전처리**: ImageNet 표준 정규화

#### 성능 요구사항

- **GPU**: CUDA 지원 GPU (권장)
- **메모리**: 최소 2GB VRAM
- **CPU**: 멀티코어 프로세서 (GPU 없는 경우)

#### 에러 처리

AI 모델 로드에 실패한 경우:

- 서버 시작 시 모델 로드 실패 로그 출력
- API 호출 시 503 Service Unavailable 에러 반환
- 명확한 에러 메시지 제공 (서버 관리자 문의 안내)

## Swagger API 문서

### 접속 방법

애플리케이션 실행 후 다음 URL에서 Swagger UI에 접속할 수 있습니다:

- **개발 환경**: http://localhost:5000/docs/
- **배포 환경**: http://your-domain.com/docs/

### 주요 기능

- **자동화된 API 문서**: 각 엔드포인트를 직접 테스트할 수 있습니다
- **요청/응답 스키마**: 모든 API의 입력과 출력 형식을 확인할 수 있습니다
- **에러 코드 문서화**: 각 API에서 발생할 수 있는 에러 코드와 메시지를 확인할 수 있습니다

### API 엔드포인트

#### Diagnosis API

- **POST /api/diagnosis/step1/**: 질병여부판단
  - 요청 파라미터:
    - `image_url` (string, required): 분석할 이미지의 URL
  - 응답:
    ```json
    {
      "success": true,
      "message": "Success",
      "data": {
        "is_normal": false,
        "confidence": 0.9
      }
    }
    ```

#### 이미지 요구사항

- **지원 형식**: JPEG, PNG, GIF, BMP
- **최소 크기**: 100x100 픽셀
- **최대 크기**: 4096x4096 픽셀
- **다운로드 제한시간**: 30초

## CLI 명령어

### 데이터베이스 관리

#### 데이터베이스 초기화

```bash
# 데이터베이스 테이블 생성
flask init-db

# 데이터베이스 테이블 삭제
flask drop-db

# 데이터베이스 초기화 (삭제 후 재생성)
flask reset-db
```

#### 청크 데이터 저장

```bash
# 청크 데이터와 임베딩을 PostgreSQL에 저장
flask load-chunks

# 임베딩 없이 청크 데이터만 저장
flask load-chunks-without-embeddings
```

### 청크 데이터 구조

저장되는 청크 데이터는 다음 구조를 가집니다:

```json
{
  "id": "각막궤양.정의",
  "content": "각막의 표면층인 상피가 손상되고...",
  "keywords": ["건성안", "외상", "만성화"]
}
```

### 임베딩 파일

- **경로**: `app/rag/embeddings/`
- **형식**: `.npy` 파일 (NumPy 배열)
- **차원**: 384차원 벡터
- **모델**: km-bert 기반 임베딩

## Docker 설정

### 주요 설정

- **Flask 앱**
  - 포트: 5000
  - 개발 모드: 활성화
  - 코드 변경: 실시간 반영 (볼륨 마운트)
  - Python 경로: /app
  - AI 모델: GPU 지원 (CUDA 사용 가능 시)

### Docker 명령어

```bash
# 1. 컨테이너 중지
docker-compose down

# 2. 컨테이너 재시작
docker-compose restart

# 3. 특정 서비스만 재시작
docker-compose restart flask-app

# 4. 컨테이너 내부 접속
docker-compose exec flask-app bash

# 5. AI 모델 상태 확인
docker-compose logs flask-app | grep -i "model"
```

## API 응답 형식

성공 시:

```json
{
  "success": true,
  "message": "Success",
  "data": {
    "is_normal": false,
    "confidence": 0.9
  }
}
```

에러 발생 시:

```json
{
  "success": false,
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid image URL provided",
  "details": {
    "image_url": "Please provide a valid image URL"
  }
}
```

AI 모델 서비스 불가 시 (503):

```json
{
  "success": false,
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "AI 모델 서비스를 사용할 수 없습니다. 서버 관리자에게 문의하세요.",
  "details": {
    "service": "AI model not loaded"
  }
}
```

## 트러블슈팅

### AI 모델 관련

1. **모델 로드 실패**

   ```bash
   # 모델 파일 존재 확인
   ls -la app/diagnosis/models/step1/

   # 로그 확인
   docker-compose logs flask-app | grep -i "model"
   ```

2. **GPU 사용 불가**

   ```bash
   # CUDA 사용 가능 여부 확인
   docker-compose exec flask-app python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **메모리 부족**
   - Docker 메모리 제한 증가
   - 이미지 크기 제한 조정 (MAX_IMAGE_SIZE)

### 이미지 처리 관련

1. **이미지 다운로드 실패**

   - 네트워크 연결 확인
   - 이미지 URL 유효성 확인
   - 타임아웃 설정 조정 (IMAGE_DOWNLOAD_TIMEOUT)

2. **이미지 크기 제한**
   - MIN_IMAGE_SIZE, MAX_IMAGE_SIZE 환경 변수 조정

## 주의사항

- 코드 변경은 실시간으로 반영됨
- 환경변수는 `docker-compose.yml`에서 관리
- AI 모델 파일은 Git에 포함되지 않으므로 별도로 배치 필요
- GPU 사용 시 CUDA 드라이버와 Docker GPU 지원 설정 필요
