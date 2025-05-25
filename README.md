# Backend Eye

## 설치 방법

### 1. Docker 설치

- [Docker Desktop](https://www.docker.com/products/docker-desktop) 설치
- [Docker Compose](https://docs.docker.com/compose/install/) 설치

### 2. 프로젝트 실행 (Docker)

#### 개발 환경 (Development)

```bash
# 1. Spring 앱의 MySQL 데이터베이스가 먼저 실행되어야 합니다

# 2. 개발 환경 설정 (docker-compose.yml 파일에 이미 설정되어 있음)
# FLASK_ENV=development

# 3. Flask 앱 도커 이미지 빌드 및 컨테이너 실행
docker-compose up --build

# 4. 백그라운드에서 실행
docker-compose up -d

# 5. 로그 확인
docker-compose logs -f
```

#### 배포 환경 (Production)

```bash
# 1. Spring 앱의 MySQL 데이터베이스가 먼저 실행되어야 합니다

# 2. 배포 환경으로 설정 변경
# docker-compose.yml 파일의 environment 섹션에서 수정:
# - "FLASK_ENV=production"

# 3. Flask 앱 도커 이미지 빌드 및 컨테이너 실행
docker-compose up --build

# 4. 백그라운드에서 실행
docker-compose up -d

# 5. 로그 확인
docker-compose logs -f
```

### 3. 환경 설정

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
```

#### EC2 환경 설정

1. **환경 변수로 설정**

   ```bash
   # 환경 변수 설정
   export FLASK_ENV=production
   export FLASK_APP="app:create_app('production')"

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

### 4. 데이터베이스 정보 확인

데이터베이스 연결 정보 확인:

```bash
flask db-info
```

## Swagger API 문서

### 접속 방법

애플리케이션 실행 후 다음 URL에서 Swagger UI에 접속할 수 있습니다:

- **개발 환경**: http://localhost:5000/api/docs/
- **배포 환경**: http://your-domain.com/api/docs/

### 주요 기능

- **자동화된 API 문서**: 각 엔드포인트를 직접 테스트할 수 있습니다
- **요청/응답 스키마**: 모든 API의 입력과 출력 형식을 확인할 수 있습니다
- **에러 코드 문서화**: 각 API에서 발생할 수 있는 에러 코드와 메시지를 확인할 수 있습니다

### API 엔드포인트

#### Diagnosis API

- **POST /api/diagnosis/v1/diagnosis**: 이미지 진단 수행
  - 요청 파라미터:
    - `image_url` (string, required): 진단할 이미지의 URL
    - `cat_id` (string, required): 진단 카테고리 ID
  - 응답: 진단 결과 데이터

## Docker 설정

### 주요 설정

- **데이터베이스 연결**

  - Spring 앱에서 관리하는 MySQL 데이터베이스에 연결
  - 포트: 3306
  - 데이터베이스: hi_meow
  - 사용자: admin
  - 비밀번호: 1234

- **Flask 앱**
  - 포트: 5000
  - 개발 모드: 활성화
  - 코드 변경: 실시간 반영 (볼륨 마운트)
  - Python 경로: /app

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
```

## 데이터베이스 마이그레이션

### 환경별 데이터베이스 관리

- **개발 환경 (Development)**

  - API 서버(Spring)가 데이터베이스 스키마 관리
  - 개발 환경에서는 Spring 앱이 실행하는 MySQL에 접속

- **배포 환경 (Production)**
  - API 서버(Spring)가 데이터베이스 스키마 관리
  - 배포 환경에서는 배포된 데이터베이스에 접속

### 주의사항

- 데이터베이스 스키마와 마이그레이션은 Spring API 서버에서 전적으로 관리
- Flask 서버는 데이터베이스 읽기/쓰기만 수행하고 스키마 변경은 하지 않음
- 스키마 변경이 필요한 경우 API 서버 관리자에게 요청

## API 응답 형식

```json
{
  "success": true,
  "data": {
    // 실제 데이터
  },
  "error": null
}
```

에러 발생 시:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "에러 메시지",
    "details": {
      // 상세 에러 정보
    }
  }
}
```

## 주의사항

- 코드 변경은 실시간으로 반영됨
- 환경변수는 `docker-compose.yml`에서 관리
- 데이터베이스 스키마와 마이그레이션은 API 서버(Spring)에서 관리함
- 개발 환경에서는 Spring 앱이 실행하는 MySQL에 접속하고, 배포 환경에서는 배포된 데이터베이스에 접속
