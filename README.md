# Backend Eye

## 설치 방법

### 1. Docker 설치

- [Docker Desktop](https://www.docker.com/products/docker-desktop) 설치
- [Docker Compose](https://docs.docker.com/compose/install/) 설치

### 2. 프로젝트 실행 (Docker)

```bash
# 1. 도커 이미지 빌드 및 컨테이너 실행
docker-compose up --build

# 2. 백그라운드에서 실행
docker-compose up -d

# 3. 로그 확인
docker-compose logs -f

# 4. 특정 서비스 로그만 확인
docker-compose logs -f flask-app
docker-compose logs -f mysql
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

### 4. 데이터베이스 초기화

처음 한 번만 실행:

```bash
flask db-init
```

### 5. 모델 변경 후 마이그레이션

모델을 수정한 후에는 다음 명령어를 실행:

```bash
flask db-migrate
flask db-upgrade
```

### 6. 데이터베이스 초기화 (모든 데이터 삭제)

주의: 이 명령어는 모든 데이터를 삭제합니다!

```bash
flask init-db
```

## Docker 설정

### 주요 설정

- **MySQL**

  - 포트: 3306
  - 데이터베이스: hi_meow
  - 사용자: admin
  - 비밀번호: 1234
  - 데이터 영속성: mysql_data 볼륨 사용

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
docker-compose exec mysql bash

# 5. 볼륨 삭제 (데이터 초기화)
docker-compose down -v
```

## 데이터베이스 마이그레이션

### 환경별 마이그레이션 설정

- **개발 환경 (Development)**

  - 마이그레이션 활성화
  - `flask db migrate`, `flask db upgrade` 명령어 사용 가능
  - 테이블 생성/수정 가능

- **배포 환경 (Production)**
  - 마이그레이션 비활성화
  - Spring 서버가 데이터베이스 스키마 관리
  - Flask 서버는 읽기 전용으로 동작

### 초기 설정

처음 한 번만 실행:

```bash
flask db init
```

### 모델 변경 시

1. `models.py` 파일에서 모델 수정
2. 변경사항 감지:

```bash
flask db migrate -m "변경 내용 설명"
```

3. 변경사항 적용:

```bash
flask db upgrade
```

### 롤백이 필요한 경우

이전 버전으로 되돌리기:

```bash
flask db downgrade
```

### 주의사항

- 개발 환경에서만 마이그레이션 명령어 사용
- 배포 환경에서는 Spring 서버가 데이터베이스 스키마 관리
- 두 서버가 동시에 마이그레이션을 시도하면 데이터베이스 상태가 꼬일 수 있음

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

- 데이터베이스 데이터는 `mysql_data` 볼륨에 저장됨
- 코드 변경은 실시간으로 반영됨
- 환경변수는 `docker-compose.yml`에서 관리
- 배포 환경에서는 Spring 서버가 데이터베이스 스키마를 관리하므로 Flask 서버의 마이그레이션은 비활성화됨
