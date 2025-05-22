# Backend Eye

## 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install flask-sqlalchemy flask-migrate mysqlclient
```

### 2. MySQL 실행 (Docker)

```bash
docker run --name mysql -e MYSQL_ROOT_PASSWORD=password -p 3306:3306 -d mysql:8.0
```

### 3. 데이터베이스 초기화

처음 한 번만 실행:

```bash
flask db-init
```

### 4. 모델 변경 후 마이그레이션

모델을 수정한 후에는 다음 명령어를 실행:

```bash
flask db-migrate
flask db-upgrade
```

### 5. 데이터베이스 초기화 (모든 데이터 삭제)

주의: 이 명령어는 모든 데이터를 삭제합니다!

```bash
flask init-db
```

## 데이터베이스 마이그레이션

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

## 프로젝트 구조

```
app/
├── common/           # 공통 모듈
│   ├── database/    # 데이터베이스 관련
│   ├── response/    # API 응답 관련
│   └── util/        # 유틸리티 함수
├── diagnosis/       # 진단 관련 모듈
│   ├── models.py    # 데이터베이스 모델
│   ├── routes.py    # API 라우트
│   ├── schemas.py   # 요청/응답 스키마
│   └── service.py   # 비즈니스 로직
└── __init__.py      # 앱 초기화
```

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
