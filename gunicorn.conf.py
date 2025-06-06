import os

# t3.small에 맞는 보수적 설정
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# 워커 수: CPU 코어 수와 동일하게 (2개)
# t3.small은 버스터블이므로 과도한 워커는 성능 저하
workers = int(os.environ.get('GUNICORN_WORKERS', '1'))

# 메모리 사용량 고려한 설정
worker_class = "sync"  # async보다 메모리 효율적
worker_connections = 1000
timeout = 120  # AI 처리 시간 고려하여 늘림
keepalive = 2

# 메모리 누수 방지 (중요!)
max_requests = 100  # 요청 100개마다 워커 재시작
max_requests_jitter = 20

# 메모리 절약을 위한 설정
preload_app = False
lazy_apps = True

# 로깅 (메모리 사용량 모니터링용)
accesslog = "-"
errorlog = "-" 
loglevel = "info"

# 프로세스 이름
proc_name = "flask-eye-backend"