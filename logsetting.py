import logging

# 1) ANSI 색상 코드 매핑
LEVEL_COLORS = {
    'DEBUG':    '\033[0;36m',  # 청록 (cyan)
    'INFO':     '\033[0;32m',  # 녹색 (green)
    'WARNING':  '\033[0;33m',  # 노랑 (yellow)
    'ERROR':    '\033[0;31m',  # 빨강 (red)
    'CRITICAL': '\033[1;35m',  # 진한 자홍 (magenta)
}
RESET_COLOR = '\033[0m'       # 리셋

# 2) Formatter 서브클래싱
class ColorFormatter(logging.Formatter):
    def format(self, record):
        # 레벨 이름에 색상 코드 삽입
        color = LEVEL_COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{RESET_COLOR}"
        return super().format(record)

# 3) 로거 설정
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
# 기존 포맷에 색 적용 Formatter 사용
handler.setFormatter(ColorFormatter('[%(levelname)s: %(module)s > %(funcName)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False