import logging

# 하위 logger 설정; root logger에 영향 X
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(levelname)s: %(module)s > %(funcName)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False