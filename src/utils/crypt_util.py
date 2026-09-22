from common.security.cryptor import Crypt

from src.logger.logger_adapter import logger


def decrypt_secret(password: str) -> str:
    if password is None or password == '':
        logger.error(f'decrypt_secret password is empty')
        return ''
    crypt = Crypt()
    password = crypt.decrypt("configEncryptKey", password.encode("utf-8"))
    return password


def encrypt_secret(origin: str) -> str:
    if origin is None or origin == '':
        logger.error(f'encrypt_secret password is empty')
        return ''
    crypt = Crypt()
    pwd = crypt.encrypt(appId="configEncryptKey", origin=origin)
    return pwd.decode('utf-8')
