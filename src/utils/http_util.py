import base64
import hashlib
import hmac
import json
import time
import uuid

import numpy as np
import requests

from src.configs.config import config
from src.utils.crypt_util import decrypt_secret
from src.logger.logger_adapter import logger
from src.utils.async_http_client import get_http_client


async def request_speaker_omni(
        session_id: str,
        data: np.ndarray,
        asr: str,
        vad_start: str,
        semaphore: asyncio.Semaphore = None
) -> tuple[str, str]:
    """异步版本的 Speaker Omni 请求"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    wave_b64 = base64.b64encode(data.tobytes()).decode('utf-8')
    payload = {
        "data": wave_b64,
        "asr": asr,
        "vad_start": vad_start,
        "session_id": session_id,
    }
    request_data, headers = build_mep_request(payload, config.speaker_omni_bid, config.speaker_omni_flowId)
    result = await common_api_call_async(request_id, config.omni_address, headers, request_data, 5.0, semaphore)
    response_data = result.get("src", {}) if result else {}
    return response_data




async def request_qwen3_asr(
        session_id: str,
        data: np.ndarray,
        prev : str,
        enable_fa: bool = False,
        semaphore: asyncio.Semaphore = None
) -> tuple[str, str]:
    """异步版本的 Qwen3 ASR 请求"""
    request_id = session_id
    wave_b64 = base64.b64encode(data.tobytes()).decode('utf-8')
    payload = {
        "data": wave_b64,
        "session_id": session_id,
        "prev_src": prev,
    }

    if enable_fa:
        payload["enable_fa"] = "true"
    request_data, headers = build_mep_request(payload, config.qwen3_asr_bid, config.qwen3_asr_flowId)
    result = await common_api_call_async(request_id, config.omni_address, headers, request_data, 5.0, semaphore)
    response_data = result.get("src", {}) if result else {}
    return response_data


def common_api_call(
        request_id: str,
        url: str,
        headers: dict,
        data: dict,
        timeout: int
) -> dict:
    """
    通用API调用函数（同步版本，保留用于兼容性）
    """
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        result = response.json()
        if result['result'] and result['result']['code'] == '0':
            return result['result']['content'][0]
        else:
            return {}
    except Exception as e:
        return {}


async def common_api_call_async(
        request_id: str,
        url: str,
        headers: dict,
        data: dict,
        timeout: float,
        semaphore: asyncio.Semaphore = None
) -> dict:
    """
    通用API调用函数（异步版本）
    """
    try:
        client = await get_http_client()
        result = await client.post(
            url=url,
            headers=headers,
            data=data,
            timeout=timeout,
            retry_times=2,
            retry_delay=0.1,
            semaphore=semaphore,
        )
        return result if result else {}
    except Exception as e:
        logger.error(f"API call failed for request_id={request_id}: {e}")
        return {}


def build_mep_request(payload, bId, flowId) -> tuple[dict, dict]:
    app_id = config.mep_app_id
    sign_key = decrypt_secret(config.mep_sign_key)

    data = {
        'data': payload,
        "meta": {
            "bId": f"{bId}",
            "flowId": f"{flowId}"
        },
        "version": "1.0"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": calc_mep_sign(json.dumps(data), sign_key, app_id)
    }
    return data, headers


def calc_mep_sign(data, sign_key, app_id):
    payload = ""
    if data is not None and data != "" and len(data) > 0:
        payload += data
    query_str = ""
    path_str = ""
    query_str = query_str[:-1]
    for v in ["service"]:
        path_str += "/{}".format(v)
    timestamp = int(time.time() * 1000)
    sign_str = "{}&{}&{}&{}&appid={}&timestamp={}".format(
        "POST", path_str, query_str, data, app_id, timestamp)
    sign_key = sign_key.encode('utf-8')
    sign_str = sign_str.encode('utf-8')
    sign = base64.b64encode(hmac.new(
        sign_key, sign_str, digestmod=hashlib.sha256).digest()).decode("utf-8")
    return 'CLOUDSOA-HMAC-SHA256 appid={}, timestamp={}, signature="{}"'.format(app_id, timestamp, sign)

