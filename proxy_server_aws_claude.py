#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWS Bedrock (Claude) only proxy server.
Exposes an OpenAI-compatible endpoint: POST /v1/chat/completions

- Uses boto3 bedrock-runtime Converse / ConverseStream
- Supports: system/user/assistant messages, temperature, max_tokens, stream (SSE)
- Keeps /health and /v1/models

Run:
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_DEFAULT_REGION=us-west-2
  # optional: CONFIG_PATH=config.json, PORT=8008
  python proxy_server_aws_claude.py
"""
import os
import json
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ------------------------- logging -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("proxy_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ------------------------- pydantic -------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 8192
    stream: bool = False
    routing: Optional[Dict[str, Any]] = None  # reserved


# ------------------------- app -------------------------
app = FastAPI(title="AWS Bedrock Claude Proxy (OpenAI compatible)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------- config -------------------------
clients: Dict[str, Any] = {}

# NOTE:
# - Prefer IAM role / env vars. If you *must* hardcode keys, put them in config.json,
#   and set CONFIG_PATH to point to it.
config: Dict[str, Any] = {
    "routes": {
        "bedrock": {
            "type": "bedrock",
            # If omitted, boto3 will use standard credential chain (env/role/profile).
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "region_name": os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2",
            "default_model": "sonnet",
            "model_map": {
                # aliases -> Bedrock modelId (can be ARN inference profile or model string id)
                "opus": "arn:aws:bedrock:us-west-2:767398088105:inference-profile/us.anthropic.claude-opus-4-20250514-v1:0",
                "sonnet": "arn:aws:bedrock:us-west-2:767398088105:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
                "sonnet37": "arn:aws:bedrock:us-west-2:767398088105:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            },
            # Extra configs are optional; keep your existing defaults.
            "inference_config": {},
            "additional_model_request_fields": {
                "reasoning_config": {"type": "disabled"}
            },
            "performance_config": {"latency": "standard"},
        }
    }
}


def load_config() -> None:
    """Optionally load/override config from CONFIG_PATH (default: config.json)."""
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # shallow merge
            if isinstance(loaded, dict):
                config.update(loaded)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")


def init_clients() -> None:
    route_config = config["routes"]["bedrock"]
    kwargs = {"region_name": route_config.get("region_name")}

    # If keys are provided explicitly, pass them; otherwise boto3 uses default chain.
    ak = route_config.get("aws_access_key_id")
    sk = route_config.get("aws_secret_access_key")
    if ak and sk:
        kwargs.update({"aws_access_key_id": ak, "aws_secret_access_key": sk})

    clients["bedrock"] = boto3.client("bedrock-runtime", **kwargs)
    logger.info("Initialized bedrock-runtime client")


@app.on_event("startup")
async def startup_event():
    load_config()
    init_clients()
    logger.info("Proxy server started (AWS Bedrock Claude only)")


# ------------------------- helpers -------------------------
def _epoch() -> int:
    return int(time.time())


def _json(obj: Any) -> str:
    # keep Chinese readable, etc.
    return json.dumps(obj, ensure_ascii=False)


def is_client_disconnect_error(e: Exception) -> bool:
    """Best-effort: detect client disconnect during streaming."""
    name = type(e).__name__.lower()
    s = str(e).lower()
    indicators = [
        "connectionreseterror",
        "brokenpipeerror",
        "connectionabortederror",
        "client disconnected",
        "connection reset",
        "broken pipe",
        "socket.send()",
    ]
    return any(i in name or i in s for i in indicators)


def determine_model_id(model_alias: str, route_config: Dict[str, Any]) -> str:
    mm = route_config.get("model_map", {})
    if model_alias in mm:
        return mm[model_alias]
    # allow direct pass-through modelId/ARN
    return model_alias


def openai_to_bedrock_converse_messages(messages: List[Message]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """OpenAI messages -> Bedrock Converse (messages, system)."""
    br_messages: List[Dict[str, Any]] = []
    system: List[Dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system.append({"text": m.content})
        elif m.role in ("user", "assistant"):
            br_messages.append({"role": m.role, "content": [{"text": m.content}]})
        else:
            # ignore tool/function roles for this minimal Claude-only proxy
            continue
    return br_messages, system


# ------------------------- endpoints -------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    route = config["routes"]["bedrock"]
    models = []
    for alias in route.get("model_map", {}).keys():
        models.append({"id": alias, "object": "model", "created": 0, "owned_by": "bedrock"})
    return {"object": "list", "data": models}


async def handle_bedrock_request(
    client: Any,
    route_config: Dict[str, Any],
    model_id: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> Union[Dict[str, Any], StreamingResponse]:
    # message conversion
    br_messages, system = openai_to_bedrock_converse_messages(messages)

    request_params: Dict[str, Any] = {
        "modelId": model_id,
        "messages": br_messages,
        "inferenceConfig": {
            "temperature": float(temperature),
            "maxTokens": int(max_tokens),
            **route_config.get("inference_config", {}),
        },
    }
    if system:
        request_params["system"] = system
    if route_config.get("additional_model_request_fields"):
        request_params["additionalModelRequestFields"] = route_config["additional_model_request_fields"]
    if route_config.get("performance_config"):
        request_params["performanceConfig"] = route_config["performance_config"]

    if stream:
        async def generate_stream():
            bedrock_stream = None
            completion_id = f"chatcmpl-{_epoch()}-{os.urandom(4).hex()}"
            created = _epoch()

            # initial chunk (role)
            init_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            try:
                yield f"data: {_json(init_chunk)}\n\n"
            except Exception as e:
                if is_client_disconnect_error(e):
                    logger.info("Client disconnected before Bedrock streaming started")
                    return
                raise

            try:
                response = client.converse_stream(**request_params)
                bedrock_stream = response.get("stream")
                if bedrock_stream is None:
                    raise RuntimeError("Bedrock converse_stream response missing 'stream'")

                for event in bedrock_stream:
                    # text deltas
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        txt = delta.get("text") or ""
                        if txt:
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": txt}, "finish_reason": None}],
                            }
                            try:
                                yield f"data: {_json(chunk)}\n\n"
                            except Exception as e:
                                if is_client_disconnect_error(e):
                                    logger.info("Client disconnected during Bedrock streaming")
                                    return
                                logger.error(f"Error yielding Bedrock chunk: {e}")
                                return

                    # stop
                    if "messageStop" in event:
                        break

                # final chunk + DONE
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                try:
                    yield f"data: {_json(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    if is_client_disconnect_error(e):
                        logger.info("Client disconnected while sending DONE")
                    return

            except asyncio.CancelledError:
                logger.info("Bedrock streaming task cancelled")
                raise
            except (BotoCoreError, ClientError) as e:
                logger.error(f"Bedrock stream error: {e}")
                try:
                    yield f"data: {_json({'error': {'message': str(e), 'type': 'provider_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception:
                    return
            except Exception as e:
                logger.error(f"Bedrock stream error: {e}")
                try:
                    yield f"data: {_json({'error': {'message': str(e), 'type': 'internal_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception:
                    return
            finally:
                if bedrock_stream is not None:
                    logger.debug("Bedrock stream completed")

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # non-stream
    try:
        resp = client.converse(**request_params)
        content = ""
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]

        usage_raw = resp.get("usage", {}) or {}
        usage = {
            "prompt_tokens": usage_raw.get("inputTokens", -1),
            "completion_tokens": usage_raw.get("outputTokens", -1),
            "total_tokens": usage_raw.get("totalTokens", -1),
        }

        created = _epoch()
        return {
            "id": f"chatcmpl-{created}-{os.urandom(4).hex()}",
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": usage,
        }

    except (BotoCoreError, ClientError) as e:
        logger.error(f"Bedrock request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Bedrock request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    route_config = config["routes"]["bedrock"]
    # allow routing override: {"routing": {"model": "..."} } or {"routing":{"route":"bedrock","model":"..."}}
    model_alias = req.model or route_config.get("default_model", "sonnet")
    if req.routing and isinstance(req.routing, dict):
        model_alias = req.routing.get("model") or model_alias

    model_id = determine_model_id(model_alias, route_config)
    logger.info(f"Bedrock route: alias='{model_alias}' modelId='{model_id}' stream={req.stream}")

    result = await handle_bedrock_request(
        clients["bedrock"],
        route_config,
        model_id,
        req.messages,
        req.temperature,
        req.max_tokens,
        req.stream,
    )

    # StreamingResponse is returned directly; dict is JSONResponse
    if isinstance(result, StreamingResponse):
        return result
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8008))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False,
    )
