#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工具调用转换网关
使用 FastAPI 将不支持 tool_call 的服务转换为支持 tool_calls 格式的响应
"""

import json
import requests
import uuid
import importlib.util
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Tool Call Gateway", description="将不支持tool_call的服务转换为tool_calls格式")

# 目标服务的URL（不支持tool_call的真实服务）
TARGET_API_URL = "http://127.0.0.1:5001/v1/chat/completions"



class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

# ===================== 核心模拟函数 =====================

def simulate_tools_call(
    api_url: str,
    authorization: str,
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    使用 requests 模拟 tools 调用，支持多轮工具调用循环。

    参数：
        api_url: API 端点地址，例如 "https://api.openai.com/v1/chat/completions"
        api_key: API 密钥
        model: 模型名称
        messages: 对话历史列表，格式同 OpenAI
        tools: 工具定义列表，格式同 OpenAI 的 tools 参数
        **kwargs: 其他传递给 API 的参数，如 temperature, max_tokens 等

    返回：
        最终模型的响应（字典格式），包含最终的助手消息。
    """
    # 1. 将 tools 定义转换成自然语言提示
    tools_description = "\n\n".join([
        f"工具 {i+1}：\n名称：{tool['function']['name']}\n描述：{tool['function']['description']}\n"
        f"参数 JSON Schema：{json.dumps(tool['function']['parameters'], ensure_ascii=False)}"
        for i, tool in enumerate(tools)
    ])

    system_prompt = (
        "\n你是一个可以调用外部工具的助手。当需要获取实时信息或执行特定操作时，请输出一个 JSON 对象表示要调用的工具。"
        "可用的工具如下：\n" + tools_description + "\n"
        "输出格式必须严格为：{\"tool\": \"工具名称\", \"arguments\": {参数对象}}。"
        "**重要提示：请判断用户的问题，是否需要调用工具，如果需要调用工具，聪明的选择对应的工具，并使用以上格式输出。**"
        "如果不需要调用工具，请直接给出普通回答。"
    )

    if len(tools) == 0:
        system_prompt = ''

    # 构建消息列表：确保 system prompt 存在
    new_messages = []
    system_exists = False
    for msg in messages:
        if msg["role"] == "system":
            # 合并到系统提示
            new_messages.append({"role": "system", "content": msg["content"] + system_prompt})
            system_exists = True
        else:
            new_messages.append(msg)
    if not system_exists:
        new_messages.insert(0, {"role": "system", "content": system_prompt})

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization
    }

    # 2. 调用模型
    payload = {
        "model": model,
        "messages": new_messages,
        **kwargs
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API 请求失败，状态码 {response.status_code}：{response.text}")

    result = response.json()

    return result

def parse_tool_call_from_content(content: str) -> Optional[Dict[str, Any]]:
    """
    从模型返回的content中解析工具调用
    simulate_tool_call.py期望的格式是: {"tool": "工具名称", "arguments": {参数对象}}
    """

    try:
        # 尝试直接解析JSON
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # 尝试从文本中提取JSON - 改进的正则表达式，可以处理嵌套的花括号
    import re

    # 首先尝试匹配整个JSON对象（包括嵌套的花括号）
    # 使用平衡括号匹配的正则表达式
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    json_objects = re.findall(json_pattern, content, re.DOTALL)

    for obj in json_objects:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    # 如果上面的方法失败，尝试用更简单的方法：在文本中查找可能包含tool的JSON
    # 查找包含 "tool" 的 JSON 对象
    import re
    # 查找包含 "tool" 的 JSON 对象
    tool_pattern = r'\{[^{}]*"tool"[^{}]*\}'
    match = re.search(tool_pattern, content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "tool" in parsed:
                # 检查是否有arguments，如果没有则使用空对象
                if "arguments" not in parsed:
                    parsed["arguments"] = {}
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return None

def convert_to_tool_calls_format(original_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    将原始响应转换为tool_calls格式
    """
    # 获取原始消息内容
    choice = original_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")

    # 生成工具调用ID
    tool_call_id = f"chatcmpl-tool-{uuid.uuid4().hex[:16]}"

    # 如果结尾包含 "FINISHED" 那么移除
    if content[-8:] == "FINISHED":
        content = content[:-8]

    # 尝试从内容中解析工具调用
    tool_call_data = parse_tool_call_from_content(content)

    if tool_call_data:
        # 构建tool_calls响应
        tool_calls = [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_call_data["tool"],
                    "arguments": json.dumps(tool_call_data["arguments"], ensure_ascii=False)
                }
            }
        ]

        # 提取think标签内容（如果存在）
        think_content = ""
        import re
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()

        # 构建新的消息内容
        if think_content:
            new_content = f"<think>\n{think_content}\n</think>\n\n"
        else:
            new_content = content

        # 构建响应
        response = {
            "id": original_response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": original_response.get("created", 1677652288),
            "model": original_response.get("model", "deepseek-chat"),
            "system_fingerprint": original_response.get("system_fingerprint", "fp_44709d6fcb"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": new_content,
                        "refusal": None,
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": tool_calls,
                        "reasoning": None,
                        "reasoning_content": None
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": original_response.get("usage", {
                "prompt_tokens": 100,
                "completion_tokens": 150,
                "total_tokens": 250
            })
        }
    else:
        # 没有工具调用，返回普通响应
        response = {
            "id": original_response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": original_response.get("created", 1677652288),
            "model": original_response.get("model", "deepseek-chat"),
            "system_fingerprint": original_response.get("system_fingerprint", "fp_44709d6fcb"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                        "reasoning": None,
                        "reasoning_content": None
                    },
                    "logprobs": None,
                    "finish_reason": choice.get("finish_reason", "stop")
                }
            ],
            "usage": original_response.get("usage", {
                "prompt_tokens": 100,
                "completion_tokens": 150,
                "total_tokens": 250
            })
        }

    return response

async def generate_streaming_response(original_response: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    生成流式响应
    """
    # 转换为tool_calls格式
    converted = convert_to_tool_calls_format(original_response)

    # 获取内容
    content = converted["choices"][0]["message"]["content"]
    tool_calls = converted["choices"][0]["message"].get("tool_calls")

    # 生成第一个chunk：角色信息
    first_chunk = {
        "id": converted["id"],
        "object": "chat.completion.chunk",
        "created": converted["created"],
        "model": converted["model"],
        "system_fingerprint": converted["system_fingerprint"],
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": ""
                },
                "logprobs": None,
                "finish_reason": None
            }
        ]
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    # 分块发送内容
    if content:
        # 按字符分块发送，模拟流式效果
        chunk_size = 5  # 每块5个字符
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i+chunk_size]
            chunk = {
                "id": converted["id"],
                "object": "chat.completion.chunk",
                "created": converted["created"],
                "model": converted["model"],
                "system_fingerprint": converted["system_fingerprint"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk_content
                        },
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # 如果有工具调用，发送工具调用信息
    if tool_calls:
        tool_chunk = {
            "id": converted["id"],
            "object": "chat.completion.chunk",
            "created": converted["created"],
            "model": converted["model"],
            "system_fingerprint": converted["system_fingerprint"],
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": tool_calls
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls"
                }
            ]
        }
        yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n"
    else:
        # 发送完成chunk
        final_chunk = {
            "id": converted["id"],
            "object": "chat.completion.chunk",
            "created": converted["created"],
            "model": converted["model"],
            "system_fingerprint": converted["system_fingerprint"],
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": converted["choices"][0].get("finish_reason", "stop")
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

    # 发送结束标记
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    处理聊天完成请求，调用真实服务并转换为tool_calls格式
    支持流式和非流式响应
    """
    try:
        auth_header = request.headers.get("Authorization", "")

        req_data = await request.json()
        stream = req_data.get("stream", False)

        # 准备消息列表
        messages = req_data.get("messages", [])
        model = req_data.get("model", "deepseek-chat")

        # 从请求中获取工具定义，如果没有则使用空列表
        request_tools = req_data.get("tools", [])

        # 调用真实的不支持tool_call的服务
        final_response = simulate_tools_call(
            api_url=TARGET_API_URL,
            authorization=auth_header,
            model=model,
            messages=messages,
            tools=request_tools,
            temperature=req_data.get("temperature", 0.8),
            max_tokens=req_data.get("max_tokens", 500),
        )

        if stream:
            # 返回流式响应
            return StreamingResponse(
                generate_streaming_response(final_response),
                media_type="text/event-stream"
            )
        else:
            # 返回非流式响应
            converted_response = convert_to_tool_calls_format(final_response)
            return JSONResponse(content=converted_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5002)
