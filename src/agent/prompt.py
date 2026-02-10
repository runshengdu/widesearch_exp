# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from datetime import datetime

default_system_prompt_zh = """# 角色设定
你是一位联网信息搜索专家，你需要根据用户的问题，通过联网搜索来搜集相关信息，然后根据这些信息来回答用户的问题。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，利用我提供给你的工具，获取相对应的信息、资料，以解答用户的问题。
以下是你在执行任务过程中需要遵循的原则：
- 充分理解用户需求：你需要全面分析和理解用户的问题，必要时对用户的问题进行拆解，以确保领会到用户问题的主要意图。
- 灵活使用工具：当你充分理解用户需求后，请你使用我提供的工具获取信息；当你认为上次工具获取到的信息不全或者有误，以至于不足以回答用户问题时，请思考还需要搜索什么信息，再次调用工具获取信息，直至信息完备。"""

default_system_prompt_en = """# Role
You are an expert in online search. You task is gathering relevant information using advanced online search tools based on the user's query, and providing accurate answers according to the search results.

# Task Description
Upon receiving the user's query, you must thoroughly analyze and understand the user's requirements. In order to effectively address the user's query, you should make the best use of the provided tools to acquire comprehensive and reliable information and data. Below are the principles you should adhere to while performing this task:

- Fully understand the user's needs: Analyze the user's query, if necessary, break it down into smaller components to ensure a clear understanding of the user's primary intent.
- Flexibly use tools: After fully comprehending the user's needs, employ the provided tools to retrieve the necessary information.If the information retrieved previously is deemed incomplete or inaccurate and insufficient to answer the user's query, reassess what additional information is required and invoke the tool again until all necessary data is obtained."""

tools_api_description_zh_map = {
    "web_search_chinese": {
        "type": "function",
        "function": {
            "name": "web_search_chinese",
            "description": "联网搜索工具（中文优化）：针对中文问题进行互联网搜索，返回相关网页列表与摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "互联网搜索的关键词或问题",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "web_search_internationl": {
        "type": "function",
        "function": {
            "name": "web_search_internationl",
            "description": "联网搜索工具（国际版）：针对英文或国际化问题进行互联网搜索，返回相关网页列表与摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "互联网搜索的关键词或问题",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "get_content": {
        "type": "function",
        "function": {
            "name": "get_content",
            "description": "网页内容获取工具：打开指定 URL，返回页面的主要文本内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "目标链接，应该是一个完整的 URL（以 http 开头）",
                    }
                },
            },
            "required": ["url"],
        },
    },
}


tools_api_description_en_map = {
    "web_search_chinese": {
        "type": "function",
        "function": {
            "name": "web_search_chinese",
            "description": "Online search tool (Chinese optimized): perform internet search for Chinese queries and get a list of relevant webpages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or question).",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "web_search_internationl": {
        "type": "function",
        "function": {
            "name": "web_search_internationl",
            "description": "Online search tool (International): perform internet search for English or international queries and get a list of relevant webpages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or question).",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "get_content": {
        "type": "function",
        "function": {
            "name": "get_content",
            "description": "Webpage content tool: open the given URL and return the main text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target link (a complete URL starting with http).",
                    }
                },
            },
            "required": ["url"],
        },
    },
}


def get_system_prompt(language: str) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    if language == "zh":
        return f"当前日期: {current_date}\n\n" + default_system_prompt_zh
    elif language == "en":
        return f"Current Date: {current_date}\n\n" + default_system_prompt_en
    else:
        raise ValueError(f"Unknown language {language}")


def get_tools_api_description(language: str, func_list: list[str]) -> list[dict]:
    if language == "zh":
        return [tools_api_description_zh_map[k] for k in func_list]
    elif language == "en":
        return [tools_api_description_en_map[k] for k in func_list]
    else:
        raise ValueError(f"Unknown language {language}")
