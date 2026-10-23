# 接入不同的模型 / 客户端

设计前提：**能力在 Python 侧，模型只是调用方**。所以"支持哪些模型"不是一个代码分支问题，而是"客户端能不能连 MCP"。目前主流客户端都行。

## 一次性准备

```bash
# 1) 确认服务可用（不依赖任何客户端）
embodied-mcp --list-tools

# 2) 如果没 pip install，用绝对路径 + PYTHONPATH 也可以
PYTHONPATH=D:/具身平台插件开发/src \
  /path/to/python -m embodied_core.mcp_server --list-tools
```

服务是 stdio JSON-RPC（`PROTOCOL_VERSION = "2024-11-05"`），没有端口、没有常驻进程，客户端拉起它、用完退出。

## 通用 MCP 配置片段

几乎所有客户端都认这个形状（字段名可能有细微差别）：

```json
{
  "mcpServers": {
    "embodied-data": {
      "command": "embodied-mcp",
      "args": [],
      "env": {
        "PYTHONPATH": "D:/具身平台插件开发/src"
      }
    }
  }
}
```

Windows / 未安装成 console script 时，把 `command` 换成解释器、`args` 换成模块名：

```json
{
  "mcpServers": {
    "embodied-data": {
      "command": "C:/path/to/python.exe",
      "args": ["-m", "embodied_core.mcp_server"],
      "env": {
        "PYTHONPATH": "D:/具身平台插件开发/src;C:/path/to/deps"
      }
    }
  }
}
```

## Codex CLI

```bash
codex mcp add embodied-data -- embodied-mcp
codex mcp list          # 确认已注册
```

等价的手写配置（`~/.codex/config.toml`）：

```toml
[mcp_servers.embodied-data]
command = "embodied-mcp"
args = []
# env = { PYTHONPATH = "D:/具身平台插件开发/src" }
```

装好之后在 Codex 里直接描述任务即可，它会自己挑工具：

> `./datasets/night_run` 是同事刚同步过来的，帮我识别格式、跑校验和质检，
> 把有跳变或停滞的 episode 整理成标注任务清单，再转成 LeRobot 到 `./out/lerobot`。

## Claude Code

```bash
claude mcp add embodied-data -- embodied-mcp
```

或写进项目根的 `.mcp.json`：

```json
{
  "mcpServers": {
    "embodied-data": {
      "command": "embodied-mcp",
      "args": []
    }
  }
}
```

Claude Code 的 MCP 服务器可以随项目提交（`.mcp.json`），团队共享同一套数据工具链。

## DeepSeek

DeepSeek 的模型 API 本身不托管 MCP；接法是**经由支持 MCP 的客户端**。两条路：

1. **DeepSeek Harness（本仓库自带插件）** —— 见 [plugin.md](plugin.md)。插件把 13 个工具注册成原生 agent 工具，并且注入一段工作流提示词，模型不需要先"发现"工具就知道该怎么用。
2. **任何支持自定义 MCP 的 DeepSeek 客户端**（Cherry Studio、Cline、Roo Code 等）—— 用上面的通用 JSON 片段即可。

一个实际可用的提示词（也放在 `examples/agent_prompt.md`）：

```
你在维护一个具身智能数据平台。处理任何数据集前，先 detect_dataset 确认格式，
再 validate_dataset 拿 score 和 issues；只有 score >= 80 且没有 error 才进入转换。
需要逐帧分析时用 quality_report（不要自己对 parquet 写代码）。
转换统一用 convert_dataset，不要手写 parquet。
```

## Cursor / 其他 IDE 客户端

`~/.cursor/mcp.json` 或项目级 `.cursor/mcp.json`，内容同上（`mcpServers` 键）。

## 模型能力差异的实际影响

工具是同一套，但不同模型的使用习惯不同，插件层做了两件事来抹平：

| 差异 | 处理 |
| --- | --- |
| 有的模型不主动探索工具 | DSH 插件通过 `systemPrompt.section` 注入 7 步工作流与约定（score≥80、必须含 state/action、保留索引列） |
| 有的模型喜欢输出大 JSON | 工具描述里写清"返回结构与字段含义"，输出统一 `{tool, ok, data}` 并截断到 24k 字符 |
| 有的模型不会处理异常 | 所有工具返回结构化错误而不是抛栈；`validate` 对非数据集路径返回 `ok:false` 而不是异常 |

## 没有 MCP 的客户端怎么办

退化到命令行：模型的"工具"就是 shell。把 `embodied` 放进 PATH，然后在提示词里说明可用命令：

```bash
embodied detect|info|validate|quality|segment|label|convert|scaffold|catalog ... --json
```

所有命令都支持 `--json`，退出码有语义（0 通过 / 1 校验失败 / 2 用法错误 / 3 不支持的格式），便于脚本和 agent 判断。
