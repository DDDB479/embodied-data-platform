# DSH 插件

`packages/dsh-plugin` 是给 DeepSeek Harness 的封装：把 19 个 MCP 工具注册成**原生 agent 工具**（外加一个不依赖 Python 的 `embodied_self_check`），再注入一段工作流提示词。插件本身不实现任何数据逻辑——它启动 `python -m embodied_core.mcp_server` 并做 JSON-RPC 桥接，所以模型换成 Codex / Claude Code 时，行为完全一致。

## 为什么还要插件

MCP 客户端都得先"发现工具"再"决定用哪个"。插件多做三件事：

1. **工具直接可用**：`embodied_detect_dataset` 等出现在工具列表里，带 `embodied_` 前缀，和其他插件的工具不撞名。
2. **提示词注入**：`systemPrompt.section` 里写明"先 detect、再 summarize、score≥80 才转换、必须含 state/action"——模型不需要先试错才学会正确顺序。
3. **自检工具**：`embodied_self_check` 不依赖 Python 也能回答"我的解释器在哪、PYTHONPATH 对不对、MCP 起来没有"，配置出问题时不用猜。

## 安装

### 1. 把包装进 profile

```bash
PROFILE=~/.dsh/profiles/desktop
cd "$PROFILE"
pnpm add "link:D:/具身平台插件开发/packages/dsh-plugin"
```

`link:` 让 profile 直接引用仓库目录，改插件代码不需要重新安装。pnpm 会提示 `@deepseek-ai/cordis`、`@deepseek-ai/dsh` 两个 peer 依赖未解析——它们是 `optional`，由 Harness 进程自己提供，忽略即可。

从 npm tarball 安装时不需要 `link:`，也不需要额外 `pip install`：

```bash
pnpm add dsh-embodied-data
```

发布包在 `prepack` 阶段把 `src/embodied_core` 复制到 `python/embodied_core`，运行时自动探测这个内置目录。仓库开发仍用 `link:`，这样改 Python 源码立即生效。可选的 `pyarrow`/`h5py`/`mcap` 读取器仍装在使用它的 Python 环境里。

### 2. 在 profile 的 package.json 里登记 bundle

```json
{
  "dependencies": {
    "dsh-embodied-data": "link:D:/具身平台插件开发/packages/dsh-plugin"
  },
  "dsh": {
    "profile": {
      "bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app", "dshmarket", "dsh-ops", "billion-context", "dsh-embodied-data"]
    }
  }
}
```

`dependencies` 由 `pnpm add` 写入，只有 `bundles` 需要手工加——它决定启动时加载哪些插件。

### 3. 配置解释器并重启 Harness

在 profile 的 `cordis.patch.yml` 末尾追加一行按 id 覆盖的配置（bundle 层已经 insert 过这个插件，这里只补 config）：

```yaml
- id: dsh-embodied-data
  config:
    python: /abs/path/to/python
    pythonPath:
      - /abs/path/to/deps
```

改完重启 Harness（或重新加载 profile）。工具列表里应出现 20 个以 `embodied_` 开头的工具；第一次先调 `embodied_self_check`，它会直接告出解释器、`PYTHONPATH` 和 MCP 进程状态。

## 配置项

`cordis.patch.yml` 里的 `config` 会传给 `apply(ctx, config)`：

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `python` | `""` | Python 解释器绝对路径；空则按 `EMBODIED_PYTHON`/`DSH_PYTHON`/`PYTHON`/`python3`/`python` 顺序探测 |
| `pythonRoot` | 自动探测 | 含 `src/embodied_core` 或 `embodied_core` 的目录；优先使用包内 `python/`，其次向上查找仓库源码，最后尝试 `pythonPath`/`PYTHONPATH` 里的已安装包 |
| `pythonPath` | `[]` | 额外加入 `PYTHONPATH` 的目录（相对插件根解析，例如工作区共享的 `.devtools/pydeps`） |
| `prefix` | `"embodied_"` | 注册到 Harness 的工具名前缀 |
| `timeoutMs` | `120000` | 单次工具调用超时 |
| `startupTimeoutMs` | `20000` | 会话启动时等待 MCP `tools/list` 的时间 |
| `restart` | `true` | Python 进程退出后，下次调用自动重启 |
| `promptPolicy` | `true` | 是否注入工作流提示词 |
| `includeTools` | `[]` | 只暴露这些 MCP 工具（空=全部） |
| `excludeTools` | `[]` | 屏蔽这些 MCP 工具 |
| `maxOutputChars` | `24000` | 单次输出截断长度，防止大 JSON 冲爆上下文 |
| `confirmOutsideWorkspace` | `true` | 写工具目标在工作区外时要求一次性审批；无审批服务时失败关闭 |
| `workspaceRoot` | `""` | 可选的兜底工作区根；空则使用 DSH sandbox policy 或会话 cwd |

示例（把依赖目录也带上）：

```yaml
- insert:
    - id: dsh-embodied-data
      name: dsh-embodied-data
      config:
        python: C:/Users/me/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe
        pythonPath:
          - D:/具身平台插件开发/src
          - C:/work/deps
        promptPolicy: true
```

## 自检

不启动 Harness 也能验证插件本身：

```bash
cd packages/dsh-plugin
node scripts/probe.mjs \
  --python /abs/path/to/python \
  --deps /abs/path/to/deps          # 可选，需要读 parquet 时给
```

它会用一个 mock 的 cordis ctx 走一遍真实流程：`apply()` → 注册工具 → 启动 MCP → `tools/list` → 调 `detect_dataset` 和 `summarize_dataset`，最后打印工具清单与真实返回。全部通过时退出码 0：

```
ok            yes
python root   D:/具身平台插件开发
import path   D:/具身平台插件开发/src
tools         20
              embodied_self_check, embodied_detect_dataset, ...
prompt        embodied-data:toolbox
detect        {"format":"lerobot","version":"v2.1","confidence":0.98,...}
```

常见失败：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `python root (not found)` | 插件找不到 `embodied_core` | 设 `pythonPath`，或把 `src` 加进 `PYTHONPATH` |
| `tools 1` | MCP 没起来，只剩自检工具 | 看 `embodied_self_check` 的 `server` 字段；多为解释器路径错或缺依赖 |
| `frames_unavailable` | 缺 `pyarrow`/`h5py`/`mcap` | 装依赖，或在 `pythonPath` 里指到已装好的目录 |

验证发布包而不是源码目录：

```bash
node tools/check_clean_install.mjs \
  --python /abs/path/to/python \
  --fixture .devtools/fixtures/lerobot_v21
```

这个脚本会执行 `pack → 临时目录安装 → 解析内置 Python → 启动 MCP → detect`，用来防止“仓库里能跑、装完就坏”的回归。

## 写入安全

- 只读工具（detect / summarize / validate / quality / segment / catalog search…）声明为可并发。
- 写工具（`convert_dataset`、`scaffold_project`、`catalog_index`、带 `out` 的 `labeling_tasks`）不声明并发，DSH 会串行调度。
- 写目标不在 DSH 工作区内时，插件通过 `ctx.get('approval').request(...)` 申请一次 `allowed-once`；拒绝、不可用或只读模式都会阻止写入。
- 相对路径参数统一按会话工作区解析，不再相对 Python 子进程的安装目录。

## 异步作业与进度

- `embodied_submit_job` 只接受 `quality_report` / `validate_dataset` / `quality_gate`，后台线程运行，`embodied_job_status` 轮询进度与结果，`embodied_cancel_job` 请求协作式取消。
- 异步作业路径同样按会话工作区解析；由于只读，不需要写入审批，也不可能绕过写工具的工作区策略。
- MCP 客户端会解析 `notifications/progress`，DSH 侧按 50% 里程碑写入日志；有自定义进度 UI 的宿主可以直接订阅 `client.onProgress(...)`。
- `quality_report` 的缓存写在用户缓存目录，不在数据集或工作区里产生新文件；`refresh_cache=true` 可强制重算。
- `quality_gate` 的规则可内联传入；CLI 额外支持 `--rules rules.json|rules.toml`，便于把同一套策略放进 CI。
- 编目工具使用 schema v2；`catalog_history` 可直接读取版本、运行和谱系，不需要重新扫描数据盘。

## 与 MCP 的关系

两条路都能用，选一个即可：

- **只用 MCP**：不装插件，在 Codex/Claude 里直接配 `embodied-mcp`。工具名是 `detect_dataset` 等原始名字，没有提示词注入。
- **用插件**：工具名为 `embodied_*`，带提示词与自检。插件内部仍通过 MCP 调用，所以两条路的能力与返回结构完全一致。

插件不会自动安装 Python 依赖，也不会下载数据；它只负责进程生命周期与协议桥接。
