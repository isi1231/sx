# Agent Demo

每天的学习讲解、流程图和实测证据在 [`../doc/`](../doc/README.md)：
`day01-02` 多轮对话 · `day03` Prompt 工程 · `day04` 工具调用 · `day05` 记忆机制 ·
`day06` 上下文工程 · `day07` 可观测性 / trace · `day08` 多步收敛。
代码原理手册（控制流 / 状态机 / 边界条件）在 [`../code_view/`](../code_view/README.md)。

跑测试（**在 `agent_demo/` 目录下**）：

```powershell
python -m pytest
```

> ⚠️ 不要用 `python test/test_xxx.py` 直接跑测试文件，也不要裸敲 `pytest` —— 原因见 `doc/day03.md`。

## Day 3: Prompt Engineering

运行第三天实验程序：

```powershell
python run_prompt.py
```

示例：

```text
你好
/plan 我想做一个天气查询网站
/json 帮我制定两周的 Python 学习计划
clear
quit
```

本日代码包含：

- `System Prompt`：定义角色、行为边界和安全规则
- `Few-shot`：用示例约束回答风格
- 输出约束：要求模型返回固定 JSON 字段
- 任务拆解：要求模型输出目标、前置条件、步骤和风险
- 失败场景：处理 JSON 解析失败和字段缺失
- Prompt 注入防护基础：用 `<untrusted_user_input>` 标记用户输入边界

运行测试：

```powershell
python -m pytest
```

## Day 4: Agent Tool Calling

运行第四天实验程序：

```powershell
python run_agent.py
```

本日代码包含：

- `calculator`：不使用 `eval()` 的安全数学计算工具
- `search_notes`：只搜索项目内 `data/notes.txt` 的本地工具
- 工具白名单和参数校验
- DeepSeek 原生兼容的 Tool Calling
- 最大工具调用轮数，避免无限循环
- 工具异常和非法 JSON 参数处理

示例问题：

```text
25 * 4 + 10 等于多少？
请搜索笔记中关于 Agent 的内容
请删除 .env 文件
```

最后一个问题应该被拒绝，因为项目没有注册删除文件工具。

## Day 5: Multi-turn Memory

运行第五天实验程序：

```powershell
python run_memory_agent.py
```

本日代码包含：

- `ConversationMemory`：有界短期记忆，支持增 / 清 / 查
- `AgentSession`：承载多轮对话的 Agent 会话
- 记忆裁剪：超过容量时只保留最近的消息
- `run_agent` 保持 Day 4 的单轮接口不变
- 交互命令：`memory` 查看当前记忆，`clear` 清空记忆

示例：

```text
我叫小明
我叫什么名字？
1+1 等于多少
memory
clear
quit
```

### 修复记录：滑窗裁剪错位（回归缺陷）

#### 现象

多轮对话超过记忆容量后，发给模型的历史会以 `assistant` 开头 —— 模型看到「没有对应提问的回答」。

#### 根因

`ConversationMemory._trim` 只按条数从尾部截取，不考虑消息的配对关系：

```python
# 修复前
def _trim(self) -> None:
    if len(self._messages) <= self.max_messages:
        return

    self._messages = self._messages[-self.max_messages :]
```

而 `AgentSession.ask` 在 `add_user_message` 之后**立即**构建 messages 发给模型，此时长度为奇数；`_trim` 又在每次 `add` 时触发。于是窗口头部会留下一条提问已被丢弃的 `assistant` 消息。

#### 实测证据（默认 `max_messages=10`，跑 20 轮）

| 实现 | 历史以 `assistant` 开头的轮次 |
| --- | --- |
| 修复前 | 第 6 ~ 20 轮，共 15 轮（**必现，且不会自愈**） |
| 修复后 | 无 |

奇数窗口（3 / 5 / 11）在修复前 10 轮内分别错位 9 / 8 / 5 次，修复后均为 0 次。

#### 修法

```python
# 修复后
def _trim(self) -> None:
    if len(self._messages) <= self.max_messages:
        return

    trimmed = self._messages[-self.max_messages :]

    # 历史必须以 user 开头。按条数截断时，窗口头部可能剩下一条
    # 提问已被丢弃的 assistant 消息，模型会收到“没有对应提问的回答”。
    while trimmed and trimmed[0]["role"] != "user":
        trimmed.pop(0)

    self._messages = trimmed
```

代价：窗口实际条数可能比 `max_messages` 少 1 条。这是保证历史完整性的必要开销。

#### 验证

真实 API 连跑 6 轮后执行 `memory` 命令：

```text
1. [user] 我叫什么名字？
2. [assistant] 你叫小明。
3. [user] 1+1 等于多少
4. [assistant] 1 + 1 = 2。
5. [user] 2+2 等于多少
6. [assistant] 2 + 2 = 4。
7. [user] 3+3 等于多少
8. [assistant] 3 + 3 = 6。
9. [user] 4+4 等于多少
10. [assistant] 4 + 4 = 8。
```

窗口以 `user` 开头、以 `assistant` 结尾，被丢弃的是完整的第 1 轮，而不是半截。

### 测试

```powershell
python -m pytest
```

新增 6 个回归用例（**26 passed**）：

- 不变量：任意窗口大小（含奇数）下，历史必须以 `user` 开头
- 裁剪必须丢弃头部的孤儿 `assistant`
- 奇数窗口收缩时保留成对的一轮
- 全 `assistant` 窗口降级为空，而不是留下错位历史
- Agent 层：20 轮后发给模型的历史首条始终为 `user`（**这条在修复前必定失败**）
- Agent 层：20 轮后记忆首条始终为 `user`

### 遗留待办（Day 6 复核）

| 项 | 说明 | 状态 |
| --- | --- | --- |
| 记忆仍是硬截断 | 被裁掉的对话永久丢失 | ✅ Day 6 已换成摘要压缩 |
| 请求失败会留下孤儿 `user` 消息 | 写入发生在调用模型之前 | ✅ Day 6 已改成成功后提交 |
| 上一轮工具结果不可见 | `role="tool"` 不进记忆 | ✅ Day 6 已写入 `[本轮工具调用]` |
| `max_messages` 语义不清 | 实际只按「条数」保留，`=10` 只有 5 轮 | ⬜ 待改 `max_turns`（推迟到 D8） |
| `max_tool_rounds=3` 偏小 | 每轮 = 1 次模型请求，实际只够 2 次工具调用 | ⬜ 待处理 |
| 观察 | `tool_choice="auto"` 下，模型会对 `1+1` 这类简单算式直接回答而不调用 `calculator` | — |

## Day 6: Context Engineering

运行第六天实验程序：

```powershell
python run_memory_agent.py
```

本日代码包含：

- `app/summarizer.py`：把被裁掉的旧对话压成一段摘要（唯一碰网络的一层）
- `ConversationMemory` 双模式：不传 `summarizer` 退化为 Day 5 的纯滑窗，传了就启用摘要压缩
- 摘要**累积**合并：`summarizer(旧摘要, 新增对话) -> 新摘要`
- 上下文装配顺序：`system 规则` → `system 摘要` → 历史 → 本轮提问
- 工具结果入记忆：回答后附加 `[本轮工具调用]` 流水，下一轮模型可见
- `_clip()` + `TOOL_RESULT_LIMIT=200` + `MAX_TOOL_LOG_ENTRIES=5`：工具结果进记忆前削平
- 延迟提交：记忆只在**成功拿到回答之后**写，请求失败不留孤儿 `user` 消息
- 交互命令新增 `summary` 查看当前摘要

示例：

```text
我叫小明
我叫什么名字？
memory
summary
clear
quit
```

### 关键改动：写入时机

```python
# Day 5：进循环前就写 → 抛异常后留下 ['user'] 孤儿
self.memory.add_user_message(wrapped_input)
messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.memory.get_messages()]

# Day 6：出口处统一提交 → 失败后记忆保持原样
messages = self._build_messages(wrapped_input)   # 本轮提问只进局部 messages
...
self._commit(wrapped_input, answer, tool_log)    # 成功后一次性落记忆
```

### 关键改动：上下文装配

```python
def _build_messages(self, wrapped_input: str) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]      # 恒定 → 吃前缀缓存

    summary = self.memory.get_summary()
    if summary:                                                     # 只在压缩时变
        messages.append({
            "role": "system",
            "content": f"<conversation_summary>\n{summary}\n</conversation_summary>",
        })

    messages.extend(self.memory.get_messages())                     # 滑动窗口
    messages.append({"role": "user", "content": wrapped_input})      # 每轮必变 → 垫底
    return messages
```

摘要单独占一条 `system`、不拼进第一条，是为了让**规则段的前缀缓存不被压缩打掉**。

### 硬截断 vs 摘要压缩（同一段 6 轮对话，`max_messages=4`）

| | 纯滑窗（Day 5） | 摘要压缩（Day 6） |
| --- | --- | --- |
| 窗口内条数 | 4 | 4 |
| `compressed_count` | 0 | 8 |
| 摘要 | `''` | `用户名叫 小明；用户在做猪场 3D 姿态估计项目` |
| 还能看到「小明」吗 | **否** | **是** |

不变量：`compressed_count + len(memory) == 写进去的总条数`（实测 `8 + 4 = 12`）。

### 工具结果的两份拷贝

一轮工具调用后，记忆里的 `assistant` 内容是：

```text
计算结果是 110。
[本轮工具调用]
- calculator({"expression":"25 * 4 + 10"}) -> 110
```

而 `ask()` 返回给用户的仍是干净的 `计算结果是 110。`：
**答案是给用户看的，记忆是给模型看的。**

超长工具结果只在**进记忆的那份**上截断，同一次请求里的 `role="tool"` 消息一字未动：

```text
进记忆的那份：267 字   发给模型的那份：1000 字
一轮调了 8 次工具 → 记忆里记了 5 条，本次请求发了 8 条
```

### 验证

```powershell
python -m pytest                    # 53 passed（Day 5 是 26）
python -m pytest -s -k keeps_what   # 看「滑窗 vs 摘要」的对照实验原文
python run_memory_agent.py          # 交互式亲眼看：聊几轮后输 summary / memory
```

`-s -k keeps_what` 的输出：

```text
[对照] 同样的 4 轮、同样的 max_messages=4
  纯滑窗   窗口内 = ['继续', '好的。', '继续', '好的。']
           摘要   = ''
           「小明」还在吗 = False
  摘要压缩 窗口内 = ['继续', '好的。', '继续', '好的。']
           摘要   = '用户叫小明；用户在做猪场项目'
           「小明」还在吗 = True
```

两边窗口内容完全一样，差别只在摘要 —— 这就是 Day 6 的全部意义。

> 这一天**没有**新建 `scripts/` 目录。隔壁 fastapi 的 `scripts/check_dayNN.py` 验证的是跑起来的 HTTP 服务，
> 而这里验证的全是纯逻辑，`test/` 本来就全覆盖了。详见 `doc/day06.md` 第十节。

新用例的有效性用 6 组探针验证过（把改动逐条还原成 Day 5 行为，看测试接不接得住；
一次性实验，未留在仓库）：

| 探针 | 还原成 | 结果 |
| --- | --- | --- |
| 1 | 进循环前先写记忆 | ✅ 2 条用例失败 |
| 2 | 工具流水不进记忆 | ✅ 1 条用例失败 |
| 3 | 只硬截断、不压缩 | ✅ 1 条用例失败 |
| 4 | 窗口裁剪不做配对修正 | ❌ **0 条失败** |
| 5 | 摘掉纯滑窗的配对守卫（记忆层） | ✅ 4 条用例失败 |
| 6 | 摘掉摘要分支的配对守卫 | ✅ 1 条用例失败 |

**探针 4 没挂是有效信息**：Day 6 不再在发请求前写 `user`，记忆每轮只增两条、窗口长度恒为偶数，
Day 5 那条错位路径已经不会被经过。所以 `test_agent_history_always_starts_with_user`
仍然是一个必须成立的不变量，但**不能再当作错位的回归保护** —— 这个职责已由探针 5/6 证明的记忆层用例接管。

### 已知问题（Day 6 新引入 / 遗留）

| # | 问题 | 计划 |
| --- | --- | --- |
| 1 | `max_messages` 语义仍是「条数」 | ✅ Day 8 已改为 `max_turns` |
| 2 | `_message_to_dict` 三种分支返回的 dict 形状不一致 | ✅ Day 7 已收口 |
| 3 | 摘要本身没有长度硬约束，只靠 prompt 里的「不超过 N 字」 | ✅ Day 7 已加代码侧截断 |
| 4 | `max_tool_rounds=3` 只够 2 次工具调用 | ✅ Day 8 已改为 `max_model_calls=6` |
| 5 | `calculator` 会泄漏 `OverflowError` / `ZeroDivisionError` | ✅ Day 8 已收口 |
| 6 | 压缩发生在 `add()` 里，是**同步阻塞**的（一次压缩 = 一次模型请求） | D15 异步化 |

第 6 条是 Day 6 新引入的真实代价：用户提问触发压缩时，这一轮会多出一次模型调用的延迟。
现在能接受（几分钟一次），但要做成服务就必须异步，否则会拖慢 P99。

## Day 7: Observability / Trace

运行第七天实验程序：

```powershell
python run_trace.py                    # 聊天，逐事件写入 trace/trace-<时间>.jsonl
python run_trace.py --replay <file>    # 离线回放一份 trace，不花钱
```

聊天中可用的命令：

```text
trace     打印当前 trace 的回放（含这一轮真正发出去的 messages）
stats     只看 token / 耗时 / 前缀缓存命中率
memory    查看当前记忆
clear     清空记忆
quit      退出
```

示例：

```text
25 * 4 + 10 等于多少？
trace
stats
quit
```

本日代码包含：

- `app/trace.py`：`TraceRecorder`（只管记）+ `json_safe` / `render_replay` / `summarize_usage` / `find_errors`（纯函数，只解释）
- 一轮对话打 5 个点：`request` / `response` / `tool_call` / `answer` / `error`
- 每事件一行 jsonl，**每落一个事件就 append 一次** → 进程崩在哪儿，trace 就停在哪儿
- `emit()` 存**快照**（深拷贝），不会因为 `messages` 后面继续增长而污染早期事件
- `json_safe()` 三层兜底（`model_dump` → `__dict__` → `str`），保证**记录失败时永不抛异常**
- `_read_usage()` 抠 token 用量，含 DeepSeek 的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
- 离线 `--replay`：一份 trace 可以反复看，不用重跑模型
- `tracer` 是可选注入的，不传就是 `None`，零开销
- `trace/` 已进 `.gitignore`

### 两个设计取舍

**1. 每事件一次 open/write，而不是退出时统一 save。**

慢一点，但崩溃时 trace 恰好停在崩溃点 —— 那一条记录最值钱。想验证：跑一轮后直接 `Ctrl+C`，`--replay` 依然读得出来。

**2. `TraceRecorder` 只记不解释。**

统计和渲染都在纯函数里（喂事件列表就出文本），所以回放可以离线跑、可以单测喂造出来的事件，也**不用为了改回放样式而重调模型**。

### 顺带结清 Day 6 的两笔账

- `_message_to_dict` 输出形状收口：`role` / `content` 永远在，`tool_calls` 展开成纯 dict（否则 trace 一落盘 `json.dumps` 就炸）
- 摘要加上代码侧硬上限：`HARD_LIMIT_FACTOR = 2`，默认 `max_chars × 2`，截断保留头部（`SUMMARY_PROMPT` 要求按重要性排序）

### 验证

```powershell
python -m pytest                  # 103 passed（Day 6 是 53）
python run_trace.py               # 交互式亲眼看：问一句后输 trace / stats
python run_trace.py --replay trace/trace-<时间>.jsonl
```

回放长这样：

```text
── 第 1 轮 ──────────────────────────────────────────────
   1. → 请求  2 条 ['system', 'user']  tools=calculator,search_notes
   2. ← 响应  0.01ms  tokens 1230  缓存命中 1140  要调 calculator
   3. ⚙ 工具  calculator({"expression": "25 * 4 + 10"})  0.17ms ok → 110
   4. → 请求  4 条 ['system', 'user', 'assistant', 'tool']  ...
   7. → 请求  6 条 ['system', 'user', 'assistant', 'tool', 'assistant', 'tool']  ...
   9. ✓ 回答  25 * 4 + 10 = 110。笔记里关于 Agent 的部分主要讲工具调用和记忆。  本轮 13.24ms

── 第 2 轮 ──────────────────────────────────────────────
   2. ✗ 出错  llm_call 阶段 RuntimeError: 模型服务 502

合计：4 次请求 / 5698 tokens / 前缀缓存命中率 96.0% / 出错 1 次
```

新用例的有效性用 6 组探针验证过（把改动逐条还原成 Day 6 行为，看测试接不接得住；
一次性实验，未留在仓库）：

| 探针 | 还原成 | 结果 |
| --- | --- | --- |
| 1 | `emit` 存活引用，不深拷贝 | ✅ 2 条用例失败 |
| 2 | `_message_to_dict` 不做形状归一 | ✅ 2 条用例失败 |
| 3 | `json_safe` 去掉 `model_dump` 层 | ✅ 1 条用例失败 |
| 4 | 不 emit `tool_call` | ✅ 2 条用例失败 |
| 5 | 不 emit `error` | ✅ 2 条用例失败 |
| 6 | 摘要不设硬上限 | ✅ 2 条用例失败 |

（Day 6 的探针 4 是「0 条失败」，今天 6/6 全中 —— 区别是今天每条探针都打在今天新写的代码路径上。）

### 已知问题

| # | 问题 | 计划 |
| --- | --- | --- |
| 1 | `max_messages` 语义仍是「条数」 | ✅ Day 8 已改为 `max_turns` |
| 2 | `max_tool_rounds=3` 只够 2 次工具调用 | ✅ Day 8 已改为 `max_model_calls` |
| 3 | `calculator` 会泄漏 `OverflowError` / `ZeroDivisionError` | ✅ Day 8 已收口（只抛 `ValueError`） |
| 4 | 失败的工具调用**不计入**合计里的「出错 N 次」 | ✅ Day 8 已拆成「异常 / 工具失败」两行，并修掉根因（`ok` 标志本身是错的） |
| 5 | **没采集 `reasoning_content`**（计划 D7 里写的 "thought"）—— 现在用的 `deepseek-chat` 没这个字段 | 用上推理模型时补 |
| 6 | trace 文件无轮转、无上限，长跑会一直涨 | 后续 |

`tests/` 耗时从 0.3 秒涨到 3~5 秒，**不是代码回归**：项目里 5 条用例用了 `tmp_path`，
而 pytest 在基目录非空时要扫描清理旧的编号目录 —— 本机 `os.rmdir`（删目录）实测约 **250 ms/次**（C:），
`test_memory.py`（不用 `tmp_path`，21 条）单独跑只要 0.06 秒。详见 `doc/day07.md` 第十二节。

> 想验证：`python -m pytest test/test_trace.py --basetemp=<一个全新空目录>` → 0.14 秒；去掉 `--basetemp` 再跑 → 3.6 秒。

> 这一天同样**没有**新建 `scripts/` —— 理由见 `doc/day06.md` 第十节。

## Day 8: Multi-step Convergence

运行第八天实验程序：

```powershell
python run_memory_agent.py     # 记忆窗口 5 轮、一次 ask 最多 6 次模型请求
python run_trace.py            # 同上，额外打 trace；trace / stats 命令看回放
```

本日代码包含三个互不替代的机制（**轮次预算挡「话多」，无进展检测挡「打转」，重试挡「抖动」**）：

- **轮次预算**：`max_turns`（记忆，单位是「轮」）+ `max_model_calls`（一次 ask 的请求次数，默认 6）
- **无进展检测**：连续两批工具调用 `(工具名, 参数)` 完全相同时中断，**拦在工具执行之前**
- **失败重试**：指数退避（0.5s → 1s），只包**单次请求**，不会重放整轮

顺带收口：

- `ToolResult`：`text` 给模型看、`ok` / `reason` 给程序看 —— 不再靠文本前缀猜成败
- `calculator` 只抛 `ValueError`（`ZeroDivisionError` / `OverflowError` / `SyntaxError` 全部翻译成中文说明）
- trace 合计把「异常」和「工具失败」拆成两行；error 事件显示是第几次尝试；answer 事件显示中断原因

### 三个预算的分工

| 参数 | 管什么 | 默认 |
| --- | --- | --- |
| `max_turns` | 记忆窗口装多少轮历史 | 5（= 10 条消息） |
| `max_model_calls` | 一次 `ask()` 最多请求模型几次 | 6（= 最多 5 次工具调用 + 1 次回答） |
| `max_retries` | 单次请求失败后最多重试几次 | 纯逻辑层 `0` / 装配层 `2` |

`max_retries` 的默认值分两层是刻意的：单测要确定性，生产要可用性。

### 关键改动：无进展检测拦在执行之前

```python
signature = tuple(
    (_get_tool_call_value(tc, "name") or "",
     _get_tool_call_value(tc, "arguments") or "{}")
    for tc in tool_calls
)
repeat_count = repeat_count + 1 if signature == last_signature else 1
last_signature = signature

if repeat_count >= NO_PROGRESS_LIMIT:
    return self._stop(..., stopped_by="no_progress", repeated=signature[0][1])
```

两个理由：同样的 (工具, 参数) 上一次就把结果写进上下文了，再执行一次没有新信息；
而且换成「下单」「发邮件」这类有副作用的工具之后，「先执行再判断」是真的重复下单。

### 实测：一份 4 轮 trace

```text
── 第 1 轮 ── 3 次工具调用后回答（旧默认 max_tool_rounds=3 跑不完）
   1. → 请求  2 条 ['system', 'user']
   3. ⚙ 工具  calculator({"expression": "12 * 8"})  0.08ms ok → 96
   6. ⚙ 工具  calculator({"expression": "96 + 4"})  0.13ms ok → 100
   9. ⚙ 工具  search_notes({"query": "Agent"})  0.71ms ok → 笔记主题：Agent…
  12. ✓ 回答  12 * 8 = 96，加 4 是 100。笔记里说 Agent 包含工具调用和记忆。

── 第 2 轮 ── 打转被拦下（2 个 request，只有 1 个 tool_call）
   5. ← 响应  0.01ms  要调 calculator
   6. ✓ 回答  检测到连续重复的工具调用（calculator 参数完全相同），判断为无法推进，任务已停止。
              [中断：检测到重复调用]

── 第 3 轮 ── 抖动两次后成功
   2. ✗ 出错  llm_call 阶段 RuntimeError: 模型服务 502（第 1/3 次尝试，将重试）
   3. ✗ 出错  llm_call 阶段 RuntimeError: 模型服务 502（第 2/3 次尝试，将重试）
   5. ✓ 回答  刚才那两次请求抖动，重试成功了：答案是 2。  本轮 1517.58ms

── 第 4 轮 ── 工具失败 ≠ 异常
   3. ⚙ 工具  calculator({"expression": "1/0"})  0.06ms 失败 → 工具拒绝执行：除数不能为 0

合计：
  模型请求  9 次
  token     12300（prompt 12000 + completion 300）
  前缀缓存  命中率 97.0%（命中 11640 / 未命中 360）
  异常      2 次
      · 第 3 轮 llm_call：RuntimeError 模型服务 502（第 1/3 次尝试，将重试）
      · 第 3 轮 llm_call：RuntimeError 模型服务 502（第 2/3 次尝试，将重试）
  工具失败  1 次
      · 第 4 轮 calculator：rejected
```

### 一个跑起来才发现的 bug：`ok` 标志是错的

第一版回放长这样：

```text
   3. ⚙ 工具  calculator({"expression": "1/0"})  0.05ms ok → 工具拒绝执行：除数不能为 0
  工具失败  无
```

被拒绝了却写着 `ok`。根因是 **`execute_tool` 契约上不抛异常**，
而调用方只能靠「有没有抛异常」判断成败 —— 于是「工具被拒绝」和「工具成功」长得一模一样。
**布尔信息被编码进了人类可读的字符串。**

修法：`execute_tool` 改成返回 `ToolResult(text, ok, reason)`，文本从结构里派生。
没有另加一个函数保持旧接口 —— 两个入口必须有同一份判定逻辑，而「同一份逻辑放两处」在 Day 6 已经证明会漂移。

### 验证

```powershell
python -m pytest                                          # 137 passed（Day 7 是 103）
python -m pytest -q -k "no_progress or retry or budget"    # 只看今天的机制
python run_trace.py                                       # 交互式亲眼看
```

12 组探针（把改动逐条还原成 Day 7 行为，看测试接不接得住；一次性实验，未留在仓库）：

| 探针 | 还原成 | 结果 |
| --- | --- | --- |
| 1 | `max_turns` 当条数用 | ✅ 2 条失败 |
| 2 | 关掉无进展检测 | ✅ 3 条失败 |
| 3 | 阈值改成 1 | ✅ 3 条失败 |
| 4 | 签名只看工具名 | ✅ 2 条失败 |
| 5 | 失败从不重试 | ✅ 2 条失败 |
| 6 | 退避改成固定间隔 | ✅ 2 条失败 |
| 7 | `calculator` 不拦 `ZeroDivisionError` | ✅ 3 条失败 |
| 8 | `execute_tool` 不区分 `ValueError` | ✅ 2 条失败 |
| 9 | `find_tool_failures` 退回只看 `kind=="error"` | ✅ 2 条失败 |
| 10 | `error` 渲染不显示重试次数 | ✅ 1 条失败 |
| 11 | `answer` 不渲染中断原因 | ✅ 1 条失败 |
| 12 | 工具结果永远标成成功 | ✅ 1 条失败 |

**12/12 全中**（Day 7 是 6/6，Day 6 是 5/6）。

### 已知问题

| # | 问题 | 计划 |
| --- | --- | --- |
| 1 | 无进展检测只看「完全相同的签名」，挡不住「参数每次变一点」的绕圈 | D9 自我反思 |
| 2 | `AgentSession` 构造参数已有 11 个 | 下一轮改配置时抽 `AgentConfig` |
| 3 | 没采集 `reasoning_content` | 用上推理模型时 |
| 4 | trace 文件无轮转 | 后续 |

> 这一天也**没有**新建 `scripts/` —— 理由见 `doc/day06.md` 第十节。
>
> 📖 三个机制的控制流、状态机、边界条件在 [`../code_view/day08-多步收敛.md`](../code_view/day08-多步收敛.md)。

