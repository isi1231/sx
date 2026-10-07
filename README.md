# Agent Demo

每天的学习讲解、流程图和实测证据在 [`../doc/`](../doc/README.md)：
`day01-02` 多轮对话 · `day03` Prompt 工程 · `day04` 工具调用 · `day05` 记忆机制 · `day06` 上下文工程。

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
| `max_messages` 语义不清 | 实际只按「条数」保留，`=10` 只有 5 轮 | ⬜ 待改 `max_turns` |
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
| 1 | `max_messages` 语义仍是「条数」 | D7 改 `max_turns` |
| 2 | `_message_to_dict` 三种分支返回的 dict 形状不一致 | D7 |
| 3 | 摘要本身没有长度硬约束，只靠 prompt 里的「不超过 N 字」 | D7 加代码侧截断 |
| 4 | `max_tool_rounds=3` 只够 2 次工具调用 | D8 |
| 5 | `calculator` 会泄漏 `OverflowError` / `ZeroDivisionError` | D8 |
| 6 | 压缩发生在 `add()` 里，是**同步阻塞**的（一次压缩 = 一次模型请求） | D15 异步化 |

第 6 条是 Day 6 新引入的真实代价：用户提问触发压缩时，这一轮会多出一次模型调用的延迟。
现在能接受（几分钟一次），但要做成服务就必须异步，否则会拖慢 P99。

