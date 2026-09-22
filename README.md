# Agent Demo

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

### 已知待办

| 项 | 说明 |
| --- | --- |
| 记忆仍是硬截断 | 被裁掉的对话永久丢失，待换成摘要压缩（把旧对话总结成 `<summary>` 常驻上下文） |
| `max_messages` 语义不清 | 实际只按「条数」保留，`max_messages=10` 只有 5 轮，建议改为 `max_turns` |
| 请求失败会留下孤儿 `user` 消息 | `add_user_message` 在调用模型之前执行，异常后消息留在记忆里；应把写入推迟到成功之后 |
| 上一轮工具结果不可见 | `role="tool"` 不进记忆，模型只能从自然语言答案里读结果 |
| `max_tool_rounds=3` 偏小 | 每轮 = 1 次模型请求，实际只够 2 次工具调用，建议改为 5~6 |
| 观察 | `tool_choice="auto"` 下，模型会对 `1+1` 这类简单算式直接回答而不调用 `calculator` |
