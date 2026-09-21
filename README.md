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
