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
