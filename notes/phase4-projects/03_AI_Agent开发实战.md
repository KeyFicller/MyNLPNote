# AI Agent 开发实战 - ReAct 智能代理

**项目**: ReAct 模式 AI Agent  
**技术栈**: Python, 类 LangChain 架构  
**任务**: 实现感知-思考-行动-观察循环的智能代理

---

## 1. 什么是 AI Agent

### 1.1 核心定义

AI Agent = **LLM** + **Tools** + **Memory** + **Planning**

区别于传统的单次问答，Agent 可以：
- 自主规划任务步骤
- 调用外部工具获取信息
- 记忆上下文进行多轮交互
- 反思和迭代优化答案

### 1.2 Agent 架构

```
用户输入
    ↓
┌─────────────────────────────────────┐
│         ReAct 循环                  │
│  ┌──────────┐    ┌──────────────┐  │
│  │ Thought  │←──→│    Memory    │  │
│  │   思考   │    │  记忆管理    │  │
│  └────┬─────┘    └──────────────┘  │
│       ↓                              │
│  ┌──────────┐    ┌──────────────┐  │
│  │  Action  │←──→│    Tools     │  │
│  │   行动   │    │   工具调用   │  │
│  └────┬─────┘    └──────────────┘  │
│       ↓                              │
│  ┌──────────┐                       │
│  │Observation│  ← 工具返回结果      │
│  │   观察   │                       │
│  └──────────┘                       │
└─────────────────────────────────────┘
    ↓ (循环直到任务完成)
┌──────────────┐
│ Final Answer │
│   最终回答   │
└──────────────┘
```

---

## 2. ReAct 模式详解

### 2.1 ReAct = Reasoning + Acting

**论文**: "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)

#### 传统方式 vs ReAct 方式

**传统方式**（直接回答）：
```
用户: "2024年诺贝尔文学奖得主是谁？"
AI: 可能是村上春树...（猜测，可能错误）
```

**ReAct 方式**（推理+行动）：
```
Thought: 我需要搜索2024年诺贝尔文学奖的最新信息
Action: search(query="2024年诺贝尔文学奖得主")
Observation: 2024年诺贝尔文学奖授予了韩江(Han Kang)
Final Answer: 2024年诺贝尔文学奖得主是韩国作家韩江
```

### 2.2 ReAct 循环步骤

1. **Thought（思考）**: 分析当前状态，确定下一步行动
2. **Action（行动）**: 选择并调用合适的工具
3. **Observation（观察）**: 获取工具返回的结果
4. **[循环]**: 根据观察结果继续思考，直到获得足够信息
5. **Final Answer（最终回答）**: 整合所有信息，给出答案

---

## 3. 工具系统（Tools）

### 3.1 工具定义

```python
@dataclass
class Tool:
    name: str                    # 工具名称
    description: str             # 工具描述（LLM通过描述理解工具用途）
    parameters: Dict[str, Any]   # 参数定义
    func: Callable               # 执行函数
```

### 3.2 常用工具类型

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| **search** | 搜索引擎 | 查询事实性信息、新闻 |
| **calculator** | 计算器 | 数学运算 |
| **weather** | 天气查询 | 获取天气信息 |
| **code_interpreter** | 代码执行 | 执行Python代码 |
| **database** | 数据库查询 | 结构化数据检索 |
| **api** | API调用 | 调用第三方服务 |

### 3.3 工具调用示例

```python
# 定义工具函数
def search_tool(query: str) -> str:
    """模拟搜索"""
    # 实际应用中调用搜索引擎API
    return f"搜索结果: {query}"

# 注册工具
TOOLS = {
    "search": Tool(
        name="search",
        description="搜索引擎，用于查询事实性信息",
        parameters={"query": "搜索关键词"},
        func=search_tool
    )
}
```

---

## 4. 记忆管理（Memory）

### 4.1 记忆类型

```
┌─────────────────────────────────────┐
│           记忆系统                   │
├─────────────────────────────────────┤
│  短期记忆 (Short-term Memory)        │
│  • 当前对话的上下文                  │
│  • ReAct循环中的思考链               │
│  • 随新对话开始而清空                │
├─────────────────────────────────────┤
│  长期记忆 (Long-term Memory)         │
│  • 用户的偏好设置                    │
│  • 历史对话的关键信息                │
│  • 需要显式存储和检索                │
├─────────────────────────────────────┤
│  工作记忆 (Working Memory)           │
│  • 当前任务的中间结果                │
│  • 工具调用的返回数据                │
│  • 临时计算结果                      │
└─────────────────────────────────────┘
```

### 4.2 记忆管理实现

```python
class Memory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history: List[Dict] = []
        self.user_preferences: Dict[str, Any] = {}
    
    def add_turn(self, role: str, content: str):
        """添加对话轮次"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # 保持最大轮数
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]
    
    def get_context(self) -> str:
        """获取对话上下文"""
        return "\n".join([
            f"{turn['role']}: {turn['content']}"
            for turn in self.conversation_history
        ])
```

---

## 5. ReAct Agent 实现

### 5.1 Agent 核心代码

```python
class ReActAgent:
    def __init__(self, tools, memory, max_iterations=5):
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
    
    def run(self, user_input: str) -> str:
        """执行 ReAct 循环"""
        # 添加到记忆
        self.memory.add_turn("user", user_input)
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # 1. 生成思考过程
            thought = self._generate_thought(user_input)
            
            # 2. 解析 Action
            tool_name, params = self._parse_action(thought)
            
            # 3. 执行工具
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                observation = tool.execute(**params)
                
                # 4. 基于观察继续循环或给出答案
                if "Final Answer:" in thought:
                    return thought.split("Final Answer:")[1].strip()
                
                # 更新当前思考状态
                current_thought = f"{thought}\nObservation: {observation}"
            else:
                # 没有可执行工具，给出默认回答
                return "我需要更多信息来回答这个问题"
        
        return "达到最大迭代次数，未能完成任务"
```

### 5.2 Action 解析

```python
def _parse_action(self, text: str) -> tuple:
    """从文本中解析 Action"""
    import re
    
    # 匹配格式: Action: tool_name(param1=value1, param2=value2)
    pattern = r'Action:\s*(\w+)\s*\(([^)]+)\)'
    match = re.search(pattern, text)
    
    if match:
        tool_name = match.group(1)
        params_str = match.group(2)
        
        # 解析参数
        params = {}
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip().strip('"\'')
        
        return tool_name, params
    
    return None, None
```

---

## 6. 与 LLM 集成

### 6.1 实际部署架构

```
┌─────────────────────────────────────────┐
│              用户层                      │
│         Streamlit / Web / CLI           │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│            Agent 核心层                  │
│  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │ Planning│  │ Memory  │  │ Tools  │ │
│  └────┬────┘  └────┬────┘  └───┬────┘ │
│       └──────────────┼───────────┘      │
│                      ↓                  │
│              ┌──────────────┐            │
│              │  LLM API    │            │
│              │ (GPT-4等)   │            │
│              └──────────────┘            │
└─────────────────────────────────────────┘
```

### 6.2 LLM Prompt 设计

```python
SYSTEM_PROMPT = """你是一个智能助手，可以使用以下工具来完成任务：

{tools_description}

请使用 ReAct 模式（思考-行动-观察）来解决问题：

格式要求：
1. 思考(Thought): 分析问题，制定计划
2. 行动(Action): 选择工具调用
   格式: Action: tool_name(param1=value1, param2=value2)
3. 观察(Observation): 工具返回结果（由系统提供）
4. 最终回答(Final Answer): 任务完成后的总结

重要规则：
- 每次只能调用一个工具
- 如果不需要工具，直接给出 Final Answer
- 最多执行 {max_iterations} 轮行动
"""
```

### 6.3 调用 LLM API 示例

```python
import openai

def call_llm(prompt: str, system_prompt: str) -> str:
    """调用 LLM 生成思考过程"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content
```

---

## 7. 进阶功能

### 7.1 多步推理（Multi-hop Reasoning）

```
用户: "2024年诺贝尔文学奖得主的作品有哪些？"

Step 1:
Thought: 我需要先搜索2024年诺贝尔文学奖得主
Action: search(query="2024年诺贝尔文学奖得主")
Observation: 2024年诺贝尔文学奖授予韩国作家韩江

Step 2:
Thought: 现在我需要搜索韩江的作品
Action: search(query="韩江 韩国作家 作品")
Observation: 韩江代表作包括《素食者》(The Vegetarian)、《少年来了》等

Step 3:
Thought: 我已经获得所需信息
Final Answer: 2024年诺贝尔文学奖得主韩江的代表作包括《素食者》、《少年来了》等
```

### 7.2 自我反思（Self-Reflection）

```python
# 检查结果是否合理
def self_reflection(observation: str, expectation: str) -> bool:
    """判断观察结果是否符合预期"""
    # 使用 LLM 判断结果合理性
    reflection_prompt = f"""
    观察结果: {observation}
    预期目标: {expectation}
    
    这个结果是否满足预期？如果不满足，原因是什么？
    """
    
    reflection = call_llm(reflection_prompt)
    return "满足" in reflection
```

### 7.3 工具链（Tool Chaining）

```python
# 多个工具串联使用
chain = [
    {"tool": "search", "params": {"query": "用户问题"}},
    {"tool": "calculator", "params": {"expression": "{search_result} + 100"}},
    {"tool": "format", "params": {"data": "{calc_result}"}}
]
```

---

## 8. 与 LangChain 对比

### 8.1 本项目 vs LangChain

| 特性 | 本项目（从零实现） | LangChain |
|------|------------------|-----------|
| 学习成本 | 高（理解原理） | 中（学习API） |
| 灵活性 | 极高 | 高 |
| 生产就绪 | 否（教学用） | 是 |
| 工具生态 | 需要自己实现 | 丰富的预置工具 |
| 调试难度 | 容易 | 中等 |

### 8.2 LangChain Agent 快速入门

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 定义工具
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="搜索引擎"
    ),
    Tool(
        name="Calculator",
        func=calculator_function,
        description="计算器"
    )
]

# 创建 Agent
llm = OpenAI(temperature=0)
agent = create_react_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
response = agent_executor.run("2024年诺贝尔文学奖得主是谁？")
```

---

## 9. 实际应用场景

### 9.1 个人助理
- 日程管理 + 天气查询 + 地图导航
- 邮件撰写 + 翻译 + 发送

### 9.2 数据分析
- 数据库查询 + 数据清洗 + 可视化 + 报告生成

### 9.3 代码开发
- 需求分析 + 代码生成 + 测试 + 调试

### 9.4 智能客服
- 意图识别 + 知识库检索 + 工单创建

---

## 10. 未来发展方向

### 10.1 Multi-Agent 系统
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  规划Agent   │────→│  执行Agent   │────→│  审核Agent   │
│  (Planner)  │     │  (Executor) │     │ (Reviewer)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 10.2 自主学习
- 从用户反馈中学习
- 自动发现新工具
- 优化任务规划策略

### 10.3 具身智能
- Agent + 机器人控制
- 物理世界交互

---

## 11. 核心要点总结

### ReAct 循环
```
Thought → Action → Observation → (循环) → Final Answer
思考      行动      观察                    最终答案
```

### Agent 三要素
1. **LLM**: 推理和决策的大脑
2. **Tools**: 扩展能力的手段
3. **Memory**: 保持上下文连贯

### 关键代码模板
```python
# 1. 定义工具
Tool(name, description, parameters, func)

# 2. 初始化 Agent
agent = ReActAgent(tools, memory, max_iterations)

# 3. 运行
result = agent.run(user_input)

# 4. ReAct 循环内部
while not done and iterations < max:
    thought = generate_thought(context)
    action = parse_action(thought)
    observation = execute_tool(action)
    update_context(thought, action, observation)
```

---

**实践项目文件**: `examples/llm-apps/03_ai_agent.py`

**运行命令**:
```bash
python examples/llm-apps/03_ai_agent.py
```

**推荐资源**:
- ReAct 论文: https://arxiv.org/abs/2210.03629
- LangChain Agents: https://python.langchain.com/docs/modules/agents
- AutoGPT: https://github.com/Significant-Gravitas/AutoGPT
