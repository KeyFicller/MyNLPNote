# AI Agent 接入真实 LLM API 实战

**项目**: 接入 GPT-4/Claude/国产大模型的智能代理  
**技术栈**: Python, OpenAI API, 多平台LLM  
**任务**: 实现具备真实推理能力的 ReAct Agent

---

## 1. 为什么需要接入真实API

### 1.1 模拟 vs 真实LLM

| 对比项 | 模拟LLM（规则-based） | 真实LLM（GPT-4/Claude） |
|--------|---------------------|------------------------|
| 推理能力 | ❌ 固定规则匹配 | ✅ 强大的语言理解和推理 |
| 泛化能力 | ❌ 只能处理预设场景 | ✅ 应对各种新问题 |
| 工具选择 | ❌ 简单的关键词匹配 | ✅ 理解任务需求，智能选择工具 |
| 回答质量 | ❌ 模板化回复 | ✅ 自然流畅，针对性强 |
| 成本 | ✅ 免费 | 💰 按token计费 |

### 1.2 真实LLM的优势

```
用户: "我想了解最近发布的AI大模型，有什么值得关注的吗？"

模拟Agent:
Thought: 用户询问AI大模型，我需要搜索
Action: search(query="AI大模型")  ← 无法理解"最近发布"的时间概念

真实LLM (GPT-4):
Thought: 用户想了解最近发布的AI大模型，这是一个时效性问题。
我需要搜索2024年最新发布的AI大模型信息，包括国内外的进展。
Action: search(query="2024年最新发布的大语言模型 GPT-4 Claude")
← 理解"最近"、"值得关注"的含义，自动优化搜索词
```

---

## 2. 支持的LLM平台

### 2.1 平台对比

| 平台 | 推荐模型 | 特点 | 价格（每1K tokens） |
|------|----------|------|-------------------|
| **OpenAI** | gpt-4, gpt-3.5-turbo | 推理最强，生态完善 | $0.01-0.03 |
| **Anthropic** | claude-3-opus/sonnet | 上下文长(200K)，安全性高 | $0.008-0.015 |
| **智谱AI** | glm-4, glm-4-flash | 中文友好，速度快 | ¥0.005-0.1 |
| **DeepSeek** | deepseek-chat/coder | 性价比高，代码强 | ¥0.001-0.002 |
| **Moonshot** | moonshot-v1-8k/128k | 上下文超长(200K) | ¥0.006-0.012 |

### 2.2 API接入方式

#### OpenAI
```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ]
)
print(response.choices[0].message.content)
```

#### 智谱AI / DeepSeek / Moonshot (OpenAI兼容)
```python
from openai import OpenAI

# 这些平台都提供OpenAI兼容的API
client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1"  # 或智谱、Moonshot的URL
)

# 调用方式完全相同
response = client.chat.completions.create(
    model="deepseek-chat",  # 或 glm-4, moonshot-v1-8k
    messages=[...]
)
```

#### Claude
```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=2000,
    messages=[{"role": "user", "content": "你好"}]
)
print(response.content[0].text)
```

---

## 3. API Key 管理

### 3.1 环境变量（推荐）

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-xxxxxxxx"
$env:ZHIPU_API_KEY = "your-zhipu-key"
$env:DEEPSEEK_API_KEY = "your-deepseek-key"

# 验证
python -c "import os; print(os.environ.get('OPENAI_API_KEY'))"
```

### 3.2 .env 文件

创建 `.env` 文件：
```
OPENAI_API_KEY=sk-xxxxxxxx
ZHIPU_API_KEY=your-zhipu-key
DEEPSEEK_API_KEY=your-deepseek-key
```

加载：
```python
from dotenv import load_dotenv
load_dotenv()  # 自动加载.env文件
```

### 3.3 安全注意事项

⚠️ **永远不要**：
- 将API Key提交到Git仓库
- 在代码中硬编码API Key
- 将API Key分享给他人
- 在前端代码中使用API Key

✅ **推荐做法**：
- 使用环境变量
- 使用.gitignore排除.env文件
- 定期轮换API Key
- 使用API Key管理服务（如AWS Secrets Manager）

---

## 4. 代码架构

### 4.1 整体架构

```
┌───────────────────────────────────────────┐
│                用户层                      │
│          CLI / Web / GUI                 │
└──────────────────┬──────────────────────┘
                   ↓
┌───────────────────────────────────────────┐
│              ReAct Agent                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Thought │  │  Action │  │  Memory │  │
│  │  思考   │  │  行动   │  │  记忆   │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  │
│       └──────────────┼─────────────┘       │
│                      ↓                    │
│              ┌──────────────┐              │
│              │  LLM Client │              │
│              │  (多平台支持) │              │
│              └──────┬──────┘              │
└─────────────────────┼─────────────────────┘
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ OpenAI  │  │ Claude  │  │  国产   │
   │  GPT-4  │  │  Opus   │  │  GLM-4  │
   └─────────┘  └─────────┘  └─────────┘
```

### 4.2 LLM Client 设计

```python
class LLMClient:
    """
    统一的LLM客户端，支持多平台
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = self._get_api_key()
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """
        统一的聊天接口
        
        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
            **kwargs: 额外参数如 temperature, max_tokens
        
        Returns:
            LLM生成的文本
        """
        # 根据平台调用不同API
        if self.config.provider == LLMProvider.OPENAI:
            return self._call_openai(messages, **kwargs)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._call_claude(messages, **kwargs)
        # ...
```

### 4.3 成本监控

```python
def _calculate_cost(self, tokens: int) -> float:
    """估算API调用成本"""
    prices = {
        "gpt-4": 0.03,           # $0.03 / 1K tokens
        "gpt-3.5-turbo": 0.002,  # $0.002 / 1K tokens
        "glm-4": 0.001,          # ¥0.001 / 1K tokens (约)
        "deepseek-chat": 0.0005, # ¥0.0005 / 1K tokens
    }
    price = prices.get(self.config.model_name, 0.01)
    return (tokens / 1000) * price
```

---

## 5. ReAct 循环实现

### 5.1 完整流程

```python
def run(self, user_input: str) -> str:
    """
    执行 ReAct 循环
    """
    # 1. 构建系统提示词（包含工具描述）
    messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    iteration = 0
    while iteration < self.max_iterations:
        iteration += 1
        
        # 2. 调用LLM生成思考
        response = self.llm.chat(messages)
        
        # 3. 检查是否已有最终答案
        if "Final Answer:" in response:
            return extract_final_answer(response)
        
        # 4. 解析 Action
        tool_name, params = parse_action(response)
        
        if tool_name in self.tools:
            # 5. 执行工具
            tool = self.tools[tool_name]
            observation = tool.execute(**params)
            
            # 6. 将观察结果加入上下文
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # 工具不存在，提示LLM重新思考
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "工具不存在，请使用可用工具或直接给出 Final Answer"})
    
    return "达到最大迭代次数，未能完成任务"
```

### 5.2 Prompt 工程

#### 系统提示词模板

```
你是一个智能助手，可以使用以下工具来完成任务：

可用工具：
- search(query="搜索关键词"): 网络搜索引擎，用于获取实时信息
- calculator(expression="数学表达式"): 数学计算器，支持加减乘除
- python(code="Python代码"): Python代码执行器，用于数据处理
- datetime(): 获取当前时间

请使用 ReAct 模式（思考-行动-观察）来解决问题：

格式要求：
1. Thought: 分析当前情况，思考下一步该怎么做
2. Action: 如果需要使用工具，格式为：Action: tool_name(param="value")
3. Observation: 工具返回的结果（由系统自动提供）
4. Final Answer: 当你获得足够信息后，给出最终回答

示例：
用户：北京今天天气怎么样？
Thought: 用户询问北京今天的天气，我需要使用天气查询工具。
Action: weather(location="北京")
Observation: 北京天气: 晴天, 气温 25°C
Thought: 我已经获得天气信息，可以直接回答了。
Final Answer: 北京今天天气晴朗，气温25°C，适合出行！
```

---

## 6. 工具系统增强

### 6.1 网络搜索（DuckDuckGo）

```python
def search_tool(query: str) -> str:
    """
    使用DuckDuckGo搜索引擎
    优点：无需API Key，免费
    """
    from duckduckgo_search import DDGS
    
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"[{i}] {result['title']}\n{result['body'][:200]}...\n")
    
    return "\n".join(output)
```

### 6.2 Python 代码执行（安全沙箱）

```python
def python_tool(code: str) -> str:
    """
    在沙箱环境中执行Python代码
    """
    # 禁止危险操作
    forbidden = ['import os', 'import sys', 'open(', '__import__', 
                 'eval(', 'exec(', 'subprocess', 'os.system']
    
    for f in forbidden:
        if f in code:
            return f"安全限制: 不允许使用 '{f}'"
    
    # 受限的执行环境
    safe_globals = {
        "__builtins__": {
            "print": print,
            "len": len, "range": range,
            "sum": sum, "max": max, "min": min,
            # ... 其他安全函数
        }
    }
    
    import io, contextlib
    output_buffer = io.StringIO()
    
    with contextlib.redirect_stdout(output_buffer):
        exec(code, safe_globals, {})
    
    return output_buffer.getvalue()
```

### 6.3 工具调用示例

```
用户：计算前100个质数的和

LLM思考：
Thought: 这是一个数学问题，我可以使用Python代码来计算前100个质数的和。
Action: python(code="""
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

primes = [n for n in range(2, 1000) if is_prime(n)][:100]
print(f'前100个质数: {primes}')
print(f'和: {sum(primes)}')
""")

Observation: 前100个质数: [2, 3, 5, 7, 11, 13, ...]
和: 24133

Final Answer: 前100个质数的和是 24133
```

---

## 7. 实际应用场景

### 7.1 智能客服

```python
# 工具：知识库检索 + 订单查询 + 退款处理
agent = ReActAgent(
    tools=[
        knowledge_base_search,  # 检索FAQ
        order_query,          # 查询订单状态
        refund_process        # 处理退款
    ],
    llm=llm_client
)

# 用户：我的订单怎么还没到？
# Thought: 用户询问订单状态，我需要先查询订单信息
# Action: order_query(order_id="用户提供的订单号")
```

### 7.2 数据分析助手

```python
# 工具：SQL查询 + 数据分析 + 可视化
tools = [
    sql_query,      # 数据库查询
    python_exec,    # pandas数据处理
    plot_chart      # 生成图表
]

# 用户：帮我分析最近一个月的销售额趋势
# Thought: 需要从数据库查询销售数据，然后用Python分析
# Action: sql_query(sql="SELECT date, amount FROM sales WHERE date >= DATE_SUB(NOW(), INTERVAL 1 MONTH)")
```

### 7.3 编程助手

```python
# 工具：代码生成 + 代码执行 + 调试
tools = [
    code_search,      # 搜索相关代码示例
    code_generate,    # 生成代码
    code_execute,     # 执行测试
    debug_assist      # 调试建议
]
```

---

## 8. 性能优化

### 8.1 减少Token消耗

```python
# 1. 截断过长的观察结果
def truncate_observation(text: str, max_len: int = 500) -> str:
    if len(text) > max_len:
        return text[:max_len] + "... [内容已截断]"
    return text

# 2. 限制对话轮数
max_iterations = 5  # 避免无限循环

# 3. 使用更便宜的模型处理简单任务
def route_by_complexity(query: str) -> str:
    if is_simple_query(query):
        return "gpt-3.5-turbo"  # 便宜
    else:
        return "gpt-4"  # 能力强
```

### 8.2 缓存机制

```python
import hashlib

class LLMCache:
    """LLM响应缓存"""
    
    def __init__(self):
        self.cache = {}
    
    def get_key(self, messages):
        """生成缓存key"""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages):
        key = self.get_key(messages)
        return self.cache.get(key)
    
    def set(self, messages, response):
        key = self.get_key(messages)
        self.cache[key] = response
```

---

## 9. 常见问题

### Q1: API Key泄露怎么办？

**立即**：
1. 在平台后台撤销该API Key
2. 生成新的API Key
3. 更新环境变量
4. 检查Git历史，删除泄露的提交

### Q2: Token消耗太快？

**优化**：
1. 限制max_tokens参数
2. 使用更便宜的模型（GPT-3.5代替GPT-4）
3. 缓存常见问题的回答
4. 截断过长的工具返回结果

### Q3: LLM不按照ReAct格式输出？

**解决**：
1. 优化Prompt，给出更多示例
2. 调整temperature参数（降低随机性）
3. 使用Few-shot prompting
4. 添加输出格式约束

### Q4: 工具调用失败？

**排查**：
1. 检查工具名称和参数是否正确
2. 查看工具执行的错误日志
3. 确保LLM理解工具用途
4. 添加错误处理逻辑

---

## 10. 核心要点总结

### 架构流程
```
用户输入 → ReAct Agent → LLM Client → 真实LLM API
                ↓              ↓
            工具调用 ← 响应解析 ← 生成回复
                ↓
           观察结果 → 继续循环 → 最终答案
```

### 关键代码模板
```python
# 1. 配置API Key
import os
api_key = os.environ.get("OPENAI_API_KEY")

# 2. 初始化LLM客户端
from openai import OpenAI
client = OpenAI(api_key=api_key)

# 3. 定义工具
@dataclass
class Tool:
    name: str
    description: str
    func: Callable

# 4. ReAct循环
for i in range(max_iterations):
    # 调用LLM
    response = client.chat.completions.create(...)
    
    # 解析Action
    tool_name, params = parse_action(response)
    
    # 执行工具
    if tool_name in tools:
        observation = tools[tool_name].execute(**params)
        messages.append(f"Observation: {observation}")

# 5. 获取最终答案
return extract_final_answer(response)
```

### 最佳实践
1. ✅ 使用环境变量管理API Key
2. ✅ 监控系统token消耗和成本
3. ✅ 实现错误处理和降级策略
4. ✅ 添加缓存减少重复调用
5. ✅ 优化Prompt提高响应质量

---

**实践项目文件**: `examples/llm-apps/04_ai_agent_with_api.py`

**运行命令**:
```bash
# 设置API Key
$env:OPENAI_API_KEY="sk-xxxxxxxx"

# 运行
python examples/llm-apps/04_ai_agent_with_api.py
```

**推荐资源**:
- OpenAI文档: https://platform.openai.com/docs
- Claude文档: https://docs.anthropic.com
- 智谱AI: https://open.bigmodel.cn/dev/howuse/glm-4
- DeepSeek: https://platform.deepseek.com/api-docs
