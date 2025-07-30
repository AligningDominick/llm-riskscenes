# 🚀 快速启动指南

欢迎使用 LLM 多语言安全评估框架！本指南将帮助您在 5 分钟内开始使用。

## 1️⃣ 安装（1分钟）

```bash
# 进入项目目录
cd C:/llm-multilingual-safety-eval

# 安装框架
pip install -e .
```

## 2️⃣ 运行演示（2分钟）

```bash
# 运行快速演示
python demo.py
```

这将展示框架的基本功能，包括：
- 加载模型
- 运行安全评估
- 分析结果
- 生成报告

## 3️⃣ 第一次真实评估（2分钟）

### 选项 A: 使用命令行

```bash
# 评估 Claude 模型
lmse evaluate --model claude-3-opus --languages chinese --domains healthcare

# 注意：需要先设置 API 密钥
export ANTHROPIC_API_KEY="your-api-key"
```

### 选项 B: 使用 Python

```python
from lmse import SafetyEvaluator, ModelLoader

# 加载模型
model = ModelLoader.load("claude-3-opus", api_key="your-key")

# 运行评估
evaluator = SafetyEvaluator("configs/default.yaml")
results = evaluator.evaluate(model, languages=["chinese"])

# 查看结果
print(f"安全评分: {results['safety_score'].mean():.1f}")
```

## 📊 查看结果

评估完成后，您可以：

1. **查看 CSV 结果**
   ```
   results/evaluation_*.csv
   ```

2. **打开 HTML 报告**
   ```
   results/report_*.html
   ```

3. **使用 Jupyter 笔记本**
   ```bash
   jupyter lab visualizations/analysis_dashboard.ipynb
   ```

## 🎯 下一步

1. **探索更多语言**
   ```bash
   lmse list-scenarios --format table
   ```

2. **比较多个模型**
   ```bash
   python examples/model_comparison.py
   ```

3. **自定义评估**
   - 编辑 `configs/default.yaml`
   - 添加自己的场景到 `datasets/scenarios/`

## ❓ 需要帮助？

- 📚 查看完整文档: `docs/`
- 💡 参考示例代码: `examples/`
- 🐛 报告问题: 创建 GitHub Issue

---

**提示**: 使用 `lmse --help` 查看所有可用命令！

祝您评估愉快！ 🎉