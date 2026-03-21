# model-service-ctl

用于启动 vLLM 模型服务，并通过 OpenAI 兼容接口对模型服务做长上下文基准测试。

## Repository

- GitHub: [buyangnie/model-service-ctl](https://github.com/buyangnie/model-service-ctl)

## Files

- [`bench_model_service.py`](/Users/buyangnie/Documents/GitHub/model-service-ctl/bench_model_service.py)
  OpenAI 兼容接口测试脚本。支持 `smoke`、`bench`、`all` 三种模式，支持流式输出统计、首 token 超时控制、Markdown 报告输出、环境基线采集。
- [`model_service_testset.json`](/Users/buyangnie/Documents/GitHub/model-service-ctl/model_service_testset.json)
  `bench` 模式使用的测试集。当前是单条长 ReAct agent trace，包含 20 次工具调用，按 milestone 逐步累积上下文。
- [`model_service_ctl_c4ai_command_r_08_2024.sh`](/Users/buyangnie/Documents/GitHub/model-service-ctl/model_service_ctl_c4ai_command_r_08_2024.sh)
  `c4ai-command-r-08-2024` 的 vLLM Docker 启动脚本。
- [`model_service_ctl_qwen3.5_27B.sh`](/Users/buyangnie/Documents/GitHub/model-service-ctl/model_service_ctl_qwen3.5_27B.sh)
  Qwen3.5 27B 的 vLLM Docker 启动脚本。

## Benchmark Script

[`bench_model_service.py`](/Users/buyangnie/Documents/GitHub/model-service-ctl/bench_model_service.py) 走的是 OpenAI 兼容接口：

- `POST /v1/chat/completions`
- `stream=true`
- 标准 `messages / model / temperature / max_tokens` 请求体

脚本会统计：

- `TTFT`
- `Total`
- `Decode`
- `Completion Tokens`
- `Output TPS`
- `GPU Util`
- `Prefix Cache` 相关指标
- 主机环境基线

### 默认行为

- 默认首 token 超时：`60s`
- 默认总请求超时：`1800s`
- 默认输出 Markdown 报告，文件名包含模型名和时间戳
- `bench` 模式从 [`model_service_testset.json`](/Users/buyangnie/Documents/GitHub/model-service-ctl/model_service_testset.json) 读取长 agent trace，并按 milestone 做前缀累积测试

## Usage

只跑 `bench`：

```bash
python3 bench_model_service.py \
  --mode bench \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key change-me \
  --models c4ai-command-r-08-2024 \
  --testset ./model_service_testset.json
```

只跑 `smoke`：

```bash
python3 bench_model_service.py \
  --mode smoke \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key change-me \
  --models c4ai-command-r-08-2024
```

全部都跑：

```bash
python3 bench_model_service.py \
  --mode all \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key change-me \
  --models c4ai-command-r-08-2024 \
  --testset ./model_service_testset.json
```

设置首 token 超时：

```bash
python3 bench_model_service.py \
  --mode bench \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key change-me \
  --models c4ai-command-r-08-2024 \
  --testset ./model_service_testset.json \
  --first-token-timeout 60
```

指定输出报告文件：

```bash
python3 bench_model_service.py \
  --mode bench \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key change-me \
  --models c4ai-command-r-08-2024 \
  --testset ./model_service_testset.json \
  --markdown-output ./report_c4ai.md
```

查看帮助：

```bash
python3 bench_model_service.py --help
```

## Output

报告内容包括：

- Test Config
- Tokenizer Status
- Environment Baseline
- Model Summary
- Smoke Results
- Bench Results
- Raw Results

默认报告文件名类似：

```text
model_service_benchmark_report_c4ai-command-r-08-2024_20260321_124240.md
```

## Notes

- 当前 `bench` 主要用于测长上下文输入侧成本，最终输出被压到极短，便于隔离 `prefill` 延迟。
- 如果当前环境没有安装 `transformers`，报告里的 token 统计为近似值，不是精确 tokenizer 真值。
- `bench` 当前结果会受到 vLLM prefix cache 命中的影响，更接近“持续 agent 会话”场景，而不是完全冷启动场景。
