# Model Service Evaluation Summary

> Updated: 2026-03-22

## 说明

- `可启动`：vLLM Docker 服务可拉起。
- `可服务`：可通过 OpenAI-compatible`/v1/chat/completions` 提供稳定响应。
- `智能体能跑`：能承载 Goose / tool use / agentic 场景。
- `可交付`：在当前环境与目标下，值得进入下一阶段集成。
- `最低硬件建议`：优先采用官方 serving 示例；官方未给出时，按模型规模、精度类型和当前测试目标做保守工程推断，不等同于厂商承诺下限。

## 总表

| 模型                        | 首次公开发布时间 | 最低硬件配置需求                                                                            | 可启动 | 可服务 | 性能结论                                                         | 智能体能跑 | 可交付 | 备注                                                                                      |
| --------------------------- | ---------------- | ------------------------------------------------------------------------------------------- | ------ | ------ | ---------------------------------------------------------------- | ---------- | ------ | ----------------------------------------------------------------------------------------- |
| Qwen3.5-35B-A3B             | 2026-02-24       | **不建议 V100；最低建议 Ampere，4×32GB**                                             | NO     | NO     | 未进入有效测试                                                   | 未验证     | NO     | 官方示例口径偏 TP=8，这类新一代 A3B/MoE 模型更适合 Ampere/Hopper 及以上，当前阶段淘汰     |
| Qwen3.5-27B                 | 2026-02-24       | **不建议 V100；最低建议 Ampere，4×32GB**                                             | NO     | NO     | 未进入有效测试                                                   | 未验证     | NO     | 同上，当前阶段淘汰                                                                        |
| GLM4.7-Flash                | 2025-08-08       | **不建议 V100；最低建议 Ampere，4×32GB**                                             | NO     | NO     | 未进入有效测试                                                   | 未验证     | NO     | 这类新模型更应按 BF16/Ampere 起步来写，当前阶段淘汰                                       |
| GPT-OSS-20B                 | 2025-08-05       | **若走官方低比特路径：最低建议 Hopper，1×80GB；若走 BF16：最低建议 Ampere，2×32GB** | NO     | NO     | 未进入有效测试                                                   | 未验证     | NO     | 官方默认强调 MXFP4，Hopper 或更新架构支持更合适；若退回 BF16，显存需求会明显增大          |
| Cohere Command-R-08-2024    | 2024-08-30       | **V100 可用；最低建议 4×32GB**                                                       | YES    | YES    | 好；基础服务稳定，输出 TPS 约 45                                 | NO         | NO     | 这一项最适合保留成“V100 可服务”的正例；Cohere 也明确说 08-2024 版硬件占用相比上一版下降 |
| Qwen3-30B-A3B-Thinking-2507 | 2025-07-29       | **V100 可启动但不建议；最低建议 Ampere，4×32GB；更优 Hopper**                        | YES    | YES    | 差；当前 vLLM + V100 上 MoE 路径异常慢，8K 左右首 token 超过 60s | YES        | NO     | 不建议在当前 V100 环境继续投入                                                            |
| Qwen3-32B                   | 2025-04-29       | **V100 可用；最低建议 4×32GB；更优 2×80GB / 4×40GB Ampere**                        | YES    | YES    | 中；长上下文下 TTFT 约 8s-16s，输出 TPS 约 17                    | YES        | YES    | 当前最均衡的候选                                                                          |

## 关键结论

- 当前唯一建议继续推进的模型是`Qwen3-32B`。
- `Cohere Command-R-08-2024` 适合普通服务，不适合 Goose / tools。
- `Qwen3-30B-A3B-Thinking-2507` 的主要问题更像是当前`vLLM + V100` 组合下的 MoE 路径性能异常，而不是单纯的 thinking 输出成本；在当前环境不建议继续投入。

## 最终建议

- 主选：`Qwen3-32B`
- 备选：`Cohere Command-R-08-2024`，仅限不依赖 Goose / tools 的普通服务场景
- 暂停：`Qwen3-30B-A3B-Thinking-2507` 及其余未进入有效测试或未通过启动验证的模型

## 来源

- Qwen3.5 模型页与 vLLM / SGLang serving 示例：[Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 、[Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- Qwen3 主模型页：[Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
- Qwen3 MoE Thinking 模型页：[Qwen/Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)
- Cohere 模型页：[CohereLabs/c4ai-command-r-08-2024](https://huggingface.co/CohereLabs/c4ai-command-r-08-2024)
- OpenAI 开源模型页：[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- GLM 模型页：[zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
