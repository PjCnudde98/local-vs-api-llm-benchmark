# Model lineup, hardware and serving

Which models the experiment runs, where each one runs, and how to serve it. The lineup is
configured in `config/variants.yaml`; `cexp variants` prints every runnable variant.
Checked against the Hugging Face model cards on 2026-11-09. Lock exact versions at gate 1
(see "Locking versions" below): this landscape changes monthly.

## Size classes

| Size class | Model | Params (total / active) | Runs on | Cost profile |
|---|---|---|---|---|
| Cloud | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | n/a | Microsoft Foundry or Claude API | per token |
| Cloud | Claude Haiku 5.5 (`claude-haiku-5-5`) | n/a | Microsoft Foundry or Claude API | per token |
| Cloud | Newest mid-tier GPT in Azure EU Data Zone Standard | n/a | Azure OpenAI | per token |
| XL local | GLM-5.3 | 753B MoE | XL server (512 GB RAM + GPUs) | `xl-server-512gb` |
| XL local | DeepSeek-V4.1-Flash-1017 | 763B MoE | XL server | `xl-server-512gb` |
| XL local | Kimi K2.6 | 1T / 32B | XL server | `xl-server-512gb` |
| XL local | Qwen3.8-Flash-Next | 125B / 6B (+51B n-gram embeddings) | XL server | `xl-server-512gb` |
| Large local | Mistral Small 4 | 119B / ~6.5B | 80 GB+ GPU or two GPUs | `gpu-80gb` |
| Medium local | Qwen3.6-27B | 27B dense | one 24-48 GB GPU | `gpu-48gb` |
| Medium local | Gemma 4 26B A4B | 26B / 4B | one 24 GB GPU | `gpu-48gb` |
| Medium local | gpt-oss-20b | 21B / 3.6B | one 16-24 GB GPU | `gpu-48gb` |
| Small local | Gemma 4 12B | 12B dense | 16 GB GPU or fast CPU | `laptop` |
| Small local | Qwen3 8B | 8B dense | 16 GB GPU or fast CPU | `laptop` |

Every local model runs on the `local` route only. Every cloud model runs on three routes:
`pseudonymise`, `abstract` and `raw` (synthetic data only). Every model runs with reasoning
off (`local/qwen3-8b`) and on (`local/qwen3-8b+think`). That makes 40 variants, plus the
hybrid and the FAKE dry-run variant.

## XL class: the >128 GB server

The XL class answers one question: how far does the best open-weight model of each major
family get on the ladder? Each family's flagship is far larger than the server, so the class
uses the best model of each family that fits:

| Family | Newest flagship (not used) | Its size | Used instead | Build | On disk |
|---|---|---|---|---|---|
| Kimi | Kimi K3.1 Pro (2.8T, 104B active) | ~1.56 TB (MXFP4) | Kimi K2.6 | `unsloth/Kimi-K2.6-GGUF` UD-Q2_K_XL | 340 GB |
| GLM | GLM-5.3 (753B) | ~755 GB (FP8) | GLM-5.3, quantised | `unsloth/GLM-5.3-GGUF` UD-Q4_K_XL | 467 GB |
| DeepSeek | DeepSeek-V4-Pro-0813 (1.65T) | ~893 GB | DeepSeek-V4.1-Flash-1017 (newest release) | `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` UD-Q4_K_XL | 501 GB |
| Qwen | Qwen3.8-2.4T-A95B | ~2.4 TB (FP8) | Qwen3.8-Flash-Next | `unsloth/Qwen3.8-Flash-Next-GGUF` Q8_0 | 188 GB |

What this means for the results:

- **Kimi runs at about 2 bits.** Its native INT4 weights are 595 GB and do not fit. Kimi is
  therefore tested at a lower precision than the other three. If it fails a tier the others
  pass, quantisation is a likely cause. Note it in the readout.
- **GLM-5.3 and DeepSeek need GPU memory on top of the 512 GB of RAM.** With 467 GB and 501 GB
  of weights, the KV cache and the OS need the GPUs' VRAM. If loading fails, step down one
  quant (GLM UD-Q3_K_XL is 343 GB) and record the quant you used.
- **One XL model fits at a time.** Run them one after another: `cexp run --variants 'local/glm-5.3*'`,
  restart the server with the next model, and so on.
- **Latency depends on how many experts sit in RAM.** Experts read from system RAM are much
  slower than experts in VRAM. Record the GPU model and count, RAM speed and the offload
  setting, because they decide whether the p95 gate can pass. Run `cexp throughput` for each
  XL model so the cost model uses measured capacity.
- **Licences differ.** DeepSeek is MIT. GLM-5.3, Kimi K2.6 and Qwen3.8-Flash-Next have custom
  licences, which may restrict large model-as-a-service use. Check them before a production
  recommendation.

### Serving on the XL server

llama.cpp's `llama-server` exposes the OpenAI-compatible API the harness uses. It can keep
attention and shared layers on the GPUs while MoE experts stay in RAM. A starting point
(check the flags against your llama.cpp version):

```bash
llama-server -m GLM-5.3-UD-Q4_K_XL-00001-of-000NN.gguf --alias glm-5.3 \
  --host 0.0.0.0 --port 8080 --jinja \
  -ngl 999 --n-cpu-moe 60 \
  -c 65536 -np 4
```

- `--alias` must match the model name in `.env` (`XL_GLM_MODEL`, default `glm-5.3`).
- `--jinja` applies the model's chat template, which carries the thinking switch (`chat_template_kwargs`).
- `--n-cpu-moe N` keeps the experts of the first N layers in RAM. Lower it until VRAM is full.
- `-np 4` gives 4 parallel slots; `-c` is shared across them (16K tokens each here, enough for tier 6).

SGLang + KTransformers and vLLM are alternatives with better throughput when more of the model
fits in VRAM. The GLM-5.3 and Qwen3.8 model cards link recipes.

### Thinking switches

`config/variants.yaml` sends these per variant. They come from the model cards; send one
request in each mode before a scored run and check that thinking really is on or off (look at
the completion tokens).

| Model | Off | On | Source |
|---|---|---|---|
| Kimi K2.6 | `chat_template_kwargs.thinking: false` ("instant mode") | `true` | model card |
| GLM-5.3 | cannot be turned off: `reasoning_effort: low` | `reasoning_effort: high` (default `max`) | model card; also send `clear_thinking: true` for multi-turn |
| DeepSeek-V4.1-Flash-1017 | `chat_template_kwargs.thinking: false` | `true` | no Jinja template ships with the weights; check your server's DeepSeek-V4 support |
| Qwen3.8-Flash-Next, Qwen3.6-27B | `chat_template_kwargs.enable_thinking: false` | `true` | model card |
| Gemma 4 | `chat_template_kwargs.enable_thinking: false` | `true` | check the model card |
| Mistral Small 4, Qwen3 8B (Ollama) | `reasoning_effort: none` | `high` | Ollama/vLLM |
| gpt-oss-20b | always reasons: `reasoning_effort: low` | `high` | model card |
| Claude Sonnet 5.5 | `thinking: {type: between_tools}` (disabled is rejected) | `{type: adaptive}` | Claude API docs |
| Claude Haiku 5.5 | `thinking: {type: disabled}` | `{type: adaptive}` | Claude API docs |
| GPT (Azure) | `reasoning_effort: none` | `medium` | depends on the deployment |

## Cloud models

- **Claude** goes through the Anthropic SDK. Set `ANTHROPIC_FOUNDRY_RESOURCE` and
  `ANTHROPIC_FOUNDRY_API_KEY` for Microsoft Foundry, or `ANTHROPIC_API_KEY` for the Claude API.
  Claude in Foundry offers only Global and US Data Zone deployments. That is acceptable for
  synthetic and preprocessed data. For a production recommendation, also evaluate Claude
  through AWS Bedrock or Google Vertex AI in EU regions.
- The Claude API has rejected `temperature` on every model since 2026-10-21, so Claude variants
  run at the provider default. The GPT reasoning models reject it too. All other variants send
  0.2. This is the only generation setting that differs.
- JSON output is asked for in the prompt for every model. OpenAI-compatible servers also get
  `response_format: json_object`; Claude gets no schema, so the instruction text stays
  identical. Parse failures are reported per variant.
- Server-side refusal fallbacks are deliberately off. They would answer with a different model
  and blur the comparison.

## Locking versions

At gate 1, write down for every model: the exact repository and revision (commit hash) or API
model id, the quant file, the serving stack and version, the server flags and the hardware.
Every run row also stores the model id the server reported (`runs.model_id`), so a silent
model swap shows up in the report.
