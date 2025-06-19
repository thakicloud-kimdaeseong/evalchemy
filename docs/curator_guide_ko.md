# 큐레이터(Curator) 사용 가이드

[Curator](https://github.com/bespokelabsai/curator/)를 통해 API 기반 모델을 평가할 수 있습니다. `--model curator` 옵션을 사용하면 [LiteLLM](https://docs.litellm.ai/docs/providers)에서 지원하는 모든 모델을 포함하여 더욱 다양한 API 기반 모델로 평가를 진행할 수 있습니다.

다음은 `curator` 모델을 사용하는 기본 예시입니다.
```bash
python -m eval.eval \
      --model curator  \
      --tasks AIME24,MATH500,GPQADiamond \
      --model_name "gemini/gemini-2.0-flash-thinking-exp-01-21" \
      --apply_chat_template False \
      --model_args 'tokenized_requests=False' \
      --output_path logs
```

## Curator 상세 사용법

Curator를 사용하면 API 기반 모델의 설정을 유연하게 구성할 수 있습니다. 다음은 몇 가지 예시입니다.

### 1. 사용자 정의 VLLM 엔드포인트를 사용하여 BBH(Big Bench Hard) 평가

이 예시는 특정 IP 주소의 VLLM 인스턴스를 통해 제공되는 Qwen3-8B 모델을 사용하여 `bbh` 작업을 실행하는 방법을 보여줍니다.

```bash
python -m eval.eval \
  --model curator \
  --tasks bbh \
  --model_args "base_url=http://10.7.60.170/vllm/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,tokenized_requests=False,max_requests_per_minute=15,max_tokens_per_minute=5000" \
  --num_fewshot 3 \
  --batch_size 1 \
  --limit 10 \
  --apply_chat_template False \
  --max_tokens 1000 \
  --output_path logs/qwen3-8b_bbh_full_results_test.json
```

이 설정의 주요 `--model_args`는 다음과 같습니다.
*   `base_url`: VLLM(또는 기타 호환 API) 서버의 엔드포인트입니다.
*   `model`: API 서버에서 인식하는 모델 식별자입니다.
*   `tokenizer`: 토크나이저의 Hugging Face 경로 또는 이름입니다.
*   `tokenizer_backend`: 토큰화에 `huggingface`를 사용하도록 지정합니다.
*   `tokenized_requests=False`: 모델에 대한 요청이 Curator에 의해 미리 토큰화되어서는 안 됨을 나타냅니다.
*   `max_requests_per_minute`, `max_tokens_per_minute`: 속도 제한 매개변수입니다.

### 2. 유사한 사용자 정의 VLLM 엔드포인트를 사용하여 AIME24 평가

이 예시는 `AIME24` 작업을 실행하는 방법을 보여줍니다.

```bash
python -m eval.eval \
  --model curator \
  --tasks AIME24 \
  --model_args "base_url=http://10.7.60.170/vllm/v1,model=qwen3-8b,tokenizer=Qwen/Qwen3-8B,tokenizer_backend=huggingface,max_requests_per_minute=15,max_tokens_per_minute=5000" \
  --batch_size 1 \
  --limit 1 \
  --apply_chat_template False \
  --max_tokens 1000 \
  --output_path logs/qwen3-8b_AIME24_full_results_test.json
```
`base_url`, `model`, `tokenizer` 등의 유사한 사용법을 확인하세요. `limit` 매개변수는 빠른 테스트 실행을 위해 1로 설정되었습니다. 모델 엔드포인트가 자체 템플릿을 처리하거나 원시 프롬프트를 보내는 경우 `apply_chat_template False`가 종종 중요합니다. 

## 주요 변경 사항 및 새로운 기능

Evalchemy는 지속적으로 개선되고 있으며, 최근 업데이트에는 `curator` 사용 편의성을 높이고 평가 파이프라인을 강화하는 여러 변경 사항이 포함되었습니다.

### 1. VLLM 호환성 및 유연성 향상 (`curator_lm.py`)

`curator` 모델은 이제 로컬 또는 원격 VLLM 서버와 더욱 원활하게 연동됩니다.

- **자동 `base_url` 처리**: `base_url` 인자에 VLLM 엔드포인트 주소를 제공할 때, 주소가 `/v1/completions` 또는 `/completions`로 끝나더라도 내부적으로 올바른 API 경로 (`/v1`)로 자동 변환됩니다.
- **API 키 관리**: VLLM 서버에 별도의 API 키가 필요하지 않은 경우, `api_key`를 제공하지 않아도 `curator`가 임시 키를 사용해 요청을 보냅니다.
- **기본 속도 제한**: VLLM 서버 사용 시, API 남용을 방지하기 위해 보수적인 기본 속도 제한(request/token per minute)이 적용됩니다. 물론 `--model_args`를 통해 직접 값을 지정하여 이를 재정의할 수 있습니다.
- **안정적인 요청 처리**: 다양한 생성 파라미터(예: `temperature`, `top_p`)를 가진 요청들을 그룹화하여 처리하는 기능이 개선되어, 복잡한 평가 시나리오에서도 안정적으로 작동합니다.

### 2. YAML 설정 파일을 통한 평가 간소화 (`eval.py`)

복잡한 커맨드라인 인자 대신 YAML 설정 파일을 사용하여 평가를 실행할 수 있는 `--config` 옵션이 추가되었습니다.

```bash
python -m eval.eval \
    --model hf \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.3" \
    --output_path logs \
    --config configs/light_gpt4omini0718.yaml
```

**예시 YAML 설정 파일 (`configs/light_gpt4omini0718.yaml`):**
```yaml
annotator_model: gpt-4o-mini-2024-07-18
max_tokens: 4096
tasks:
  - task_name: alpaca_eval
    batch_size: 16
  - task_name: MTBench
    batch_size: 16
  - task_name: WildBench
    batch_size: 16
```
이 파일을 사용하면 `tasks`, `batch_size`, `annotator_model`, `max_tokens` 인자를 커맨드라인에서 일일이 지정할 필요가 없습니다.

### 3. 체계적인 평가 관리 및 추적 (`eval_tracker.py`, `eval.py`)

고급 사용자를 위해 평가 결과를 체계적으로 관리하는 기능이 강화되었습니다.

- **데이터베이스 로깅**: `--use_database` 플래그를 활성화하면 평가 결과를 PostgreSQL 데이터베이스에 저장하여 실험을 추적하고 관리할 수 있습니다. (데이터베이스 설정은 `database/` 디렉토리 참조)
- **중복 평가 방지**: 데이터베이스 사용 시, `--model_id`를 지정하면 특정 모델에 대해 이미 완료된 평가는 건너뛰어 시간과 리소스를 절약할 수 있습니다.
- **모델 정보 조회**: `--model_id`를 통해 데이터베이스에 저장된 모델의 정보(예: 가중치 경로)를 가져와 평가에 사용할 수 있습니다.

이러한 변경 사항들은 `curator`를 포함한 Evalchemy의 전반적인 사용성을 개선하고, 특히 반복적인 대규모 평가를 수행하는 사용자에게 더 큰 편의성과 효율성을 제공합니다. 