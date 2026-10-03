아래처럼 정리하면 돼. HaluMem 정의/구성은 공식 GitHub와 Hugging Face dataset card 기준으로 확인했어. HaluMem은 2026년 EMNLP Main에 채택된 **agent memory system의 hallucination을 operation-level로 평가하는 benchmark**다. [GitHub](https://github.com/MemTensor/HaluMem/blob/main/README.md?utm_source=chatgpt.com)

## HaluMem Fast Binary Judge 결과

| Metric | Result |
|---|---:|
| N | 100 |
| TP | 47 |
| TN | 32 |
| FP | 18 |
| FN | 3 |
| **Accuracy** | **0.7900** |
| Precision | 0.7231 |
| **Recall** | **0.9400** |
| **F1** | **0.8174** |
| Negative Rejection Rate | 0.6400 |
| False Accept Rate | 0.3600 |
| **False Reject Rate** | **0.0600** |
| Reasoning Tokens | 0 |
| Mean Latency | **0.843 s** |

### Confusion Matrix

|  | Judge: Positive | Judge: Negative |
|---|---:|---:|
| **Actual Positive** | TP = **47** | FN = **3** |
| **Actual Negative** | FP = **18** | TN = **32** |

즉 positive 50개 중 **47개를 잡아서 Recall 94%**로 상당히 높다. 반면 negative 50개 중 18개를 positive로 잘못 받아들여 **False Accept Rate가 36%**다.

결과를 한 문장으로 요약하면:

> **Positive memory는 거의 놓치지 않지만, interference 같은 부적절한 memory도 valid memory로 받아들이는 경향이 강한 judge다.**

---

## 오분류 분석

총 오분류는 **21 / 100 = 21%**다.

### False Negative — 실제 Positive인데 Reject

3건뿐이다.

| ID | Source | Expected | Judge | 내용 요약 |
|---:|---|---:|---:|---|
| 14 | secondary | 1 | 0 | Martin의 emotional support/companionship 필요 |
| 17 | secondary | 1 | 0 | Karen의 chronic disease 상태가 Managed로 변경 |
| 46 | secondary | 1 | 0 | Sarah의 음악 취향과 성격 관련 memory |

따라서 positive memory rejection은 낮다.

**False Reject Rate = 6%**

---

### False Positive — 실제 Negative인데 Accept

18건으로 오분류 대부분을 차지한다.

특히 제공한 결과에서 FP가 전부 `memory_source=interference`다.

예를 들면:

- Christopher의 travel이 creative motivation과 연결된다는 내용
- Donna의 strategy game 선호가 adaptability를 반영한다는 내용
- Taylor의 health status가 Declining → Stable로 바뀌었다는 내용
- Barbara가 Chronic Fatigue Syndrome 진단을 받았다는 내용
- Joseph Garcia의 career industry가 변경됐다는 내용

이런 문장들을 judge가 **실제 저장할 만한 memory라고 판단해 버렸다.**

따라서 현재 병목은 명확하다.

> **FN보다 FP 문제가 훨씬 크다.**

---

# HaluMem이란?

**HaluMem (Hallucination in Memory)**은 LLM/agent의 **long-term memory system에서 발생하는 hallucination을 평가하기 위한 benchmark**다.

기존 memory benchmark가 보통 최종 QA accuracy만 보는 것과 달리 HaluMem은 memory pipeline을 세 단계로 나눠서 본다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem?utm_source=chatgpt.com)

### 1. Memory Extraction

대화에서 실제로 저장해야 할 memory를 정확하게 추출하는지 평가한다.

예를 들어:

```text
User:
I recently started learning tennis.

Valid memory:
The user recently started learning tennis.
```

반대로 assistant가 대화에 없는 내용을 만들어내거나 irrelevant한 정보를 memory로 저장하면 hallucination으로 볼 수 있다.

### 2. Memory Update

시간이 지나면서 user 정보가 변했을 때 기존 memory를 올바르게 update하는지를 평가한다.

예:

```text
Old memory:
Job = Software Engineer

New conversation:
I recently became an Engineering Manager.

Updated memory:
Job = Engineering Manager
```

HaluMem memory point에는 실제로 `is_update`와 `original_memories` 필드가 들어 있어 과거 memory와 새 memory 간 변경을 추적할 수 있다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/blob/main/README.md?code=true)

### 3. Memory Question Answering

저장된 memory를 retrieval하고 reasoning해서 최종 질문에 제대로 대답하는지를 평가한다.

즉,

```text
Extraction
   ↓
Update
   ↓
Retrieval
   ↓
Generation
```

전체 memory pipeline의 최종 품질도 측정한다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem?utm_source=chatgpt.com)

---

# HaluMem 데이터가 실제로 무엇인가?

단순한 `question → answer` dataset은 아니다.

기본 단위가 **가상의 한 user가 AI assistant와 오랫동안 나눈 multi-session dialogue history**다.

각 user 데이터는 크게 다음 형태다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/blob/main/README.md?code=true)

```text
User
 ├── persona_info
 │
 └── sessions
      ├── dialogue
      ├── memory_points
      └── questions
```

### `persona_info`

가상 사용자의 background, personality, goals, motivations 등을 담는다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/blob/main/README.md?code=true)

### `dialogue`

실제 user ↔ assistant multi-turn conversation이다.

### `memory_points`

대화에서 기억해야 하는 개별 fact/event다.

공식 dataset에는 대략 이런 구조가 있다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/blob/main/README.md?code=true)

```json
{
  "index": 1,
  "memory_content": "...",
  "memory_type": "Event Memory",
  "memory_source": "secondary",
  "is_update": true,
  "original_memories": [...],
  "timestamp": "...",
  "importance": 0.75
}
```

주요 필드는:

| Field | 의미 |
|---|---|
| `memory_content` | 기억해야 하는 사실/사건 |
| `memory_type` | Persona / Event / Relationship 등 |
| `memory_source` | primary / secondary / interference / system |
| `is_update` | 과거 memory의 수정인지 |
| `original_memories` | update 전 관련 memory |
| `importance` | memory 중요도 |
| `timestamp` | 생성/업데이트 시간 |

공식 데이터에서 `memory_source`는 실제로 **primary, secondary, interference, system** 네 종류로 제공된다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/blob/main/README.md?code=true)

---

# 특히 `interference`가 중요한 이유

HaluMem은 일부러 **잘못되거나 방해가 되는 내용(distractor/interference)**을 dialogue에 섞는다.

공식 설명에 따르면 multi-turn dialogue 생성 과정에서 **adversarial distractor memories**, 즉 AI가 미묘하게 잘못된 사실을 말하도록 넣어 hallucination 상황을 만든다. 또한 irrelevant QA도 삽입해서 long-context noise를 증가시킨다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/blob/main/README.md?code=true)

그래서 대략 이런 문제가 만들어진다.

```text
실제 user 정보
↓
User: I don't really play strategy games.

Assistant:
It sounds like your growing interest in strategy games
shows your adaptability.

↓
interference memory candidate

"Donna's gaming interests evolved to include strategy games..."
```

문장 자체만 보면 굉장히 자연스럽다.

하지만 **실제 user fact가 아니라 assistant가 만들어낸 정보**라면 memory에 저장하면 안 된다.

네 결과에서 FP가 전부 이런 `interference`인 이유도 바로 이 부분이다.

---

# HaluMem 데이터 규모

공식 데이터는 두 가지 버전이다. [GitHub](https://github.com/MemTensor/HaluMem?utm_source=chatgpt.com)

| | HaluMem-Medium | HaluMem-Long |
|---|---:|---:|
| Users | 20 | 20 |
| Dialogues | 30,073 | 53,516 |
| Avg. Sessions/User | 70 | 120 |
| Avg. Context | ~160K tokens | **~1M tokens** |
| Memory Points | 14,948 | 14,948 |
| QA Pairs | 3,467 | 3,467 |

**HaluMem-Long**에는 factual QA, math 문제 등의 distractor를 추가해서 약 **1M-token/user**까지 context를 늘린다. 긴 대화에서도 memory system이 중요한 user memory와 noise를 구별하는지를 보는 것이다. [Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/blob/main/README.md?code=true)

---

# 네 실험이 정확히 무엇을 평가하나

네 `HaluMem Fast Binary Judge`는 HaluMem 전체 benchmark를 그대로 실행한 것은 아니고, HaluMem의 memory point를 이용해 다음 binary classification을 하는 형태로 보여.

```text
memory candidate
       ↓
    LLM Judge
       ↓
  ┌───────────┐
  │ 1: Accept │ → 저장 가능한 valid memory
  │ 0: Reject │ → interference / 저장하면 안 됨
  └───────────┘
```

현재 결과를 가장 간단하게 정리하면:

```text
Positive detection : 매우 좋음
Recall             : 94%

Interference reject: 부족
Negative rejection : 64%

전체 성능
Accuracy           : 79%
F1                 : 81.7%
Latency            : 0.843 s
```

그리고 **오류 21건 중 18건(85.7%)이 interference를 valid memory로 받아들인 FP**다.

따라서 다음 개선 목표는 Recall을 더 올리는 것보다는 **interference rejection을 강화해서 FP 18개를 줄이는 것**이 맞아. 특히 `"X reflects personality"`, `"aligning with values"`, `"renewed commitment"`처럼 **대화의 명시적 사실보다 해석·추론이 섞인 memory를 reject하도록 judge prompt를 강화**하면 지금 결과에서 가장 큰 개선 여지가 있어 보여.
