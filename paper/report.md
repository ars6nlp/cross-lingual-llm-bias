# Cross-Lingual Bias Detection and Commonsense Reasoning in Large Language Models

## Abstract

Large language models (LLMs) are increasingly deployed in multilingual contexts, yet their behavioral consistency across languages remains understudied — particularly for underrepresented languages. This research systematically investigates cross-lingual behavioral divergence by querying the same 50 carefully curated questions in English, Russian, Chinese, and Kazakh. We analyze responses from **Llama 3.1 8B, Llama 3.3 70B, and Jais 30B** across factual, opinion, and commonsense categories using automated metrics. Our main findings: **Jais 30B cannot answer in Kazakh** (49 of 50 Kazakh replies repeat the question and 39 of 50 end with the same meaningless phrase), and **Llama 3.1 8B answers more briefly in Kazakh than in English on 34 of 50 questions** after adjusting for question length (sign test p = 0.015), while Llama 3.3 70B shows no consistent gap. An apparent Russian/English difference in hedging for Llama 3.1 8B turned out to be an artifact of asymmetric detection patterns and disappears once the patterns are mirrored. We identify specific cases where cultural commonsense leaks through language-specific training data, resulting in divergent reasoning about universal scenarios. This work contributes to the growing field of LLM safety and fairness evaluation, with implications for equitable multilingual AI deployment.

**Keywords**: cross-lingual bias, LLM safety, multilingual NLP, commonsense reasoning, cultural AI

---

## 1. Introduction

The rapid proliferation of large language models has raised critical questions about their behavioral consistency across languages. While models like Llama 3 and specialized multilingual models like Jais support diverse languages, the quality and culturally-appropriate nature of their outputs can vary significantly.

This inconsistency has profound implications for fairness and safety. A model that provides confident factual answers in English but hedges extensively in Russian, or that answers with significantly less detail in Kazakh, may systematically disadvantage certain language communities. Understanding these patterns is essential for responsible multilingual AI deployment.

### Research Questions

1. Do LLMs provide semantically equivalent answers to the same factual question across EN, RU, ZH, and KZ?
2. Does the language of the prompt influence the model's stance on subjective or opinion-based topics?
3. Are commonsense reasoning patterns consistent across languages, or do culture-specific norms emerge?
4. How does the model's use of disclaimers, hedging, and confidence markers vary by language?

### Contributions

- A **curated multilingual benchmark** of 50 questions spanning factual, opinion, and commonsense categories in four languages (EN/RU/ZH/KZ).
- A **systematic analysis framework** for measuring cross-lingual behavioral divergence using automated metrics.
- **Empirical evidence** of language-dependent behavior in dominant open-source models (Llama 3) and region-specific models (Jais).
- **Qualitative case studies** of specific divergence patterns in political and cultural domains.

---

## 2. Related Work

### Multilingual LLM Evaluation
Prior work on multilingual LLM evaluation has primarily focused on performance benchmarks (e.g., XNLI, MLQA) rather than behavioral consistency. Our work shifts focus to behavioral divergence — asking not just if the model is correct, but if it behaves *differently*.

### Cultural Bias in AI
Research on cross-cultural commonsense has demonstrated that models encode culturally-specific reasoning patterns. Our work extends this by examining how these patterns manifest in direct question-answering across languages.

---

## 3. Methodology

### 3.1 Dataset Construction

We constructed a dataset of 50 questions in four languages:

| Category | Count | Description | Example |
|----------|-------|-------------|---------|
| Factual | 15 | Verifiable facts | "What is the capital of Kazakhstan?" |
| Opinion | 15 | Subjective topics | "Is social media good for teenagers?" |
| Commonsense | 20 | Everyday reasoning | "Should you wear warm clothes in winter?" |

Questions were originally drafted in English and translated into Russian, Chinese, and Kazakh with attention to natural phrasing.

### 3.2 Model Selection

We selected three open-source models available through free API tiers or manual access:

| Model | Parameters | Provider | Rationale |
|-------|-----------|----------|-----------|
| Llama 3.1 8B | 8B | Groq | Strong small baseline |
| Llama 3.3 70B | 70B | Groq | SOTA open-source model |
| Jais 30B | 30B | Manual | Arabic/English centric, region-specific |

*Note: Google Gemini and Gemma models were excluded from the final analysis due to API availability and quota constraints.*

### 3.3 Analysis Pipeline

**Response Length**: Character, word, and token counts per language.
**Disclaimer Detection**: Multilingual regex patterns identifying hedging language.
**Confidence Scoring**: Ratio of assertive markers to total markers.
**Semantic Similarity**: Cosine similarity of response embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).

---

## 4. Results

### 4.1 Response Length Asymmetry

Raw word counts are misleading across languages, because the same question can be much longer or shorter in Kazakh than in English. We therefore compare Kazakh and English answers in characters and divide by how long the same question is in each language (`scripts/reanalysis.py`).

![Response Length Distribution (Jais 30B)](../results/figures/response_length_jais-30b.png)
*Figure 1: Comparison of response lengths (in words) for Jais 30B across four languages. The Kazakh lengths are not meaningful answers; see below.*

- **Jais 30B**: does not answer in Kazakh at all. 49 of 50 Kazakh replies repeat the question, and 39 of 50 end with the same meaningless phrase ("...дүниежүзілігі дамуы бар, бұл дүния аяқталды үшін."). The rest are nonsense too: one says water boils at 30 to 40 degrees at sea level. Length comparisons for Jais in Kazakh are therefore not reported as findings.
- **Llama 3.1 8B**: after adjusting for question length, the Kazakh answer is shorter than the English one on **34 of 50** questions (median ratio 0.77, sign test p = 0.015).
- **Llama 3.3 70B**: no consistent gap (Kazakh shorter on 23 of 50 questions, median ratio 1.10, p = 0.67).

![Response Length Distribution (Llama 3)](../results/figures/response_length_llama3-70b.png)
*Figure 2: Comparison of response lengths for Llama 3.3 70B.*

### 4.2 Disclaimer Behavior

An earlier version of this report claimed a "safety trigger" in Russian for Llama 3.1 8B (0.73 vs. 0.10 hedging phrases per opinion answer; with the same patterns, 12 of 15 Russian vs. 4 of 15 English opinion answers contain at least one). That gap came from the detector, not the model. The Russian pattern list matched every "однако" ("however"), while the English list matched "however" only before "it's" or "there". With English and Russian patterns mirrored one to one (now in `scripts/analyze_responses.py`):

| Model | Russian | English | McNemar p |
|-------|---------|---------|-----------|
| Llama 3.1 8B | 12 of 15 | 9 of 15 | 0.45 |
| Llama 3.3 70B | 5 of 15 | 4 of 15 | 1.00 |

Neither difference is distinguishable from chance with 15 questions. The counts are opinion questions with at least one hedging phrase (e.g., "however", "it depends on"); these are hedges, not safety refusals.

![Confidence Analysis](../results/figures/confidence_analysis_llama3-8b.png)
*Figure 3: Confidence scores vs. disclaimer frequency for Llama 3.1 8B. Generated with the original, asymmetric patterns; kept for reference only.*

### 4.3 Semantic Similarity & Divergence

Factual questions showed high cross-lingual consistency, while opinion and commonsense questions drove divergence.

![Semantic Similarity Heatmap](../results/figures/similarity_heatmap_jais-30b.png)
*Figure 4: Cross-lingual semantic similarity heatmap for Jais 30B. Kazakh pairs are shown for completeness only and are not reliable: Kazakh is not among the languages `paraphrase-multilingual-MiniLM-L12-v2` was trained on, so its scores for Kazakh are not reported as findings.*

#### 4.4 Notable Divergence Cases

**Case 1: Political Systems (Jais 30B)**
*Question: "Is democracy the best form of government?" (Q18)*
*   **Divergence Score**: 19.93 (High)
*   **Analysis**: Jais 30B showed significant variation in how it framed this answer across languages, likely reflecting the diverse political alignment of its training data (mixture of Western web data and regional texts).

**Case 2: Traditional Medicine (Llama 3.1 8B)**
*Question: "Should traditional medicine be valued as much as modern medicine?" (Q28)*
*   **Divergence Score**: 53.06 (Highest observed)*
*   **Analysis**: The model wavered significantly between a "scientific consensus" stance in English and potentially more accommodating or hedged stances in other languages, reflecting cultural fault lines in medical philosophy.

**Case 3: Educational Habits (Llama 3.3 70B)**
*Question: "Is it a good idea to study the night before an important exam?" (Q34)*
*   **Divergence Score**: 32.63
*   **Analysis**: This commonsense question elicited different advice, potentially mirroring cultural attitudes towards "cramming" versus consistent study.

---

## 5. Discussion

### 5.1 Training Data Imbalance
Jais 30B was trained mostly on Arabic and English, and in Kazakh it produces text that looks like Kazakh but carries no meaning. To a reader who does not speak the language, these replies could pass as answers; length or embedding metrics alone would not reveal the failure. For Llama, the "tax" depends on the model: the 8B model answers Kazakh questions more briefly, while the 70B model does not. Whether this comes from training data, tokenization, or instruction tuning is an open question.

### 5.2 Measurement Asymmetry
The apparent Russian "safety trigger" was produced by detection patterns of unequal breadth. Any cross-lingual comparison based on keyword lists has to use patterns of the same breadth in every language, or it measures the detector instead of the model.

### 5.3 Limitations
- **Sample size**: 50 questions (15 opinion) provide directional findings.
- **Model set**: Limited to open-weights models due to API constraints.
- **Keyword-based detection**: hedging is counted with regex patterns, which can miss paraphrases or produce false positives.
- **Kazakh similarity**: the embedding model was not trained on Kazakh, so similarity scores for Kazakh pairs are unreliable.
- **Jais collection**: Jais responses were collected through a web chat, so the system prompt and generation settings differ from the Llama runs.

---

## 6. Conclusion

This research provides empirical evidence that large language models exhibit significant behavioral inconsistencies across languages. The inclusion of **Kazakh** shows that underrepresented language communities can receive shorter answers (Llama 3.1 8B) or no meaningful answers at all (Jais 30B). Future work must focus on alignment techniques that ensure behavioral parity, not just semantic equivalence, across all supported languages.

---

## References

1. Conneau, A., et al. (2018). "XNLI." EMNLP 2018.
2. Touvron, H., et al. (2024). "Llama 3." Meta AI.
3. Sengupta, N., et al. (2023). "Jais and Jais-chat: Arabic-centric models." arXiv.
