# Programmatic SEO Engine with Automated Quality Gates

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Gumroad: Free Download](https://img.shields.io/badge/Gumroad-Free%20Download-FF385C)](https://mannyverse767.gumroad.com/l/programmatic-seo-markdown-generator-python-20260929)
[![Whop: Free](https://img.shields.io/badge/Whop-Free%20Access-orange)](https://whop.com/mr-studio-7faa/programmatic-seo-engine-with-automated-quality-gates-python-markdown)

> 🎁 **Free Open-Source Community Edition** by [MR Studio](https://manny-ruesch.github.io/tools.html)  
> ⭐ **Find this valuable?** Please take 10 seconds to star this repository and leave a **5-star review on [Gumroad](https://mannyverse767.gumroad.com/l/programmatic-seo-markdown-generator-python-20260929)** — it helps keep this engine open and updated!  
> 🚀 **Explore Full Automation Suite:** [Browse Production Engines (RAG Sync, B2B Lead Scoring, Scrapers) →](https://manny-ruesch.github.io/tools.html)

---

## 1. Overview & Architecture

Scaling programmatic SEO using standard LLM pipelines frequently introduces algorithmic search penalties due to:
* **N-gram Repetition & Syntactic Drift:** High token overlap across pages within the same programmatic cluster.
* **Information Density Deficits:** Low entity diversity and degraded Flesch-Kincaid readability metrics.
* **Structural Inconsistencies:** Malformed frontmatter that breaks headless SSG content collections.

This engine mitigates these risks by executing deterministic heuristic validation checks between the generation and disk-write steps. Content that violates safety, lexical, or structural constraints is automatically diverted to an isolated quarantine queue alongside structured debugging logs.

### Pipeline Architecture

```
[ Data Source: CSV / JSON / DB ]
               │
               ▼
┌───────────────────────────────────────────────┐
│ Async Batch Engine (main.py)                  │
│ Rate Limiter (Token Bucket / Concurrency Pool) │
└──────────────────────┬────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
 [ OpenAI API ]  [ Anthropic API ] [ Ollama Node ]
       │               │               │
       └───────────────┼───────────────┘
                       │ Markdown + YAML Frontmatter
                       ▼
┌───────────────────────────────────────────────┐
│ Quality Gate Subsystem                        │
│ 1. Schema Validation (Strict Pydantic YAML)    │
│ 2. Readability (Flesch-Kincaid Grade Level)   │
│ 3. Lexical Redundancy (N-gram Jaccard Index)  │
│ 4. Entity Verification (Spacy NER Check)      │
└──────────────────────┬────────────────────────┘
                       │
             Pass ─────┴───── Fail
              │                │
              ▼                ▼
     ┌────────────────┐ ┌────────────────┐
     │ Target SSG Out │ │ Quarantine Out │
     │  /dist/{ssg}   │ │  /quarantine   │
     │                │ │  + error.json  │
     └────────────────┘ └────────────────┘
```

---

## 2. Prerequisites & Environment Setup

* **Python:** Version `>= 3.10`
* **Node.js (Optional):** Required only if validating directly against local SSG dev servers (`astro`, `next`, or `hugo`).
* **Hardware/Compute:** 
  * Cloud APIs (OpenAI/Anthropic): Any modern multi-core system.
  * Local LLM (Ollama): Minimum 16GB VRAM (e.g., RTX 3090/4090 or Apple Silicon M-Series) for local inference at scale.

### System Dependencies (Linux/Debian)

```bash
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
```

---

## 3. Installation & Quickstart

### Step 1: Clone & Create Virtual Environment

```bash
git clone https://github.com/your-org/programmatic-seo-engine.git
cd programmatic-seo-engine
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 3: Populate Input Dataset

Create an input file at `data/inputs.json` mapping your programmatic variables:

```json
[
  {
    "slug": "postgresql-performance-tuning-aws-aurora",
    "primary_keyword": "PostgreSQL performance tuning Aurora",
    "target_entities": ["PostgreSQL", "Amazon Aurora", "IOPS", "WAL", "Buffer Cache"],
    "audience": "Principal Database Engineers"
  },
  {
    "slug": "redis-cluster-failover-strategies",
    "primary_keyword": "Redis cluster failover automation",
    "target_entities": ["Redis Sentinel", "Quorum", "Replication", "Split-Brain"],
    "audience": "Site Reliability Engineers"
  }
]
```

### Step 4: Run the Engine

Execute the engine with your target Static Site Generator output format (e.g., `astro`, `nextjs`, or `hugo`):

```bash
python main.py \
  --input data/inputs.json \
  --ssg astro \
  --provider anthropic \
  --concurrency 10
```

### Step 5: Verify Build Artifacts

Post-execution, generated files are sorted into output directories:

```
├── dist/
│   └── astro/
│       └── postgresql-performance-tuning-aws-aurora.md
└── quarantine/
    └── redis-cluster-failover-strategies/
        ├── draft.md
        └── heuristic_failures.json
```

---

## 4. Configuration Options

Runtime behavior is defined via a `.env` file and command-line flags. Copy the example configuration to begin:

```bash
cp .env.example .env
```

### Environment Variables (`.env`)

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `LLM_PROVIDER` | `string` | `openai` | Inference backend: `openai`, `anthropic`, or `ollama`. |
| `OPENAI_API_KEY` | `string` | `None` | Authentication key for OpenAI APIs. |
| `ANTHROPIC_API_KEY` | `string` | `None` | Authentication key for Anthropic APIs. |
| `OLLAMA_BASE_URL` | `string` | `http://localhost:11434` | Base HTTP endpoint for local Ollama instances. |
| `MODEL_NAME` | `string` | `claude-3-5-sonnet-20241022`| Base model identifier passed to the inference driver. |
| `MAX_CONCURRENT_REQUESTS`| `int` | `5` | Maximum simultaneous open network connections/streams. |
| `REQUESTS_PER_MINUTE` | `int` | `60` | Token-bucket rate limiter constraint. |
| `TARGET_SSG` | `string` | `astro` | Target schema syntax: `astro`, `nextjs`, or `hugo`. |
| `MIN_FLESCH_READING_EASE`| `float` | `45.0` | Lower bound score for Flesch-Kincaid Reading Ease. |
| `MAX_FLESCH_GRADE_LEVEL` | `float` | `12.0` | Upper bound for Flesch-Kincaid Grade Level index. |
| `MAX_NGRAM_REPETITION` | `float` | `0.18` | Max allowable Jaccard redundancy for tri-grams & 4-grams. |
| `MIN_ENTITY_COVERAGE` | `float` | `0.80` | Required percentage match of target entities present. |
| `OUTPUT_DIR` | `string` | `./dist` | Destination directory for validated, production-ready markdown. |
| `QUARANTINE_DIR` | `string` | `./quarantine` | Destination directory for drafts failing quality gates. |

---

## 5. Production Tips & Maintenance

### Static Site Generator Frontmatter Compliance
Each SSG handles content collection schemas differently. The output engine strictly structures frontmatter to prevent ingestion failures:

* **Astro:** Serializes clean, ISO-8601 datetimes and scalar types natively parsed by `astro:content` and Zod schemas.
* **Next.js:** Complies with Contentlayer or Content Collections conventions, enforcing raw serializable strings for all custom object payloads.
* **Hugo:** Emits YAML strictly delimited by `---` blocks with support for Hugo taxonomy mappings.

### Mitigating Algorithmic Penalties
* **Set Conservative N-gram Thresholds:** Programmatic engines often reuse fixed transition phrases across similar titles. Keep `MAX_NGRAM_REPETITION` between `0.12` and `0.20`. Any draft showing elevated lexical overlap with previously generated pages in the same run is flagged and quarantined.
* **Entity Verification Tuning:** The quality gate uses spaCy's `en_core_web_sm` model to extract Named Entities (`ORG`, `PRODUCT`, `GPE`, `PERSON`, etc.). If your targeted niche uses obscure terminology, supplement the pipeline with a custom spaCy EntityRuler or train a domain-specific model to avoid false-positive quarantine drops.

### Triage & Maintenance of the Quarantine Queue
Drafts in the quarantine directory include a `heuristic_failures.json` file. A failure log entry appears as follows:

```json
{
  "timestamp": "2024-10-27T10:45:00Z",
  "slug": "redis-cluster-failover-strategies",
  "failures": [
    {
      "gate": "FleschKincaidValidator",
      "threshold": "<= 12.0",
      "actual": 14.2,
      "message": "Content complexity exceeds target audience readability boundary."
    },
    {
      "gate": "EntityCoverageValidator",
      "threshold": ">= 0.80",
      "actual": 0.60,
      "missing_entities": ["Split-Brain"]
    }
  ]
}
```

* **Automated Feedback Loop:** Write an automated cleanup job that reads `heuristic_failures.json`, injects the explicit failures into an LLM error-correction prompt, and executes a second-pass generation.
* **Pruning:** Routinely archive or purge quarantined records to preserve build performance.

---

## 6. License & Support Info

### License
Distributed under the **Apache 2.0 License**. See `LICENSE` for structural terms, liability exclusions, and intellectual property conditions.

### Contributing & Issue Tracking
* **Bugs & Regressions:** File an issue specifying your Python environment, target SSG, and an isolated payload to reproduce the error.
* **Submitting Gates:** PRs introducing custom deterministic validators (e.g., markdown link checks, automated core-web-vitals image linting) must include corresponding unit tests inside `/tests`.
* **CI Validation:** Ensure all codebase modifications pass standard linting before submitting a PR:
  ```bash
  flake8 main.py
  black --check main.py
  pytest tests/