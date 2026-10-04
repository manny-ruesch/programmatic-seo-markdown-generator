#!/usr/bin/env python3
"""
Programmatic SEO Engine with Automated Quality Gates
Generates headless SSG markdown pages with strict YAML frontmatter,
enforcing multi-stage heuristic quality audits and automated quarantine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SEOEngine")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class QualityThresholds:
    min_word_count: int = 600
    max_word_count: int = 3500
    min_fkgl: float = 6.0
    max_fkgl: float = 13.0
    min_trigram_diversity: float = 0.72
    min_fourgram_diversity: float = 0.80
    min_entity_coverage_ratio: float = 0.80
    banned_phrases: List[str] = field(
        default_factory=lambda: [
            "as an ai",
            "as a language model",
            "in conclusion",
            "delve into",
            "testament to",
            "tapestry of",
            "bustling hub",
            "beacon of",
            "[insert",
            "todo",
        ]
    )


@dataclass
class SeedRecord:
    slug: str
    primary_keyword: str
    secondary_keywords: List[str]
    intent: str
    target_audience: str
    entities: List[str]
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetricBreakdown:
    word_count: int
    sentence_count: int
    syllable_count: int
    fkgl_score: float
    trigram_diversity: float
    fourgram_diversity: float
    entity_coverage_ratio: float
    missing_entities: List[str]
    banned_phrases_found: List[str]
    is_valid: bool
    failure_reasons: List[str]


@dataclass
class GenerationResult:
    seed: SeedRecord
    raw_response: str
    parsed_frontmatter: Dict[str, Any]
    markdown_body: str
    quality_audit: QualityMetricBreakdown
    status: str  # 'PUBLISHED' or 'QUARANTINED'


# ---------------------------------------------------------------------------
# Linguistic & Quality Validation Engine
# ---------------------------------------------------------------------------
class TextStatistics:
    """Computes phonetic and structural metrics without external C-dependencies."""

    _VOWEL_RUNS = re.compile(r"[aeiouy]+", re.IGNORECASE)
    _SENTENCE_END = re.compile(r"[.!?]+(\s+|$)")
    _WORD_RE = re.compile(r"\b[a-zA-Z0-9\'-]+\b")

    @classmethod
    def count_syllables_word(cls, word: str) -> int:
        word = word.lower().strip()
        if not word:
            return 0
        if len(word) <= 3:
            return 1

        if word.endswith("es") or word.endswith("ed"):
            if not word.endswith("ted") and not word.endswith("ded"):
                word = word[:-2]
        elif word.endswith("e") and not word.endswith("le"):
            word = word[:-1]

        matches = cls._VOWEL_RUNS.findall(word)
        count = len(matches)
        return max(1, count)

    @classmethod
    def extract_words(cls, text: str) -> List[str]:
        return cls._WORD_RE.findall(text)

    @classmethod
    def extract_sentences(cls, text: str) -> List[str]:
        sentences = [s.strip() for s in cls._SENTENCE_END.split(text) if s and len(s.strip()) > 1]
        return sentences if sentences else [text]

    @classmethod
    def calculate_fkgl(cls, words: List[str], sentences: List[str]) -> Tuple[float, int, int]:
        total_words = len(words)
        total_sentences = max(1, len(sentences))
        if total_words == 0:
            return 0.0, 0, total_sentences

        total_syllables = sum(cls.count_syllables_word(w) for w in words)
        words_per_sentence = total_words / total_sentences
        syllables_per_word = total_syllables / total_words

        fkgl = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
        return round(fkgl, 2), total_syllables, total_sentences

    @classmethod
    def calculate_ngram_diversity(cls, words: List[str], n: int) -> float:
        if len(words) < n:
            return 1.0
        lowered = [w.lower() for w in words]
        ngrams = [tuple(lowered[i : i + n]) for i in range(len(lowered) - n + 1)]
        if not ngrams:
            return 1.0
        unique_ngrams = set(ngrams)
        return round(len(unique_ngrams) / len(ngrams), 4)


class QualityGateValidator:
    """Enforces multi-stage programmatic quality validation."""

    def __init__(self, thresholds: QualityThresholds):
        self.thresholds = thresholds

    def audit(self, text: str, seed: SeedRecord) -> QualityMetricBreakdown:
        # Strip frontmatter if present
        clean_text = re.sub(r"^---[\s\S]+?---\s*", "", text).strip()
        words = TextStatistics.extract_words(clean_text)
        sentences = TextStatistics.extract_sentences(clean_text)

        word_count = len(words)
        sentence_count = len(sentences)
        fkgl, syllable_count, _ = TextStatistics.calculate_fkgl(words, sentences)
        trigram_div = TextStatistics.calculate_ngram_diversity(words, 3)
        fourgram_div = TextStatistics.calculate_ngram_diversity(words, 4)

        # Entity verification
        clean_text_lower = clean_text.lower()
        found_entities = [e for e in seed.entities if e.lower() in clean_text_lower]
        missing_entities = [e for e in seed.entities if e.lower() not in clean_text_lower]
        coverage_ratio = len(found_entities) / max(1, len(seed.entities))

        # Check banned/hallucinated phrases
        banned_found = [
            phrase for phrase in self.thresholds.banned_phrases if phrase in clean_text_lower
        ]

        # Failures aggregation
        failures = []
        if word_count < self.thresholds.min_word_count:
            failures.append(f"Word count {word_count} below threshold {self.thresholds.min_word_count}")
        elif word_count > self.thresholds.max_word_count:
            failures.append(f"Word count {word_count} exceeds threshold {self.thresholds.max_word_count}")

        if fkgl < self.thresholds.min_fkgl:
            failures.append(f"FKGL score {fkgl} too simple (min: {self.thresholds.min_fkgl})")
        elif fkgl > self.thresholds.max_fkgl:
            failures.append(f"FKGL score {fkgl} too dense/complex (max: {self.thresholds.max_fkgl})")

        if trigram_div < self.thresholds.min_trigram_diversity:
            failures.append(f"Trigram diversity {trigram_div} below {self.thresholds.min_trigram_diversity} (looping repetition)")

        if fourgram_div < self.thresholds.min_fourgram_diversity:
            failures.append(f"4-gram diversity {fourgram_div} below {self.thresholds.min_fourgram_diversity} (structural repetition)")

        if coverage_ratio < self.thresholds.min_entity_coverage_ratio:
            failures.append(
                f"Entity coverage ratio {coverage_ratio:.2f} below {self.thresholds.min_entity_coverage_ratio:.2f}. "
                f"Missing: {missing_entities}"
            )

        if banned_found:
            failures.append(f"Contains AI boilerplate or placeholder phrases: {banned_found}")

        # Primary Keyword Check
        if seed.primary_keyword.lower() not in clean_text_lower:
            failures.append(f"Primary keyword '{seed.primary_keyword}' absent from body")

        is_valid = len(failures) == 0

        return QualityMetricBreakdown(
            word_count=word_count,
            sentence_count=sentence_count,
            syllable_count=syllable_count,
            fkgl_score=fkgl,
            trigram_diversity=trigram_div,
            fourgram_diversity=fourgram_div,
            entity_coverage_ratio=round(coverage_ratio, 2),
            missing_entities=missing_entities,
            banned_phrases_found=banned_found,
            is_valid=is_valid,
            failure_reasons=failures,
        )


# ---------------------------------------------------------------------------
# SSG Serialization Engine
# ---------------------------------------------------------------------------
class SSGSerializer:
    """Handles YAML frontmatter serialization tailored to Astro, Next.js, and Hugo."""

    @staticmethod
    def _escape_yaml_str(value: str) -> str:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'

    @classmethod
    def generate_frontmatter(
        cls,
        flavor: str,
        seed: SeedRecord,
        title: str,
        meta_description: str,
    ) -> Dict[str, Any]:
        iso_now = datetime.now(timezone.utc).isoformat()
        base_meta = {
            "title": title,
            "description": meta_description,
            "slug": seed.slug,
            "primaryKeyword": seed.primary_keyword,
            "canonical": f"https://example.com/insights/{seed.slug}",
            "tags": [seed.primary_keyword] + seed.secondary_keywords[:3],
        }

        if flavor == "astro":
            base_meta.update({
                "pubDate": iso_now,
                "isDraft": False,
                "author": "Editorial Team",
                **seed.custom_metadata,
            })
        elif flavor == "nextjs":
            base_meta.update({
                "publishedAt": iso_now,
                "status": "published",
                "seo": {
                    "openGraph": {"type": "article"},
                    "keywords": [seed.primary_keyword] + seed.secondary_keywords,
                },
                **seed.custom_metadata,
            })
        elif flavor == "hugo":
            base_meta.update({
                "date": iso_now,
                "draft": False,
                "keywords": [seed.primary_keyword] + seed.secondary_keywords,
                **seed.custom_metadata,
            })
        else:
            base_meta["date"] = iso_now

        return base_meta

    @classmethod
    def serialize_yaml(cls, data: Dict[str, Any], indent_level: int = 0) -> str:
        lines = []
        indent = "  " * indent_level
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent}{key}:")
                lines.append(cls.serialize_yaml(value, indent_level + 1))
            elif isinstance(value, list):
                lines.append(f"{indent}{key}:")
                for item in value:
                    if isinstance(item, (dict, list)):
                        raise ValueError("Deeply nested collections not supported in simple serializer.")
                    lines.append(f"{indent}  - {cls._escape_yaml_str(str(item))}")
            elif isinstance(value, bool):
                lines.append(f"{indent}{key}: {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                lines.append(f"{indent}{key}: {value}")
            else:
                lines.append(f"{indent}{key}: {cls._escape_yaml_str(str(value))}")
        return "\n".join(lines)

    @classmethod
    def assemble_markdown(cls, frontmatter_data: Dict[str, Any], body: str) -> str:
        yaml_payload = cls.serialize_yaml(frontmatter_data)
        return f"---\n{yaml_payload}\n---\n\n{body.strip()}\n"

    @classmethod
    def extract_sections(cls, raw_content: str) -> Tuple[str, str, str]:
        """Extracts Title, Meta Description, and Markdown Body from LLM response."""
        title = "Article"
        description = "Article meta description"
        body = raw_content

        title_match = re.search(r"^#\s+(.+)$", raw_content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        desc_match = re.search(
            r"(?:META_DESCRIPTION|Description):\s*([^\n\r]+)", raw_content, re.IGNORECASE
        )
        if desc_match:
            description = desc_match.group(1).strip()
            # Strip explicit meta description tag from body
            body = re.sub(
                r"(?:META_DESCRIPTION|Description):\s*[^\n\r]+\n*", "", body, flags=re.IGNORECASE
            )

        return title, description, body.strip()


# ---------------------------------------------------------------------------
# Asynchronous Multi-Provider LLM Client
# ---------------------------------------------------------------------------
class AsyncLLMClient:
    """Asynchronous client with retry, backoff, and concurrency rate limiting."""

    def __init__(
        self,
        provider: str,
        api_key: Optional[str],
        model: str,
        concurrency_limit: int = 5,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.base_url = base_url

        if not self.base_url:
            if self.provider == "openai":
                self.base_url = "https://api.openai.com/v1/chat/completions"
            elif self.provider == "anthropic":
                self.base_url = "https://api.anthropic.com/v1/messages"
            elif self.provider == "ollama":
                self.base_url = "http://127.0.0.1:11434/api/generate"

    def _prepare_request(self, system_prompt: str, user_prompt: str) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}

        if self.provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
            }
        elif self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            payload = {
                "model": self.model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": 4000,
                "temperature": 0.3,
            }
        elif self.provider == "ollama":
            payload = {
                "model": self.model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {"temperature": 0.3},
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        req_data = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(self.base_url, data=req_data, headers=headers, method="POST")

    def _sync_http_post(self, request: urllib.request.Request) -> str:
        with urllib.request.urlopen(request, timeout=120) as resp:
            data = resp.read().decode("utf-8")
            parsed = json.loads(data)

            if self.provider == "openai":
                return parsed["choices"][0]["message"]["content"]
            elif self.provider == "anthropic":
                return parsed["content"][0]["text"]
            elif self.provider == "ollama":
                return parsed["response"]
            raise ValueError("Malformed response format")

    async def generate_article(self, seed: SeedRecord) -> str:
        system_prompt = (
            "You are an expert editorial technical writer and SEO engineer. "
            "Write comprehensive, high-value, factual markdown guides with natural structure. "
            "Avoid repetitive fluff, cliches, or boilerplate intro phrases. "
            "Always include at top:\n"
            "META_DESCRIPTION: <compelling, click-worthy 150-char description>\n"
            "# <Actionable Title with Primary Keyword>"
        )

        user_prompt = (
            f"Topic Target: {seed.primary_keyword}\n"
            f"Secondary Keywords: {', '.join(seed.secondary_keywords)}\n"
            f"Key Semantic Entities To Strictly Include: {', '.join(seed.entities)}\n"
            f"Search Intent: {seed.intent}\n"
            f"Target Audience: {seed.target_audience}\n"
            f"Requirements:\n"
            f"- Thorough guide with real-world examples, structured headings (H2, H3), and tables.\n"
            f"- Naturally ground and weave every key semantic entity.\n"
            f"- Maintain professional readability (target Flesch-Kincaid 8-11)."
        )

        request = self._prepare_request(system_prompt, user_prompt)

        async with self.semaphore:
            max_retries = 3
            backoff = 2.0
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Generating draft for slug '{seed.slug}' (Attempt {attempt})...")
                    return await asyncio.to_thread(self._sync_http_post, request)
                except Exception as err:
                    logger.warning(f"Error generating '{seed.slug}' (attempt {attempt}/{max_retries}): {err}")
                    if attempt == max_retries:
                        raise
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
            raise RuntimeError("Exceeded maximum retries.")


# ---------------------------------------------------------------------------
# Programmatic SEO Engine Pipeline
# ---------------------------------------------------------------------------
class ProgrammaticSEOEngine:
    def __init__(
        self,
        llm_client: AsyncLLMClient,
        validator: QualityGateValidator,
        output_dir: Path,
        ssg_flavor: str,
    ):
        self.client = llm_client
        self.validator = validator
        self.output_dir = output_dir
        self.ssg_flavor = ssg_flavor

        self.published_dir = self.output_dir / "published"
        self.quarantine_dir = self.output_dir / "quarantine"

        self.published_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    async def process_seed(self, seed: SeedRecord) -> GenerationResult:
        try:
            raw_text = await self.client.generate_article(seed)
        except Exception as e:
            logger.error(f"Generation completely failed for {seed.slug}: {e}")
            raw_text = f"# Generation Failure\n\nError: {str(e)}"

        title, meta_desc, body = SSGSerializer.extract_sections(raw_text)
        audit_results = self.validator.audit(body, seed)

        frontmatter = SSGSerializer.generate_frontmatter(
            flavor=self.ssg_flavor,
            seed=seed,
            title=title,
            meta_description=meta_desc,
        )

        final_markdown = SSGSerializer.assemble_markdown(frontmatter, body)

        if audit_results.is_valid:
            status = "PUBLISHED"
            out_file = self.published_dir / f"{seed.slug}.md"
            out_file.write_text(final_markdown, encoding="utf-8")
            logger.info(f"[PASS] Published: {seed.slug} -> {out_file}")
        else:
            status = "QUARANTINED"
            out_file = self.quarantine_dir / f"{seed.slug}.md"
            audit_file = self.quarantine_dir / f"{seed.slug}.audit.json"

            out_file.write_text(final_markdown, encoding="utf-8")
            audit_payload = {
                "seed": asdict(seed),
                "audit": asdict(audit_results),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            audit_file.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
            logger.warning(
                f"[QUARANTINE] {seed.slug} failed quality gates. Issues: {audit_results.failure_reasons}"
            )

        return GenerationResult(
            seed=seed,
            raw_response=raw_text,
            parsed_frontmatter=frontmatter,
            markdown_body=body,
            quality_audit=audit_results,
            status=status,
        )

    async def run(self, seeds: List[SeedRecord]) -> List[GenerationResult]:
        logger.info(f"Initiating programmatic pipeline for {len(seeds)} topics...")
        tasks = [self.process_seed(seed) for seed in seeds]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        published_count = sum(1 for r in results if r.status == "PUBLISHED")
        quarantine_count = sum(1 for r in results if r.status == "QUARANTINED")

        logger.info(
            f"Execution finished. Total: {len(results)} | "
            f"Published: {published_count} | Quarantined: {quarantine_count}"
        )
        return results


# ---------------------------------------------------------------------------
# CLI & Utilities
# ---------------------------------------------------------------------------
def load_seeds(file_path: Path) -> List[SeedRecord]:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    seeds: List[SeedRecord] = []
    for item in raw_data:
        seeds.append(
            SeedRecord(
                slug=item["slug"],
                primary_keyword=item["primary_keyword"],
                secondary_keywords=item.get("secondary_keywords", []),
                intent=item.get("intent", "Informational"),
                target_audience=item.get("target_audience", "General"),
                entities=item.get("entities", []),
                custom_metadata=item.get("custom_metadata", {}),
            )
        )
    return seeds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production Programmatic SEO Engine with Automated Quality Gates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON seed dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./seo_dist"),
        help="Target output directory for markdown assets.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider endpoint to target.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Specific model identifier (e.g., gpt-4o-mini, claude-3-haiku-20240307, llama3.1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key (defaults to reading API_KEY env variable).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override API base URL.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max simultaneous async workers.",
    )
    parser.add_argument(
        "--ssg-flavor",
        choices=["astro", "nextjs", "hugo"],
        default="astro",
        help="SSG schema specification for frontmatter headers.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=600,
        help="Minimum required word count.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=3500,
        help="Maximum allowed word count.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seeds = load_seeds(args.input)
    thresholds = QualityThresholds(
        min_word_count=args.min_words,
        max_word_count=args.max_words,
    )
    validator = QualityGateValidator(thresholds)
    llm_client = AsyncLLMClient(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        concurrency_limit=args.concurrency,
        base_url=args.base_url,
    )
    engine = ProgrammaticSEOEngine(
        llm_client=llm_client,
        validator=validator,
        output_dir=args.output_dir,
        ssg_flavor=args.ssg_flavor,
    )

    asyncio.run(engine.run(seeds))


if __name__ == "__main__":
    main()