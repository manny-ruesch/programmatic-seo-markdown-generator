#!/usr/bin/env python3
"""
Programmatic SEO Engine with Automated Quality Gates
File: generator.py

Generates SEO-optimized Markdown content at scale from structured datasets
while enforcing automated quality gates to prevent keyword cannibalization,
repetitive AI cadences, hallucinated placeholders, and thin content penalties.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SEOEngine")


@dataclass
class QualityGateConfig:
    min_word_count: int = 650
    max_word_count: int = 3500
    min_type_token_ratio: float = 0.38  # Lexical diversity
    max_ngram_repetition_rate: float = 0.15  # Tri-gram repetition threshold
    min_heading_count: int = 3  # At least H1 + 2 H2s
    meta_description_min_len: int = 120
    meta_description_max_len: int = 160
    strict_mode: bool = False  # If True, any gate failure rejects page completely
    banned_phrases: List[str] = field(
        default_factory=lambda: [
            r"in conclusion",
            r"it is important to remember",
            r"delve into",
            r"testament to",
            r"beacon of",
            r"in the fast-paced world",
            r"tapestry",
            r"look no further",
            r"unlock the secrets",
        ]
    )


@dataclass
class GateEvaluation:
    gate_name: str
    passed: bool
    score: float
    threshold: float
    message: str


@dataclass
class PageResult:
    slug: str
    target_keyword: str
    passed: bool
    status: str  # "published", "flagged_review", "rejected"
    evaluations: List[GateEvaluation] = field(default_factory=list)
    word_count: int = 0
    ttr_score: float = 0.0
    output_path: Optional[str] = None


class QualityGates:
    """Automated content auditing engine ensuring pages satisfy rigorous SEO standards."""

    def __init__(self, config: QualityGateConfig):
        self.config = config
        self._compiled_banned = [
            re.compile(rf"\b{phrase}\b", re.IGNORECASE)
            for phrase in self.config.banned_phrases
        ]

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase alphanumeric tokenization
        return re.findall(r"\b[a-z0-9_-]+\b", text.lower())

    def check_word_count(self, content: str) -> GateEvaluation:
        words = self._tokenize(content)
        count = len(words)
        passed = self.config.min_word_count <= count <= self.config.max_word_count
        return GateEvaluation(
            gate_name="Word Count Check",
            passed=passed,
            score=float(count),
            threshold=float(self.config.min_word_count),
            message=f"Word count is {count} (Target: {self.config.min_word_count}-{self.config.max_word_count})",
        )

    def check_lexical_diversity(self, content: str) -> GateEvaluation:
        words = self._tokenize(content)
        if not words:
            return GateEvaluation(
                "Lexical Diversity (TTR)", False, 0.0, self.config.min_type_token_ratio, "Empty content"
            )
        
        unique_tokens = set(words)
        ttr = len(unique_tokens) / len(words)
        passed = ttr >= self.config.min_type_token_ratio
        return GateEvaluation(
            gate_name="Lexical Diversity (TTR)",
            passed=passed,
            score=round(ttr, 3),
            threshold=self.config.min_type_token_ratio,
            message=f"Type-Token Ratio is {ttr:.3f} (Min: {self.config.min_type_token_ratio})",
        )

    def check_ngram_repetition(self, content: str, n: int = 3) -> GateEvaluation:
        words = self._tokenize(content)
        if len(words) < n:
            return GateEvaluation("N-gram Repetition", True, 0.0, self.config.max_ngram_repetition_rate, "Too short")
        
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated_ngrams = sum(c for c in counts.values() if c > 1)
        repetition_rate = repeated_ngrams / len(ngrams)
        passed = repetition_rate <= self.config.max_ngram_repetition_rate
        return GateEvaluation(
            gate_name=f"{n}-gram Repetition Gate",
            passed=passed,
            score=round(repetition_rate, 3),
            threshold=self.config.max_ngram_repetition_rate,
            message=f"N-gram repetition rate is {repetition_rate:.3f} (Max: {self.config.max_ngram_repetition_rate})",
        )

    def check_unresolved_placeholders(self, content: str) -> GateEvaluation:
        # Detect unrendered template tags like {{ var }}, [Insert X], <TODO>
        placeholder_pattern = re.compile(
            r"(\{\{[^}]+\}\}|\[insert\s+[^\]]+\]|<TODO[^>]*>|UNDEFINED|NULL)",
            re.IGNORECASE,
        )
        matches = placeholder_pattern.findall(content)
        passed = len(matches) == 0
        return GateEvaluation(
            gate_name="Placeholder & Leak Check",
            passed=passed,
            score=float(len(matches)),
            threshold=0.0,
            message=f"Found {len(matches)} unrendered placeholders: {matches[:3]}",
        )

    def check_structural_headings(self, content: str) -> GateEvaluation:
        # Count Markdown H1, H2, H3
        headings = re.findall(r"^#{1,3}\s+.+$", content, flags=re.MULTILINE)
        count = len(headings)
        passed = count >= self.config.min_heading_count
        return GateEvaluation(
            gate_name="Heading Hierarchy Check",
            passed=passed,
            score=float(count),
            threshold=float(self.config.min_heading_count),
            message=f"Found {count} headings (Min: {self.config.min_heading_count})",
        )

    def check_ai_clichés(self, content: str) -> GateEvaluation:
        found = []
        for pattern in self._compiled_banned:
            matches = pattern.findall(content)
            if matches:
                found.extend(matches)
        passed = len(found) == 0
        return GateEvaluation(
            gate_name="AI Cliché Detection",
            passed=passed,
            score=float(len(found)),
            threshold=0.0,
            message=f"Detected {len(found)} banned clichés: {list(set(found))[:3]}",
        )

    def check_metadata(self, meta: Dict[str, Any]) -> GateEvaluation:
        desc = meta.get("meta_description", "").strip()
        length = len(desc)
        passed = self.config.meta_description_min_len <= length <= self.config.meta_description_max_len
        return GateEvaluation(
            gate_name="Meta Description Gate",
            passed=passed,
            score=float(length),
            threshold=float(self.config.meta_description_min_len),
            message=f"Meta description length is {length} chars (Target: {self.config.meta_description_min_len}-{self.config.meta_description_max_len})",
        )

    def run_all(self, content: str, meta: Dict[str, Any]) -> Tuple[bool, List[GateEvaluation]]:
        evals = [
            self.check_word_count(content),
            self.check_lexical_diversity(content),
            self.check_ngram_repetition(content),
            self.check_unresolved_placeholders(content),
            self.check_structural_headings(content),
            self.check_ai_clichés(content),
            self.check_metadata(meta),
        ]
        all_passed = all(e.passed for e in evals)
        return all_passed, evals


class SimpleTemplateEngine:
    """Safe, lightweight template engine supporting variables and conditional blocks."""

    @staticmethod
    def render(template_str: str, context: Dict[str, Any]) -> str:
        content = template_str

        # Handle simple boolean condition blocks: {% if var %}...{% endif %}
        def replace_conditional(match):
            condition_var = match.group(1).strip()
            block_content = match.group(2)
            val = context.get(condition_var)
            if val and val not in ("false", "False", "0", 0, None):
                return block_content
            return ""

        content = re.sub(
            r"\{%\s*if\s+([a-zA-Z0-9_]+)\s*%\}(.*?)\{%\s*endif\s*%\}",
            replace_conditional,
            content,
            flags=re.DOTALL,
        )

        # Handle simple variable interpolation: {{ var }}
        def replace_var(match):
            key = match.group(1).strip()
            return str(context.get(key, f"{{{{ {key} }}}}"))

        content = re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", replace_var, content)
        return content


class ProgrammaticSEOGenerator:
    """Orchestrates programmatic content generation, quality gates validation, and file output."""

    def __init__(
        self,
        data_source: Path,
        template_file: Path,
        output_dir: Path,
        config: Optional[QualityGateConfig] = None,
    ):
        self.data_source = Path(data_source)
        self.template_file = Path(template_file)
        self.output_dir = Path(output_dir)
        self.config = config or QualityGateConfig()
        self.gates = QualityGates(self.config)
        
        # Deduplication registry to prevent keyword cannibalization and collision
        self.seen_slugs: Set[str] = set()
        self.seen_intents: Set[str] = set()

    def _load_data(self) -> List[Dict[str, Any]]:
        if not self.data_source.exists():
            raise FileNotFoundError(f"Data source not found: {self.data_source}")

        rows: List[Dict[str, Any]] = []
        suffix = self.data_source.suffix.lower()

        if suffix == ".json":
            with open(self.data_source, "r", encoding="utf-8") as f:
                data = json.load(f)
                rows = data if isinstance(data, list) else [data]
        elif suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            with open(self.data_source, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = [dict(row) for row in reader]
        else:
            raise ValueError(f"Unsupported format '{suffix}'. Provide .json, .csv, or .tsv.")

        return rows

    def _slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[-\s]+", "-", text)

    def _generate_frontmatter(self, meta: Dict[str, Any], gate_score: float, status: str) -> str:
        yaml_lines = ["---"]
        for k, v in meta.items():
            if isinstance(v, (int, float, bool)):
                yaml_lines.append(f"{k}: {v}")
            else:
                clean_v = str(v).replace('"', '\\"')
                yaml_lines.append(f'{k}: "{clean_v}"')
        yaml_lines.append(f"seo_quality_status: {status}")
        yaml_lines.append(f"seo_quality_score: {gate_score:.2f}")
        yaml_lines.append(f'generated_at: "{datetime.now(timezone.utc).isoformat()}"')
        yaml_lines.append("---\n\n")
        return "\n".join(yaml_lines)

    def run(self) -> List[PageResult]:
        dataset = self._load_data()
        template_content = self.template_file.read_text(encoding="utf-8")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results: List[PageResult] = []

        logger.info(f"Loaded {len(dataset)} programmatic records. Executing generation pipeline...")

        for idx, row in enumerate(dataset, start=1):
            keyword = row.get("target_keyword", row.get("keyword", f"record-{idx}")).strip()
            slug = row.get("slug") or self._slugify(keyword)

            # Prevent keyword cannibalization within the batch
            intent_key = keyword.lower()
            if intent_key in self.seen_intents:
                logger.warning(f"Cannibalization Guard: duplicate keyword intent detected for '{keyword}'. Skipping row {idx}.")
                results.append(
                    PageResult(
                        slug=slug,
                        target_keyword=keyword,
                        passed=False,
                        status="rejected",
                        evaluations=[GateEvaluation("Cannibalization Check", False, 0.0, 0.0, "Duplicate search intent in batch")],
                    )
                )
                continue

            if slug in self.seen_slugs:
                slug = f"{slug}-{idx}"
            self.seen_slugs.add(slug)
            self.seen_intents.add(intent_key)

            # Render template
            rendered_body = SimpleTemplateEngine.render(template_content, row)

            # Build metadata
            meta = {
                "title": row.get("meta_title", row.get("title", keyword.title())),
                "slug": slug,
                "target_keyword": keyword,
                "meta_description": row.get("meta_description", "").strip(),
                "canonical_url": row.get("canonical_url", f"https://example.com/{slug}/"),
            }

            # Evaluate Quality Gates
            all_passed, evaluations = self.gates.run_all(rendered_body, meta)
            words = len(re.findall(r"\b\w+\b", rendered_body))
            ttr = next((e.score for e in evaluations if e.gate_name.startswith("Lexical")), 0.0)

            # Scoring calculation (0 - 100)
            passed_count = sum(1 for e in evaluations if e.passed)
            quality_score = (passed_count / len(evaluations)) * 100

            # Determine final status
            if all_passed:
                status = "published"
            elif not self.config.strict_mode and passed_count >= (len(evaluations) - 2):
                status = "flagged_review"
            else:
                status = "rejected"

            output_file = None
            if status in ("published", "flagged_review"):
                frontmatter = self._generate_frontmatter(meta, quality_score, status)
                full_markdown = frontmatter + rendered_body
                output_file = self.output_dir / f"{slug}.md"
                output_file.write_text(full_markdown, encoding="utf-8")
                logger.info(f"[{status.upper()}] Generated: {output_file.name} | Score: {quality_score:.1f}%")
            else:
                logger.warning(f"[{status.upper()}] Rejected page for '{keyword}': Score {quality_score:.1f}%")

            results.append(
                PageResult(
                    slug=slug,
                    target_keyword=keyword,
                    passed=all_passed,
                    status=status,
                    evaluations=evaluations,
                    word_count=words,
                    ttr_score=ttr,
                    output_path=str(output_file) if output_file else None,
                )
            )

        return results


def export_audit_manifest(results: List[PageResult], output_path: Path) -> None:
    manifest_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(results),
        "published": sum(1 for r in results if r.status == "published"),
        "flagged_review": sum(1 for r in results if r.status == "flagged_review"),
        "rejected": sum(1 for r in results if r.status == "rejected"),
        "pages": [asdict(r) for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Audit manifest written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Programmatic SEO Engine with Automated Quality Gates."
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to CSV or JSON file containing page variables.",
    )
    parser.add_argument(
        "--template",
        "-t",
        type=str,
        required=True,
        help="Path to Markdown template file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./dist",
        help="Output directory for generated markdown files.",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        type=str,
        default="./audit_manifest.json",
        help="Path to output the QA audit manifest file.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=650,
        help="Minimum required word count per article.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: reject files if any single quality gate fails.",
    )

    args = parser.parse_args()

    config = QualityGateConfig(
        min_word_count=args.min_words,
        strict_mode=args.strict,
    )

    try:
        generator = ProgrammaticSEOGenerator(
            data_source=Path(args.data),
            template_file=Path(args.template),
            output_dir=Path(args.output),
            config=config,
        )

        results = generator.run()
        export_audit_manifest(results, Path(args.manifest))

        published_count = sum(1 for r in results if r.status == "published")
        flagged_count = sum(1 for r in results if r.status == "flagged_review")
        rejected_count = sum(1 for r in results if r.status == "rejected")

        print("\n==========================================")
        print(" PROGRAMMATIC SEO BATCH SUMMARY")
        print("==========================================")
        print(f"Total Processed: {len(results)}")
        print(f"Published (Passed):       {published_count}")
        print(f"Flagged for Manual Review: {flagged_count}")
        print(f"Rejected:                 {rejected_count}")
        print("==========================================\n")

    except Exception as e:
        logger.error(f"Fatal error executing pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()