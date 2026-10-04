import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter


@dataclass
class HeadingNode:
    level: int
    text: str
    line_number: int


@dataclass
class HeadingReport:
    is_valid: bool
    h1_count: int
    headings: List[HeadingNode]
    hierarchy_errors: List[str]


@dataclass
class KeywordMetrics:
    keyword: str
    count: int
    density_percentage: float
    is_acceptable: bool
    status_message: str


@dataclass
class ReadabilityScores:
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    total_words: int
    total_sentences: int
    total_syllables: int
    reading_ease_rating: str


@dataclass
class AIPatternReport:
    detected_cliches: Dict[str, int]
    cliche_density_score: float
    repetitive_starters: List[Tuple[str, int]]
    has_excessive_ai_markers: bool
    flags: List[str]


@dataclass
class QualityGateResult:
    passed: bool
    total_word_count: int
    word_count_passed: bool
    heading_report: HeadingReport
    keyword_metrics: Dict[str, KeywordMetrics]
    readability: ReadabilityScores
    readability_passed: bool
    ai_pattern_report: AIPatternReport
    failure_reasons: List[str]


class SEOQualityFilter:
    """
    Evaluates Markdown content for programmatic SEO quality standards:
    - Content volume and heading integrity
    - Keyword optimization and density safety bounds
    - Readability metrics (Flesch Reading Ease and Flesch-Kincaid Grade Level)
    - Detection of generic repetitive AI tropes and structural footprints
    """

    AI_CLICHE_PHRASES = [
        "delve into", "delving into", "testament to", "rich tapestry", 
        "tapestry of", "in conclusion", "it is important to note", 
        "it is worth noting", "crucial to remember", "a beacon of", 
        "navigating the", "navigating this", "game-changer", "game changer", 
        "realm of", "bustling hub", "beacon of hope", "look no further", 
        "moreover", "furthermore", "in summary", "unlocking the", 
        "seamlessly blend", "embark on a journey", "foster a", "harnessing the power"
    ]

    def __init__(
        self,
        min_word_count: int = 800,
        max_word_count: int = 4000,
        min_flesch_reading_ease: float = 50.0,
        max_flesch_kincaid_grade: float = 12.0,
        target_keyword_density_min: float = 0.5,
        target_keyword_density_max: float = 2.5,
        max_ai_cliche_density: float = 1.2,
    ):
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        self.min_flesch_reading_ease = min_flesch_reading_ease
        self.max_flesch_kincaid_grade = max_flesch_kincaid_grade
        self.target_keyword_density_min = target_keyword_density_min
        self.target_keyword_density_max = target_keyword_density_max
        self.max_ai_cliche_density = max_ai_cliche_density

    def clean_markdown(self, markdown_text: str) -> str:
        """Strips markdown code blocks, links, images, and formatting to extract plain text."""
        text = re.sub(r'```[\s\S]*?```', '', markdown_text)
        text = re.sub(r'`[^`]*`', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'^\s{0,3}>\s?', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*#{1,6}\s+(.*)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[-*_]{3,}', '', text)
        return text

    def analyze_headings(self, markdown_text: str) -> HeadingReport:
        lines = markdown_text.splitlines()
        headings: List[HeadingNode] = []
        errors: List[str] = []

        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

        for idx, line in enumerate(lines, start=1):
            match = heading_pattern.match(line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append(HeadingNode(level=level, text=title, line_number=idx))

        h1_headings = [h for h in headings if h.level == 1]
        h1_count = len(h1_headings)

        if h1_count == 0:
            errors.append("Document lacks an H1 tag.")
        elif h1_count > 1:
            errors.append(f"Multiple H1 tags found ({h1_count}). Exactly one H1 is required.")

        prev_level = 0
        for h in headings:
            if prev_level > 0:
                if h.level > prev_level + 1:
                    errors.append(
                        f"Heading hierarchy skip detected at line {h.line_number}: "
                        f"jumped from H{prev_level} to H{h.level} ('{h.text}')."
                    )
            prev_level = h.level

        is_valid = len(errors) == 0
        return HeadingReport(
            is_valid=is_valid,
            h1_count=h1_count,
            headings=headings,
            hierarchy_errors=errors
        )

    def _count_syllables_in_word(self, word: str) -> int:
        word = word.lower().strip(".:;?!'\"()[]{}")
        if not word:
            return 0
        if len(word) <= 3:
            return 1

        word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
        word = re.sub(r'^y', '', word)
        syllables = len(re.findall(r'[aeiouy]{1,2}', word))
        return max(1, syllables)

    def calculate_readability(self, plain_text: str) -> ReadabilityScores:
        sentences = [s.strip() for s in re.split(r'[.!?]+(?:\s+|$)', plain_text) if s.strip()]
        words = [w.strip() for w in re.findall(r'\b[a-zA-Z0-9\'-]+\b', plain_text) if w.strip()]

        total_sentences = max(1, len(sentences))
        total_words = max(1, len(words))
        total_syllables = sum(self._count_syllables_in_word(w) for w in words)

        flesch_reading_ease = (
            206.835 
            - 1.015 * (total_words / total_sentences) 
            - 84.6 * (total_syllables / total_words)
        )
        flesch_reading_ease = round(max(0.0, min(100.0, flesch_reading_ease)), 2)

        flesch_kincaid_grade = (
            0.39 * (total_words / total_sentences) 
            + 11.8 * (total_syllables / total_words) 
            - 15.59
        )
        flesch_kincaid_grade = round(max(0.0, flesch_kincaid_grade), 2)

        if flesch_reading_ease >= 80:
            rating = "Very Easy"
        elif flesch_reading_ease >= 60:
            rating = "Standard / Plain English"
        elif flesch_reading_ease >= 50:
            rating = "Fairly Difficult"
        elif flesch_reading_ease >= 30:
            rating = "Difficult"
        else:
            rating = "Very Confusing"

        return ReadabilityScores(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            total_words=total_words,
            total_sentences=total_sentences,
            total_syllables=total_syllables,
            reading_ease_rating=rating
        )

    def evaluate_keyword_density(
        self, 
        plain_text: str, 
        keywords: List[str]
    ) -> Dict[str, KeywordMetrics]:
        results: Dict[str, KeywordMetrics] = {}
        words = re.findall(r'\b[a-zA-Z0-9\'-]+\b', plain_text.lower())
        total_words = max(1, len(words))
        text_lower = plain_text.lower()

        for kw in keywords:
            kw_clean = kw.strip().lower()
            if not kw_clean:
                continue

            kw_words = re.findall(r'\b[a-zA-Z0-9\'-]+\b', kw_clean)
            if not kw_words:
                continue

            kw_pattern = r'\b' + r'\s+'.join([re.escape(w) for w in kw_words]) + r'\b'
            matches = len(re.findall(kw_pattern, text_lower))
            density = round((matches * len(kw_words) / total_words) * 100, 2)

            if density < self.target_keyword_density_min:
                acceptable = False
                msg = f"Under-optimized ({density}% vs min {self.target_keyword_density_min}%)"
            elif density > self.target_keyword_density_max:
                acceptable = False
                msg = f"Keyword stuffing risk ({density}% vs max {self.target_keyword_density_max}%)"
            else:
                acceptable = True
                msg = f"Optimal density ({density}%)"

            results[kw] = KeywordMetrics(
                keyword=kw,
                count=matches,
                density_percentage=density,
                is_acceptable=acceptable,
                status_message=msg
            )

        return results

    def detect_ai_patterns(self, plain_text: str) -> AIPatternReport:
        text_lower = plain_text.lower()
        cliche_matches: Dict[str, int] = {}
        total_cliche_count = 0

        for phrase in self.AI_CLICHE_PHRASES:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                cliche_matches[phrase] = count
                total_cliche_count += count

        sentences = [s.strip() for s in re.split(r'[.!?]+(?:\s+|$)', plain_text) if s.strip()]
        starters: List[str] = []
        for s in sentences:
            tokens = re.findall(r'\b[a-zA-Z]+\b', s)
            if len(tokens) >= 2:
                starters.append(f"{tokens[0]} {tokens[1]}".lower())
            elif tokens:
                starters.append(tokens[0].lower())

        starter_counts = Counter(starters)
        excessive_starters = [
            (starter, count) for starter, count in starter_counts.items() 
            if count >= 4 and (count / max(1, len(sentences))) > 0.15
        ]

        total_words = max(1, len(re.findall(r'\b[a-zA-Z0-9\'-]+\b', plain_text)))
        cliche_density = round((total_cliche_count / (total_words / 100)), 2)

        flags: List[str] = []
        if cliche_density > self.max_ai_cliche_density:
            flags.append(f"High AI boilerplate density: {cliche_density} per 100 words.")

        if excessive_starters:
            for phrase, count in excessive_starters:
                flags.append(f"Monotonous sentence structure: '{phrase}' repeated {count} times.")

        has_excessive_ai = (cliche_density > self.max_ai_cliche_density) or (len(excessive_starters) > 1)

        return AIPatternReport(
            detected_cliches=cliche_matches,
            cliche_density_score=cliche_density,
            repetitive_starters=excessive_starters,
            has_excessive_ai_markers=has_excessive_ai,
            flags=flags
        )

    def evaluate(
        self, 
        markdown_text: str, 
        primary_keyword: Optional[str] = None, 
        secondary_keywords: Optional[List[str]] = None
    ) -> QualityGateResult:
        plain_text = self.clean_markdown(markdown_text)
        failures: List[str] = []

        headings = self.analyze_headings(markdown_text)
        if not headings.is_valid:
            failures.extend(headings.hierarchy_errors)

        readability = self.calculate_readability(plain_text)
        
        word_count_passed = self.min_word_count <= readability.total_words <= self.max_word_count
        if not word_count_passed:
            failures.append(
                f"Word count violation: Document contains {readability.total_words} words "
                f"(Target: {self.min_word_count} - {self.max_word_count})."
            )

        readability_passed = (
            readability.flesch_reading_ease >= self.min_flesch_reading_ease 
            and readability.flesch_kincaid_grade <= self.max_flesch_kincaid_grade
        )
        if not readability_passed:
            if readability.flesch_reading_ease < self.min_flesch_reading_ease:
                failures.append(
                    f"Readability failed: Flesch Reading Ease is {readability.flesch_reading_ease} "
                    f"(Minimum acceptable: {self.min_flesch_reading_ease})."
                )
            if readability.flesch_kincaid_grade > self.max_flesch_kincaid_grade:
                failures.append(
                    f"Readability failed: Flesch-Kincaid Grade Level is {readability.flesch_kincaid_grade} "
                    f"(Maximum acceptable: {self.max_flesch_kincaid_grade})."
                )

        target_keywords = []
        if primary_keyword:
            target_keywords.append(primary_keyword)
        if secondary_keywords:
            target_keywords.extend(secondary_keywords)

        keyword_metrics = self.evaluate_keyword_density(plain_text, target_keywords)
        for kw, km in keyword_metrics.items():
            if not km.is_acceptable:
                failures.append(f"Keyword '{kw}': {km.status_message}")

        ai_report = self.detect_ai_patterns(plain_text)
        if ai_report.has_excessive_ai_markers:
            failures.extend(ai_report.flags)

        overall_passed = len(failures) == 0

        return QualityGateResult(
            passed=overall_passed,
            total_word_count=readability.total_words,
            word_count_passed=word_count_passed,
            heading_report=headings,
            keyword_metrics=keyword_metrics,
            readability=readability,
            readability_passed=readability_passed,
            ai_pattern_report=ai_report,
            failure_reasons=failures
        )


if __name__ == "__main__":
    sample_markdown = """
# Mastering Technical Programmatic SEO

Navigating the landscape of modern digital search requires scalable systems. In conclusion, building high-volume websites demands precision.

## Why Programmatic SEO Matters

Delve into automated systems and discover how structured content scales efficiently. Programmatic SEO creates hundreds of pages mapped directly to query intent.
Programmatic SEO allows teams to target long-tail variants without manual friction. It is worth noting that quality matters.

### Architectural Foundations

A standard pipeline converts data models into index-ready markdown artifacts. Let us look at what makes this functional:
1. Structured validation schemas
2. Templating engines
3. Strict quality filtering mechanisms

### Automated Testing

To succeed, automated filters must evaluate keyword metrics, structure, and text density before publishing.
"""

    evaluator = SEOQualityFilter(
        min_word_count=50,
        max_word_count=500,
        min_flesch_reading_ease=45.0,
        max_flesch_kincaid_grade=14.0,
        target_keyword_density_min=0.5,
        target_keyword_density_max=3.5,
    )

    result = evaluator.evaluate(
        markdown_text=sample_markdown,
        primary_keyword="programmatic seo",
        secondary_keywords=["data models", "quality"]
    )

    print(f"Quality Gate Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Word Count: {result.total_word_count}")
    print(f"Flesch Reading Ease: {result.readability.flesch_reading_ease}")
    print(f"Flesch-Kincaid Grade: {result.readability.flesch_kincaid_grade}")
    print(f"Heading Structure Valid: {result.heading_report.is_valid}")
    for kw, stats in result.keyword_metrics.items():
        print(f"  - [{kw}]: {stats.density_percentage}% (Count: {stats.count}) -> {stats.status_message}")
    if result.ai_pattern_report.detected_cliches:
        print(f"Detected AI Clichés: {result.ai_pattern_report.detected_cliches}")
    if result.failure_reasons:
        print("Failure Reasons:")
        for r in result.failure_reasons:
            print(f"  * {r}")
