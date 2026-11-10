"""Local pseudonymisation applied before anything is sent to the cloud (variant B).

Structured identifiers use regex recognisers. Names and places use Microsoft
Presidio + spaCy (nl/fr/en) when the `redaction` extra is installed. Without
Presidio only the regex recognisers run, so names are NOT redacted.
`Pseudonymiser.has_ner` tells you which mode you are in.

Addresses are regex-only in both modes: spaCy tags the city at best, never the
street, house number or postcode. Adjacent address pieces ("Kerkstraat 12",
"9000 Gent") are merged into one <ADDRESS_n> placeholder.

Placeholders are consistent within one document (same value -> same token) and
can be restored in the model's reply: `restore(reply, mapping)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# -- addresses (BE/NL/FR/UK formats) ------------------------------------------
_CAP = r"[A-ZÀ-Þ][\w'’.-]*"  # capitalised word
_NUM = r"\d{1,4}(?:\s?[a-zA-Z](?![\w]))?(?:\s*(?:bus|bte|boîte|b\.)\s*\w{1,4})?"  # 12, 12A, 4 bus 2
_NL_SUFFIX = (
    r"straat|laan|weg|plein|dreef|lei|steenweg|kaai|kade|markt|singel|gracht|dijk|baan|hof|pad|"
    r"plaats|park|veld|berg|ring|wal|steeg|vest|dorp|meers|akker|hoek|gaarde|heide|bos"
)
# Words that start multi-word Dutch street names: Grote Markt, Koning Albertlaan.
_NL_PREFIX = (
    r"Grote|Kleine|Oude|Nieuwe|Korte|Lange|Hoge|Lage|Sint|Onze-Lieve-Vrouw|Burgemeester|Koning|"
    r"Koningin|Prins|Prinses|Graaf|Gravin|Hertog|Kardinaal|Generaal|Kapitein|Dokter|Pastoor|Pater|"
    r"Professor|Van|Vander|De|Den|Ter|Ten|Het"
)
_FR_STREET = (
    r"rue|avenue|av\.|boulevard|bd|chaussée|place|chemin|allée|impasse|quai|route|square|clos|"
    r"drève|cours|passage|sentier|rond-point"
)
_FR_NAME = r"(?:(?:de|du|des|la|le|les|l'|l’|d'|d’)\s?)*[A-ZÀ-Þ]"  # de la Paix, Louise
_EN_STREET = (
    r"Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way|Close|Court|Ct|Place|Pl|Square|Sq|"
    r"Terrace|Crescent|Boulevard|Blvd|Gardens|Grove|Row|Mews|Hill"
)
ADDRESS_PATTERNS: list[re.Pattern] = [
    # nl/be: Kerkstraat 12, Grote Markt 1, Sint-Pietersnieuwstraat 33 bus 4
    re.compile(rf"\b(?:(?:{_NL_PREFIX})\s){{0,2}}(?=[A-ZÀ-Þ])[\w'’-]*?(?i:{_NL_SUFFIX})\s+{_NUM}(?!\w)"),
    # fr (number first): 12 rue de la Paix, 3bis avenue Louise
    re.compile(rf"\b\d{{1,4}}(?:\s?(?:bis|ter|[a-zA-Z]))?,?\s+(?i:{_FR_STREET})\s{_FR_NAME}[^,\n.;:!?]{{1,40}}"),
    # fr/be (number last): rue de la Loi 16, avenue Louise 54 bte 3
    re.compile(rf"\b(?i:{_FR_STREET})\s{_FR_NAME}[^,\n\d.;:!?]{{0,40}}?\s{_NUM}(?!\w)"),
    # en: 221B Baker Street, 10 Downing St
    re.compile(rf"\b\d{{1,5}}[A-Za-z]?\s+(?:{_CAP}\s){{1,3}}(?:{_EN_STREET})\b\.?"),
    # postcode + city: 9000 Gent, 75002 Paris, 1011 AB Amsterdam (not amounts: 1500 EUR)
    # A number that looks like a year (19xx/20xx) only counts after a comma, a colon
    # or at a line start, so "In 2025 Amazon was faster" is left alone.
    re.compile(
        r"(?:(?<=[,:]\s)|(?<=[,:])|^|(?!(?:19|20)\d{2}\b))"
        r"\b(?:[1-9]\d{3}(?:\s?[A-Z]{2}(?![\w]))?|\d{5})\s+"
        r"(?!(?:EUR|Euro|euro|USD|GBP)\b)[A-ZÀ-Þ][\w'’-]+(?:[ -](?:sur|sous|en|aan|op|de|la|le|[A-ZÀ-Þ][\w'’-]+)){0,3}",
        re.MULTILINE,
    ),
    # UK postcode: NW1 6XE, SW1A 2AA
    re.compile(r"\b(?:[A-Z][a-z]+\s)?[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"),  # (London) NW1 6XE
]

# (entity type, compiled pattern). Earlier entries win on overlap at equal length.
BASE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),
    # Belgian national register number: YY.MM.DD-XXX.CC
    ("NATIONAL_ID", re.compile(r"\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}\b")),
    # BE/NL/FR mobile and landline: 0475 12 34 56, 09 123 45 67, +32 9 123 45 67,
    # 0032 478 11 22 33. The lookahead keeps dates such as 05.11.2026 out.
    (
        "PHONE",
        re.compile(
            r"(?<![\w.,])(?!\d{2}[./-]\d{2}[./-]\d{2,4}\b)"
            r"(?:\+\d{2}|00\d{2}|0)[\s./-]?\d{1,3}(?:[\s./-]?\d{2,3}){2,4}(?!\w)"
        ),
    ),
    *[("ADDRESS", p) for p in ADDRESS_PATTERNS],
]

# Customer/order IDs differ per company: adjust or extend for your own format.
DEFAULT_ID_PATTERNS: list[tuple[str, str]] = [
    ("ORDER", r"\b(?:ORD|BE|NL|FR)-?\d{6,10}\b"),
    ("CUSTOMER_ID", r"\bCUST-?\d{5,10}\b"),
]

NER_MODELS = {"en": "en_core_web_md", "nl": "nl_core_news_md", "fr": "fr_core_news_md"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str


class Pseudonymiser:
    def __init__(
        self,
        use_ner: bool = True,
        id_patterns: list[tuple[str, str]] | None = None,
    ):
        self.patterns = list(BASE_PATTERNS) + [
            (t, re.compile(p)) for t, p in (id_patterns or DEFAULT_ID_PATTERNS)
        ]
        self._analyzer = None
        self.has_ner = False
        if use_ner:
            try:
                self._analyzer = self._build_analyzer()
                self.has_ner = True
            except Exception:  # presidio/spacy/models missing
                self._analyzer = None

    @staticmethod
    def _build_analyzer():
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        cfg = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": k, "model_name": v} for k, v in NER_MODELS.items()],
        }
        nlp = NlpEngineProvider(nlp_configuration=cfg).create_engine()
        return AnalyzerEngine(nlp_engine=nlp, supported_languages=list(NER_MODELS))

    # -- detection ---------------------------------------------------------
    def detect(self, text: str, language: str = "en") -> list[Span]:
        spans = [
            Span(m.start(), m.end(), t) for t, pat in self.patterns for m in pat.finditer(text)
        ]
        if self._analyzer is not None and language in NER_MODELS:
            keys = _json_key_ranges(text)
            for r in self._analyzer.analyze(
                text=text, language=language, entities=["PERSON", "LOCATION"]
            ):
                if not any(k0 <= r.start and r.end <= k1 for k0, k1 in keys):
                    spans.append(Span(r.start, r.end, r.entity_type))
        return _merge_addresses(_resolve_overlaps(spans), text)

    # -- transform ---------------------------------------------------------
    def redact(
        self, text: str, language: str = "en", mapping: dict[str, str] | None = None
    ) -> tuple[str, dict[str, str]]:
        """Returns (redacted text, mapping placeholder -> original value).

        Pass the mapping of an earlier message to keep placeholders consistent
        across a conversation: known values reuse their placeholder, new ones
        get the next number. The returned mapping includes the old entries.
        """
        spans = self.detect(text, language)
        mapping = dict(mapping or {})
        counters: dict[str, int] = {}
        by_value: dict[tuple[str, str], str] = {}
        for ph, value in mapping.items():
            m = re.fullmatch(r"<(\w+)_(\d+)>", ph)
            if m:
                counters[m[1]] = max(counters.get(m[1], 0), int(m[2]))
                by_value[(m[1], value.lower())] = ph
        pieces: list[str] = []
        pos = 0
        for s in spans:  # sorted by start
            value = text[s.start : s.end]
            key = (s.type, value.lower())
            if key not in by_value:
                counters[s.type] = counters.get(s.type, 0) + 1
                by_value[key] = f"<{s.type}_{counters[s.type]}>"
                mapping[by_value[key]] = value
            pieces.append(text[pos : s.start])
            pieces.append(by_value[key])
            pos = s.end
        pieces.append(text[pos:])
        return "".join(pieces), mapping

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        for placeholder, value in mapping.items():
            text = text.replace(placeholder, value)
        return text


# A quoted JSON object key: "delivered_on": ... The customer-facts block and tier-5
# tool results are JSON, and spaCy tags field names such as "delivered_on" or
# "shipping_charged_eur" as PERSON. Replacing a key hides what the value means,
# so NER hits inside keys are ignored. Regex identifiers (order ids...) still apply.
_JSON_KEY = re.compile(r'"((?:[^"\\\n]|\\.)*)"(?=\s*:)')


def _json_key_ranges(text: str) -> list[tuple[int, int]]:
    return [(m.start(1), m.end(1)) for m in _JSON_KEY.finditer(text)]


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Keep the longest span where several overlap; return sorted by start."""
    kept: list[Span] = []
    for s in sorted(spans, key=lambda x: (-(x.end - x.start), x.start)):
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda x: x.start)


_ADDRESS_GLUE = re.compile(r"[\s,]{0,3}")
_ADDRESS_PARTS = {"ADDRESS", "LOCATION"}


def _merge_addresses(spans: list[Span], text: str) -> list[Span]:
    """Join an ADDRESS span with neighbouring ADDRESS/LOCATION spans ("Kerkstraat 12, 9000 Gent").

    One placeholder per address keeps the prompt readable and avoids leaving a
    street and its city as separate, recombinable pieces. A lone city stays LOCATION.
    """
    out: list[Span] = []
    for s in spans:
        prev = out[-1] if out else None
        if (
            prev
            and prev.type in _ADDRESS_PARTS
            and s.type in _ADDRESS_PARTS
            and "ADDRESS" in (prev.type, s.type)
            and _ADDRESS_GLUE.fullmatch(text[prev.end : s.start])
        ):
            out[-1] = Span(prev.start, s.end, "ADDRESS")
        else:
            out.append(s)
    return [_absorb_city(s, text, out[i + 1].start if i + 1 < len(out) else len(text))
            if s.type == "ADDRESS" else s for i, s in enumerate(out)]


# ", Bristol" or ", Sint-Niklaas" right after a street, ending the clause. NER
# misses such a city depending on context, which would leave it next to <ADDRESS_n>.
_TRAILING_CITY = re.compile(r",\s?[A-ZÀ-Þ][\w'’]+(?:[ -][A-ZÀ-Þ][\w'’]+){0,2}(?=\s*(?:$|[.,;:!?\n]))")


def _absorb_city(s: Span, text: str, limit: int) -> Span:
    m = _TRAILING_CITY.match(text, s.end)
    return Span(s.start, m.end(), s.type) if m and m.end() <= limit else s
