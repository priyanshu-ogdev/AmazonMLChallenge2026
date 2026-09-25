"""
Country-agnostic text normalization for entity resolution (Stage 0).

Design principle: NO country-specific branching. Every operation must work
on US, India, AND unseen France text without hardcoded country logic.

Per the official problem statement: country must be treated as an open set
of string labels — never hard-coded, filtered, or one-hot encoded to only
{US, India}.

Operations:
1. Unicode normalization (NFKD) for consistent character representation
2. Lowercase + whitespace collapse
3. Punctuation stripping (preserve meaningful chars like /)
4. Bidirectional legal-suffix canonicalization (Corp↔Corporation)
5. Address abbreviation canonicalization (Rd↔Road)
6. Landmark phrase stripping ("Near SBI ATM" → removed)
7. & ↔ and replacement
"""

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Legal suffix canonicalization (bidirectional)
# Maps all variants to a single canonical form
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_MAP = {
    # Corporation variants
    "corp": "corporation", "corpn": "corporation", "corporation": "corporation",
    # Incorporated variants
    "inc": "incorporated", "incorporated": "incorporated", "incorp": "incorporated",
    # Limited variants
    "ltd": "limited", "limited": "limited", "ltda": "limited",
    # Private variants
    "pvt": "private", "private": "private", "pvte": "private",
    # Company variants
    "co": "company", "company": "company",
    # LLC / LLP (already canonical)
    "llc": "llc", "l.l.c": "llc", "l.l.c.": "llc",
    "llp": "llp", "l.l.p": "llp", "l.l.p.": "llp",
    # Services / Solutions / Systems / Associates / Group / International
    "svcs": "services", "services": "services", "svc": "services",
    "assoc": "associates", "associates": "associates", "assocs": "associates",
    "intl": "international", "international": "international", "internatl": "international",
    "grp": "group", "group": "group",
    "mfg": "manufacturing", "manufacturing": "manufacturing",
    "tech": "technology", "technology": "technology", "technologies": "technologies",
    "engg": "engineering", "engineering": "engineering",
    "mgmt": "management", "management": "management",
    "ent": "enterprises", "enterprises": "enterprises", "enterprise": "enterprise",
    "ind": "industries", "industries": "industries",
    "est": "establishment", "establishment": "establishment",
    # French common suffixes (country-agnostic: we normalize these regardless)
    "sarl": "sarl", "s.a.r.l": "sarl", "s.a.r.l.": "sarl",
    "sas": "sas", "s.a.s": "sas", "s.a.s.": "sas",
    "sa": "sa", "s.a": "sa", "s.a.": "sa",
    "eurl": "eurl",
}

# ---------------------------------------------------------------------------
# Address abbreviation canonicalization (bidirectional)
# ---------------------------------------------------------------------------
ADDRESS_ABBREV_MAP = {
    # Street types — English
    "rd": "road", "road": "road",
    "st": "street", "street": "street", "str": "street",
    "ave": "avenue", "avenue": "avenue", "av": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard", "bvd": "boulevard",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court",
    "cir": "circle", "circle": "circle",
    "pl": "place", "place": "place",
    "pkwy": "parkway", "parkway": "parkway",
    "hwy": "highway", "highway": "highway",
    "sq": "square", "square": "square",
    "trl": "trail", "trail": "trail",
    "ter": "terrace", "terrace": "terrace",
    "aly": "alley", "alley": "alley",
    "apt": "apartment", "apartment": "apartment",
    "ste": "suite", "suite": "suite",
    "fl": "floor", "floor": "floor",
    "bldg": "building", "building": "building",
    "dept": "department", "department": "department",
    # Directions
    "n": "north", "north": "north",
    "s": "south", "south": "south",
    "e": "east", "east": "east",
    "w": "west", "west": "west",
    "ne": "northeast", "northeast": "northeast",
    "nw": "northwest", "northwest": "northwest",
    "se": "southeast", "southeast": "southeast",
    "sw": "southwest", "southwest": "southwest",
    # French address types (country-agnostic)
    "rue": "rue",
    "allée": "allée", "allee": "allée",
    "chemin": "chemin", "chem": "chemin",
    "impasse": "impasse", "imp": "impasse",
    "passage": "passage",
    # Indian address types (country-agnostic)
    "nagar": "nagar", "ngr": "nagar",
    "marg": "marg", "mg": "marg",
    "gali": "gali",
    "mohalla": "mohalla",
    "colony": "colony", "col": "colony",
    "sector": "sector", "sec": "sector",
    "phase": "phase", "ph": "phase",
    "block": "block", "blk": "block",
    "plot": "plot",
    "ward": "ward",
    "dist": "district", "district": "district",
    "taluk": "taluk", "taluka": "taluk",
    "tehsil": "tehsil",
    "mandal": "mandal",
    "village": "village", "vill": "village",
}

# ---------------------------------------------------------------------------
# Landmark patterns to strip (country-agnostic)
# These reduce noise without removing useful address components
# ---------------------------------------------------------------------------
LANDMARK_PATTERNS = [
    r"\bnear\s+(?:to\s+)?[\w\s]+(?:atm|bank|hospital|station|temple|church|mosque|school|college|market|mall|hotel|tower|bridge|park|gate|chowk|circle|square|crossing|junction|flyover|metro|bus\s+stand|stop)\b",
    r"\bopp(?:osite)?\.?\s+(?:to\s+)?[\w\s]+",
    r"\bbehind\s+[\w\s]+",
    r"\bnext\s+to\s+[\w\s]+",
    r"\bbeside\s+[\w\s]+",
    r"\babove\s+[\w\s]+(?:shop|store|bank|office|restaurant)",
    r"\bbelow\s+[\w\s]+(?:shop|store|bank|office|restaurant)",
    r"\badjacent\s+to\s+[\w\s]+",
    r"\bin\s+front\s+of\s+[\w\s]+",
    r"\ben\s+face\s+de\s+[\w\s]+",  # French: "in front of"
    r"\bprès\s+de\s+[\w\s]+",  # French: "near"
]

# Compiled for performance
_LANDMARK_RE = re.compile(
    "|".join(f"({p})" for p in LANDMARK_PATTERNS),
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Core normalization functions
# ---------------------------------------------------------------------------

def _unicode_normalize(text: str) -> str:
    """NFKC normalization for consistent character representation (preserves composed accented letters)."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def _clean_basic(text: str) -> str:
    """Lowercase, standardize apostrophes/ampersands, strip punctuation (keep /, -, digits), collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    # Normalize curly apostrophes and quotes
    text = re.sub(r"[’‘`´]", "'", text)
    # Replace & with 'and'
    text = text.replace("&", " and ")
    # Remove periods from abbreviations but keep them in numbers
    text = re.sub(r"\.(?!\d)", " ", text)
    # Remove most punctuation but keep / - ' (useful in names and addresses)
    text = re.sub(r"[^\w\s/\-']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonicalize_tokens(text: str, mapping: dict) -> str:
    """Replace tokens using a mapping dict. Word-boundary-aware."""
    tokens = text.split()
    result = []
    for token in tokens:
        canonical = mapping.get(token.lower())
        if canonical:
            result.append(canonical)
        else:
            result.append(token)
    return " ".join(result)


def _strip_landmarks(text: str) -> str:
    """Remove landmark phrases that add noise to similarity computation."""
    if not text:
        return ""
    result = _LANDMARK_RE.sub(" ", text)
    return re.sub(r"\s+", " ", result).strip()


def normalize_name(name: str) -> str:
    """Normalize a business name."""
    if not name or (isinstance(name, float) and str(name) == "nan"):
        return ""
    name = str(name)
    name = _unicode_normalize(name)
    name = _clean_basic(name)
    name = _canonicalize_tokens(name, LEGAL_SUFFIX_MAP)
    return name.strip()


def normalize_address(address: str) -> str:
    """Normalize a business address."""
    if not address or (isinstance(address, float) and str(address) == "nan"):
        return ""
    address = str(address)
    address = _unicode_normalize(address)
    address = _clean_basic(address)
    address = _canonicalize_tokens(address, ADDRESS_ABBREV_MAP)
    address = _strip_landmarks(address)
    return address.strip()


def normalize_entity(name: str, address: str) -> str:
    """
    Normalize a business entity into a single text string for embedding.

    Format: "normalized_name | normalized_address"
    The pipe separator helps the model distinguish name vs address tokens
    while keeping everything in a single input sequence.

    Both sides are normalized country-agnostically.
    """
    norm_name = normalize_name(name)
    norm_address = normalize_address(address)

    if norm_name and norm_address:
        return f"{norm_name} | {norm_address}"
    elif norm_name:
        return norm_name
    elif norm_address:
        return norm_address
    else:
        return ""


def extract_postal_code(address: str) -> Optional[str]:
    """
    Extract postal/ZIP code from an address string.
    Country-agnostic: handles Indian PIN (6-digit), US ZIP (5 or 5+4),
    and French codes (5-digit).
    Prioritizes trailing occurrences to avoid misidentifying street numbers.
    """
    if not address:
        return None
    address = str(address)

    # 1. Indian PIN: exactly 6 digits (prefer last occurrence in address)
    india_pins = re.findall(r"\b([1-9][0-9]{5})\b", address)
    if india_pins:
        return india_pins[-1]

    # 2. US ZIP or French postal code: 5 digits (prefer last occurrence in address)
    five_digit_codes = re.findall(r"\b(\d{5})(?:-\d{4})?\b", address)
    if five_digit_codes:
        return five_digit_codes[-1]

    return None
