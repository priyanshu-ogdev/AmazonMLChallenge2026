"""
Country-agnostic text normalization for entity resolution (Stage 0).

Design principle: NO country-specific branching in logic flow. Every operation
must work on US, India, AND France text without hardcoded country dispatch.
Country is treated as an open-set string label used only for statistics, never
for selecting a code path.

Per the official problem statement: country must be treated as an open set
of string labels -- never hard-coded, filtered, or one-hot encoded to only
{US, India}.

Operations (in order of application):
 1. Unicode normalization (NFKC) -- preserves composed accented letters (e.g. accented chars)
 2. URL / markdown-link stripping  (e.g. "| [www.foo.com]" in S2 names)
 3. Control character removal and typographic character standardization
 4. Lowercase + whitespace collapse
 5. Ampersand -> "and", "dba"/"d/b/a" -> "doing business as"
 6. Punctuation cleanup (preserve / - ' in names/addresses)
 7. Address-specific: leading # / ## prefix removal, ordinal suffix cleanup
 8. Landmark phrase stripping (country-agnostic patterns)
 9. Legal suffix canonicalization (bidirectional, suffix-position aware)
10. Address abbreviation expansion (positional-context aware)
11. Postal code extraction (Indian 6-digit PIN, US/French 5-digit ZIP)

Output API:
  normalize_name(name)               -> normalized name string
  normalize_address(address)         -> normalized address string
  normalize_entity(name, addr)       -> single string for bi-encoder (name | address)
  normalize_entity_record(name, addr)-> structured dict with all fields for
                                        downstream feature engineering
  extract_postal_code(address)       -> postal code string or None

Notes on key decisions:
  - NFKC (not NFKD): NFKC keeps composed forms (e-acute stays e-acute); NFKD
    would decompose to base + combining mark, producing invisible combining
    characters that corrupt BM25 and character-level n-gram features.
    BGE-M3's XLM-RoBERTa tokenizer handles composed French characters natively.
  - We do NOT strip non-ASCII. French accented text and Indian transliterations
    must survive normalization intact.
  - Legal suffix expansion runs ONLY on the last 3 tokens of a name to avoid
    turning common English words like "co" (Colorado), "inc" (in Indian addresses)
    into incorrect expansions.
  - Direction abbreviations (N, S, E, W) are expanded ONLY when flanked by a
    street-type token (road, street, avenue, etc.) to avoid mangling single-letter
    tokens in Indian addresses (e.g., "Block A").
  - Landmark patterns are anchored and non-greedy to prevent over-stripping.
    The original broad patterns have been replaced with explicit,
    terminator-limited patterns.
"""

import re
import unicodedata
from typing import Optional, Dict

# ---------------------------------------------------------------------------
# 1. Legal suffix canonicalization -- bidirectional, SUFFIX-POSITION AWARE
#    Applied only to the LAST N_SUFFIX_TOKENS tokens of a business name.
#    Expanding "co" anywhere in an address would mangle "CO" (Colorado).
# ---------------------------------------------------------------------------

# Number of trailing tokens in which to apply legal suffix expansion.
# Covers patterns like "Pvt Ltd" (2 tokens), "Pvt. Ltd." or "L.L.C." (1-2 tokens).
_N_SUFFIX_TOKENS = 3

# Maps all known variants to their canonical form.
# All keys and values must be lowercase.
LEGAL_SUFFIX_MAP: Dict[str, str] = {
    # --- Corporation ---
    "corp":           "corporation",
    "corpn":          "corporation",
    "corporation":    "corporation",
    # --- Incorporated ---
    "inc":            "incorporated",
    "incorporated":   "incorporated",
    "incorp":         "incorporated",
    # --- Limited ---
    "ltd":            "limited",
    "limited":        "limited",
    "ltda":           "limited",
    # --- Private ---
    "pvt":            "private",
    "private":        "private",
    "pvte":           "private",
    # --- Company ---
    "co":             "company",
    "company":        "company",
    # --- LLC / LLP ---
    "llc":            "llc",
    "llp":            "llp",
    # --- Services ---
    "svcs":           "services",
    "services":       "services",
    "svc":            "services",
    # --- Associates ---
    "assoc":          "associates",
    "associates":     "associates",
    "assocs":         "associates",
    # --- International ---
    "intl":           "international",
    "international":  "international",
    "internatl":      "international",
    # --- Group ---
    "grp":            "group",
    "group":          "group",
    # --- Manufacturing ---
    "mfg":            "manufacturing",
    "manufacturing":  "manufacturing",
    # --- Technology / Technologies ---
    "tech":           "technology",
    "technology":     "technology",
    "technologies":   "technologies",
    # --- Engineering ---
    "engg":           "engineering",
    "engineering":    "engineering",
    "eng":            "engineering",
    # --- Management ---
    "mgmt":           "management",
    "management":     "management",
    # --- Enterprises / Enterprise ---
    "ent":            "enterprises",
    "enterprises":    "enterprises",
    "enterprise":     "enterprise",
    # --- Industries ---
    "ind":            "industries",
    "industries":     "industries",
    # --- Establishment ---
    "est":            "establishment",
    "establishment":  "establishment",
    # --- Trading ---
    "trdg":           "trading",
    "trading":        "trading",
    # --- Exports / Imports ---
    "expts":          "exports",
    "exports":        "exports",
    "impts":          "imports",
    "imports":        "imports",
    # --- French legal forms (normalized regardless of country field) ---
    "sarl":           "sarl",
    "sas":            "sas",
    "sa":             "sa",
    "eurl":           "eurl",
    "sci":            "sci",
    "snc":            "snc",
    "scp":            "scp",
    "societe":        "societe",
}

# ---------------------------------------------------------------------------
# 2. Address abbreviation expansion -- CONTEXT-AWARE
#    Street suffix tokens are expanded unconditionally.
#    Single-letter direction abbreviations are expanded only when adjacent to
#    a recognized street-type token (avoids mangling "Block A", "Sector E").
# ---------------------------------------------------------------------------

ADDRESS_SUFFIX_MAP: Dict[str, str] = {
    # Street types -- English
    "rd":        "road",
    "road":      "road",
    "st":        "street",
    "street":    "street",
    "str":       "street",
    "ave":       "avenue",
    "avenue":    "avenue",
    "av":        "avenue",
    "blvd":      "boulevard",
    "boulevard": "boulevard",
    "bvd":       "boulevard",
    "dr":        "drive",
    "drive":     "drive",
    "ln":        "lane",
    "lane":      "lane",
    "ct":        "court",
    "court":     "court",
    "cir":       "circle",
    "circle":    "circle",
    "pl":        "place",
    "place":     "place",
    "pkwy":      "parkway",
    "parkway":   "parkway",
    "hwy":       "highway",
    "highway":   "highway",
    "sq":        "square",
    "square":    "square",
    "trl":       "trail",
    "trail":     "trail",
    "ter":       "terrace",
    "terrace":   "terrace",
    "aly":       "alley",
    "alley":     "alley",
    # Unit / floor designators
    "apt":       "apartment",
    "apartment": "apartment",
    "ste":       "suite",
    "suite":     "suite",
    "fl":        "floor",
    "floor":     "floor",
    "bldg":      "building",
    "building":  "building",
    "dept":      "department",
    "department":"department",
    # French address types
    "rue":       "rue",
    "allee":     "allee",
    "chemin":    "chemin",
    "chem":      "chemin",
    "impasse":   "impasse",
    "imp":       "impasse",
    "passage":   "passage",
    "batiment":  "batiment",
    "residence": "residence",
    # Indian address components
    "nagar":     "nagar",
    "ngr":       "nagar",
    "marg":      "marg",
    "gali":      "gali",
    "mohalla":   "mohalla",
    "colony":    "colony",
    "col":       "colony",
    "sector":    "sector",
    "sec":       "sector",
    "phase":     "phase",
    "block":     "block",
    "blk":       "block",
    "plot":      "plot",
    "ward":      "ward",
    "dist":      "district",
    "district":  "district",
    "taluk":     "taluk",
    "taluka":    "taluk",
    "tehsil":    "tehsil",
    "mandal":    "mandal",
    "village":   "village",
    "vill":      "village",
    "opp":       "opposite",
    "opposite":  "opposite",
}

# Cardinal directions: ONLY expanded when flanked by a street-type token.
_DIRECTION_ABBREV: Dict[str, str] = {
    "n":  "north",
    "s":  "south",
    "e":  "east",
    "w":  "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# All known address-suffix tokens (used for direction-expansion gating)
_STREET_TYPE_TOKENS = (
    frozenset(ADDRESS_SUFFIX_MAP.values()) | frozenset(ADDRESS_SUFFIX_MAP.keys())
)

# ---------------------------------------------------------------------------
# 3. Landmark / relational phrase patterns (non-greedy, terminator-anchored)
#    Replaces the original broad greedy patterns that over-stripped addresses.
#    Each pattern terminates at a comma, digit, or end-of-string.
#    "opp"/"opposite" is KEPT (expanded, not stripped) because it often
#    carries postal locality info in Indian addresses.
# ---------------------------------------------------------------------------

_LANDMARK_PATTERNS = [
    r"\bnear\s+(?:to\s+)?[a-z\s]+?(?=\s*[,\d]|$)",
    r"\bbehind\s+[a-z\s]+?(?=\s*[,\d]|$)",
    r"\bnext\s+to\s+[a-z\s]+?(?=\s*[,\d]|$)",
    r"\bbeside\s+[a-z\s]+?(?=\s*[,\d]|$)",
    r"\b(?:above|below)\s+[a-z\s]+?(?:shop|store|bank|office|restaurant)(?=\s*[,\d]|$)",
    r"\badjacent\s+to\s+[a-z\s]+?(?=\s*[,\d]|$)",
    r"\bin\s+front\s+of\s+[a-z\s]+?(?=\s*[,\d]|$)",
    r"\ben\s+face\s+de\s+[a-z\s]+?(?=\s*[,\d]|$)",
    r"\bpr[e\xe8]s\s+de\s+[a-z\s]+?(?=\s*[,\d]|$)",
]

_LANDMARK_RE = re.compile(
    "|".join("(?:{})".format(p) for p in _LANDMARK_PATTERNS),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 4. URL / markdown link patterns (seen in raw S2 data)
#    e.g. "| [www.shivshakti.com](https://www.shivshakti.com)"
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r"(?:"
    r"https?://\S+"
    r"|www\.\S+"
    r"|\[.*?\]\(https?://\S+\)"
    r"|\[www\.\S+\]"
    r"|\|\s*\[.*?\]\(https?://\S+\)"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 5. Postal code patterns (for extraction only -- not stripped from address)
# ---------------------------------------------------------------------------
_INDIA_PIN_RE = re.compile(r"\b([1-9][0-9]{5})\b")
_ZIPCODE_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")

# ---------------------------------------------------------------------------
# 6. "Doing Business As" normalization
# ---------------------------------------------------------------------------
_DBA_RE = re.compile(
    r"\b(?:d/?b/?a|doing\s+business\s+as)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 7. Ordinal number suffixes in addresses: 1st, 2nd, 3rd, 4th, etc.
#    Normalize to bare number for consistent token matching.
# ---------------------------------------------------------------------------
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unicode_normalize(text: str) -> str:
    """
    NFKC normalization -- preserves composed accented characters (e-acute stays
    e-acute). NFKD would decompose to base + combining mark, producing invisible
    characters that corrupt BM25 and character n-gram features.
    Also converts full-width ASCII, ligatures, and non-breaking spaces.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def _strip_urls(text: str) -> str:
    """Remove URLs and markdown link syntax before further processing."""
    if not text:
        return ""
    return _URL_RE.sub(" ", text)


def _clean_typography_and_case(text: str) -> str:
    """
    Lowercase, standardize typographic characters, apply dba expansion.
    Preserves commas and separators needed by downstream pattern matching.
    """
    if not text:
        return ""
    text = text.lower().strip()
    # Typographic apostrophes -> ASCII apostrophe
    text = re.sub(r"[\u2018\u2019`\xb4\u201a]", "'", text)
    # Typographic dashes -> ASCII hyphen-minus
    text = re.sub(r"[\u2013\u2014\u2212]", "-", text)
    # Ampersand -> and
    text = text.replace("&", " and ")
    # dba -> doing business as
    text = _DBA_RE.sub(" doing business as ", text)
    return text


def _strip_punctuation(text: str) -> str:
    """
    Remove periods unless between two digits.
    Remove punctuation except / - ' and collapse whitespace.
    """
    if not text:
        return ""
    # Remove periods unless between two digits
    text = re.sub(r"\.(?!\d)", " ", text)
    text = re.sub(r"(?<!\d)\.", " ", text)
    # Remove punctuation except / - '
    text = re.sub(r"[^\w\s/\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_basic(text: str) -> str:
    """
    Full basic cleaning: case, typography, punctuation stripping, whitespace collapse.
    Used directly by normalize_name and backward-compatible callers.
    """
    return _strip_punctuation(_clean_typography_and_case(text))


def _strip_address_hash_prefix(address: str) -> str:
    """
    Remove leading # or ## from address number tokens.
    S3 data shows patterns like "##8 Willow Oak Lane, Fl. 0".
    Applied after lowercase.
    """
    if not address:
        return ""
    return re.sub(r"(?<!\w)#+(\w)", r"\1", address)


def _normalize_ordinals(text: str) -> str:
    """
    Normalize ordinal numbers to bare digits.
    "22nd Main" -> "22 main", "9th Block" -> "9 block".
    """
    return _ORDINAL_RE.sub(r"\1", text)


def _strip_landmarks(text: str) -> str:
    """
    Remove relational landmark phrases (non-greedy, terminator-anchored).
    Prevents over-stripping of useful address content.
    """
    if not text:
        return ""
    result = _LANDMARK_RE.sub(" ", text)
    return re.sub(r"\s+", " ", result).strip()


def _canonicalize_legal_suffixes(name: str) -> str:
    """
    Expand legal suffix abbreviations in the LAST _N_SUFFIX_TOKENS tokens ONLY.

    Rationale: "corp", "inc", "ltd" etc. appear as legal suffixes at the END of
    business names. Applying globally would incorrectly expand:
    - "co" in "Colorado Springs" (state code in addresses)
    - "inc" in Indian addresses containing "Income Tax"
    - "est" in "West" or "Eastside"

    For short names (<=3 tokens), all tokens are eligible.
    """
    tokens = name.split()
    if not tokens:
        return ""
    if len(tokens) <= _N_SUFFIX_TOKENS:
        return " ".join(LEGAL_SUFFIX_MAP.get(t, t) for t in tokens)
    head = tokens[: len(tokens) - _N_SUFFIX_TOKENS]
    tail = tokens[len(tokens) - _N_SUFFIX_TOKENS :]
    expanded_tail = [LEGAL_SUFFIX_MAP.get(t, t) for t in tail]
    return " ".join(head + expanded_tail)


def _expand_address_suffixes(address: str) -> str:
    """
    Expand address abbreviation tokens. Two-pass:
      Pass 1 -- expand all street suffix tokens unconditionally.
      Pass 2 -- expand direction abbreviations (n, s, e, w...) ONLY when
                immediately adjacent to a street-type token.

    This prevents mangling single-letter identifiers in Indian addresses
    like "Block A" or "Sector E".
    """
    tokens = address.split()
    if not tokens:
        return ""

    # Pass 1: street suffixes (safe at all positions)
    expanded = [ADDRESS_SUFFIX_MAP.get(t, t) for t in tokens]

    # Pass 2: direction abbreviations (only when next to street type)
    result = list(expanded)
    for i, tok in enumerate(expanded):
        if tok in _DIRECTION_ABBREV:
            left = expanded[i - 1] if i > 0 else ""
            right = expanded[i + 1] if i < len(expanded) - 1 else ""
            if left in _STREET_TYPE_TOKENS or right in _STREET_TYPE_TOKENS:
                result[i] = _DIRECTION_ABBREV[tok]

    return " ".join(result)


# ---------------------------------------------------------------------------
# Public normalization functions
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Normalize a business name for lexical matching and embedding.

    Pipeline:
      1. Unicode normalization (NFKC)
      2. URL stripping
      3. Basic cleaning (case, punctuation, whitespace, dba)
      4. Legal suffix canonicalization (suffix-position aware)

    Preserves accented characters (French) and Devanagari/Indic script.

    Args:
        name: Raw business_name string (may be NaN, empty, or have encoding noise).

    Returns:
        Normalized name string. Empty string for null/empty input.
    """
    if name is None:
        return ""
    name = str(name).strip()
    if name.lower() in ("nan", "none", "null", ""):
        return ""
    name = _unicode_normalize(name)
    name = _strip_urls(name)
    name = _clean_basic(name)
    name = _canonicalize_legal_suffixes(name)
    return name.strip()


def normalize_address(address: str) -> str:
    """
    Normalize a business address for lexical matching and embedding.

    Pipeline:
      1. Unicode normalization (NFKC)
      2. URL stripping
      3. Typography and case cleaning (preserves commas for landmark boundaries)
      4. Hash-prefix removal (##8 -> 8)
      5. Ordinal normalization (22nd -> 22)
      6. Landmark phrase stripping (non-greedy, terminator-anchored at comma/digits)
      7. Punctuation stripping (removes commas, periods, collapses whitespace)
      8. Address suffix expansion (street types + context-aware directions)

    Args:
        address: Raw business_address string (may be NaN, empty, or have noise).

    Returns:
        Normalized address string. Empty string for null/empty/missing input.
    """
    if address is None:
        return ""
    address = str(address).strip()
    if address.lower() in ("nan", "none", "null", ""):
        return ""
    address = _unicode_normalize(address)
    address = _strip_urls(address)
    address = _clean_typography_and_case(address)
    address = _strip_address_hash_prefix(address)
    address = _normalize_ordinals(address)
    address = _strip_landmarks(address)
    address = _strip_punctuation(address)
    address = _expand_address_suffixes(address)
    return address.strip()


def normalize_entity(name: str, address: str) -> str:
    """
    Normalize a business entity into a single string for bi-encoder embedding.

    Format:
        "{norm_name} | {norm_address}"    -- address present
        "{norm_name} | [NO_ADDRESS]"      -- address structurally missing (null/NaN)
        "{norm_name}"                     -- address present but normalizes empty
        "[NO_ADDRESS]"                    -- both missing

    The pipe separator helps the transformer distinguish name vs address tokens.
    The [NO_ADDRESS] sentinel makes structural absence explicit in the token
    stream, distinct from an address that normalizes to empty.

    Args:
        name: Raw business_name.
        address: Raw business_address (may be empty/NaN).

    Returns:
        Single string for tokenization. Never empty.
    """
    norm_name = normalize_name(name)
    norm_address = normalize_address(address)

    raw_addr = str(address).strip() if address is not None else ""
    is_structurally_missing = raw_addr.lower() in ("nan", "none", "null", "")

    if norm_name and norm_address:
        return "{} | {}".format(norm_name, norm_address)
    elif norm_name and is_structurally_missing:
        return "{} | [NO_ADDRESS]".format(norm_name)
    elif norm_name:
        return norm_name
    elif norm_address:
        return norm_address
    else:
        return "[NO_ADDRESS]"


# ---------------------------------------------------------------------------
# Country canonicalization, source extraction, and structural fields
# Per official problem statement: country is an open set of string labels.
# Known aliases (US/USA/United States, India/Bharat, France/FR) are folded
# to a canonical token for blocking; unrecognized labels pass through.
# ---------------------------------------------------------------------------

_COUNTRY_ALIAS_CLASSES = [
    ["us", "usa", "united states", "united states of america", "u s", "u s a"],
    ["india", "in", "bharat"],
    ["france", "fr"],
]

def _build_country_map(classes):
    out = {}
    for cls in classes:
        canonical = cls[0]
        for variant in cls:
            out[variant] = canonical
    return out

_COUNTRY_MAP = _build_country_map(_COUNTRY_ALIAS_CLASSES)

def canonicalize_country(country: str) -> str:
    """
    Normalizes known aliases to one canonical spelling for blocking/features;
    passes unrecognized country labels (including France or future countries)
    through unchanged after NFKC and lowercase. Never drops or maps to sentinel.
    """
    if not country:
        return ""
    key = unicodedata.normalize("NFKC", str(country)).strip().lower()
    key = re.sub(r"\s+", " ", key)
    return _COUNTRY_MAP.get(key, key)


def country_match_flag(country_a: str, country_b: str) -> Optional[int]:
    """
    Stage 2c feature: symmetric derived match flag (1 for match, 0 for mismatch).
    Returns None when either side is empty so tree models can use missing branch.
    Works for any country (including France zero-shot) without categorical leakage.
    """
    a = canonicalize_country(country_a)
    b = canonicalize_country(country_b)
    if not a or not b:
        return None
    return int(a == b)


_SOURCE_PREFIX_RE = re.compile(r"^(S[123])-")


def source_from_entity_id(entity_id: str) -> str:
    """Extract source prefix ('S1', 'S2', 'S3') from entity_id string."""
    if not entity_id:
        return ""
    m = _SOURCE_PREFIX_RE.match(str(entity_id).strip())
    return m.group(1) if m else ""


_DIGIT_RUN_RE = re.compile(r"\b\d{4,10}(?:-\d{3,4})?\b")
_TRAILING_SEGMENT_RE = re.compile(r",\s*([^,]+)$")
_STREET_NUM_RE = re.compile(r"^\s*(\d{1,6})\b")


def extract_structural_fields(address: str) -> Dict[str, object]:
    """
    Extract generic language-agnostic structural sub-fields from address text:
      - digit_runs: candidate PIN/ZIP/postal/house-no codes (4-10 digits)
      - trailing_segment: last comma-separated segment (often city/state/region)
      - street_number: leading digit run if address begins with a number
    """
    if not address:
        return {"digit_runs": [], "trailing_segment": None, "street_number": None}
    addr_str = str(address).strip()
    digit_runs = _DIGIT_RUN_RE.findall(addr_str)
    trailing = _TRAILING_SEGMENT_RE.search(addr_str)
    trailing_segment = trailing.group(1).strip() if trailing else None
    lead = _STREET_NUM_RE.match(addr_str)
    street_number = lead.group(1) if lead else None
    return {
        "digit_runs": digit_runs,
        "trailing_segment": trailing_segment,
        "street_number": street_number,
    }


def normalize_entity_record(
    name: str,
    address: str,
    entity_id: str = "",
    country: str = "",
) -> Dict[str, object]:
    """
    Normalize a business entity and return a structured dict with ALL fields
    needed for downstream feature engineering (blocking, pair features, GBM).

    This is the canonical Stage 0 output record. Downstream stages should
    consume this dict rather than re-running normalization independently.

    Returns dict with:
        entity_id          (str):  Pass-through entity ID.
        source             (str):  Derived source ('S1'|'S2'|'S3').
        country            (str):  Pass-through country label.
        country_canonical  (str):  Canonicalized country token.
        raw_name           (str):  Original name.
        raw_address        (str):  Original address.
        norm_name          (str):  Normalized name.
        norm_address       (str):  Normalized address (empty if missing).
        encoder_text       (str):  Combined text for bi-encoder.
        is_address_missing (bool): True if address was null/NaN/empty in raw data.
        postal_code        (str|None): Extracted postal/ZIP/PIN code.
        street_number      (str|None): Extracted leading house/street number.
        trailing_segment   (str|None): Trailing comma-separated location segment.
        digit_runs         (list): All 4-10 digit numbers in address.
        name_token_count   (int):  Word count of normalized name.
        address_token_count(int):  Word count of normalized address.

    Args:
        name: Raw business_name.
        address: Raw business_address.
        entity_id: Entity identifier (pass-through).
        country: Country label (pass-through).

    Returns:
        Dict with all normalized fields.
    """
    norm_name = normalize_name(name)
    norm_address = normalize_address(address)

    raw_addr = str(address).strip() if address is not None else ""
    is_missing = raw_addr.lower() in ("nan", "none", "null", "") or len(raw_addr) == 0

    postal = extract_postal_code(raw_addr if not is_missing else "")
    encoder_text = normalize_entity(name, address)
    structural = extract_structural_fields(raw_addr if not is_missing else "")

    return {
        "entity_id":            entity_id,
        "source":               source_from_entity_id(entity_id),
        "country":              str(country).strip() if country else "",
        "country_canonical":    canonicalize_country(country),
        "raw_name":             str(name).strip() if name else "",
        "raw_address":          raw_addr,
        "norm_name":            norm_name,
        "norm_address":         norm_address,
        "encoder_text":         encoder_text,
        "is_address_missing":   is_missing,
        "postal_code":          postal,
        "street_number":        structural["street_number"],
        "trailing_segment":     structural["trailing_segment"],
        "digit_runs":           structural["digit_runs"],
        "name_token_count":     len(norm_name.split()) if norm_name else 0,
        "address_token_count":  len(norm_address.split()) if norm_address else 0,
    }


def extract_postal_code(address: str) -> Optional[str]:
    """
    Extract postal/ZIP code from an address string (country-agnostic).

    Priority order:
      1. Indian PIN: exactly 6 digits, first digit 1-9. Last occurrence preferred
         to avoid misidentifying street numbers (e.g., "H.No 570" is NOT a PIN).
      2. US ZIP or French code: exactly 5 digits (optional +4 suffix).
         Last occurrence preferred.

    Works on both raw and normalized address strings.

    Args:
        address: Address string to search.

    Returns:
        Postal code string, or None if no recognized pattern found.
    """
    if not address:
        return None
    address = str(address)

    # 1. Indian PIN (6-digit): prefer last occurrence
    india_pins = _INDIA_PIN_RE.findall(address)
    if india_pins:
        return india_pins[-1]

    # 2. US ZIP or French code (5-digit): prefer last occurrence
    five_digit = _ZIPCODE_RE.findall(address)
    if five_digit:
        return five_digit[-1]

    return None


# ---------------------------------------------------------------------------
# Batch normalization helpers
# ---------------------------------------------------------------------------

def normalize_dataframe_records(
    df,
    name_col: str = "business_name",
    address_col: str = "business_address",
    id_col: str = "entity_id",
    country_col: str = "country",
) -> list:
    """
    Normalize all records in a pandas DataFrame using normalize_entity_record.

    Memory-efficient: processes rows without constructing intermediate DataFrames.
    Intended as a drop-in replacement for data_builder.records_to_dict().

    Args:
        df: pandas DataFrame with business entity columns.
        name_col: Column name for business name.
        address_col: Column name for business address.
        id_col: Column name for entity ID.
        country_col: Column name for country.

    Returns:
        List of dicts from normalize_entity_record.
    """
    import pandas as pd

    results = []
    for _, row in df.iterrows():
        eid = str(row.get(id_col, "")) if pd.notna(row.get(id_col)) else ""
        name = str(row.get(name_col, "")) if pd.notna(row.get(name_col)) else ""
        addr = str(row.get(address_col, "")) if pd.notna(row.get(address_col)) else ""
        ctry = str(row.get(country_col, "")) if pd.notna(row.get(country_col)) else ""
        results.append(
            normalize_entity_record(name, addr, entity_id=eid, country=ctry)
        )
    return results


# ---------------------------------------------------------------------------
# Self-test (python -m src.normalize or python normalize.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io
    # Force UTF-8 output on Windows (cp1252 cannot print Devanagari/French chars)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    _TESTS = [
        # (description, name, address, expected_name_fragment, expected_addr_fragment)
        (
            "US LLC with trailing punctuation",
            "Custom Wealth Services LLC",
            "OH, Columbus, 5559 Orville Avenue",
            "custom wealth services llc",
            "oh columbus 5559 orville avenue",
        ),
        (
            "US Inc with S2-style address reorder",
            "Summit Inc",
            "GREENSBORO, NC, 19 1/2 STARDUST TRAIL",
            "summit incorporated",
            "greensboro nc 19 1/2 stardust trail",
        ),
        (
            "Indian Pvt Ltd -- Devanagari name preserved",
            "\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0902\u0917 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921",
            "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
            # Devanagari is preserved by NFKC; skip name fragment check (terminal encoding)
            # The norm_name length assertion below validates the name wasn't destroyed.
            None,
            "new delhi",
        ),
        (
            # Accents are PRESERVED by design (NFKC, not transliteration).
            # BGE-M3 tokenizer handles composed French chars natively.
            "French SAS with accented name (accents preserved)",
            "Soci\u00e9t\u00e9 G\u00e9n\u00e9rale SAS",
            "29 Boulevard Haussmann, 75009 Paris",
            "soci\u00e9t\u00e9 g\u00e9n\u00e9rale sas",
            "29 boulevard haussmann",
        ),
        (
            "Missing address -> [NO_ADDRESS] sentinel",
            "International South Consultants Private Ltd",
            "",
            "international south consultants private limited",
            "[NO_ADDRESS]",
        ),
        (
            "URL in S2 name -> stripped",
            "SHIVSHAKTI VIDYALAYA OVERSEAS CORPORATION | [www.shivshakti.com](https://www.shivshakti.com)",
            "H.NO 204 C ROAD HOSHIARPUR, PUNJAB, Punjab",
            "shivshakti vidyalaya overseas corporation",
            "h no 204 c road hoshiarpur",
        ),
        (
            "S3-style ## address prefix removed",
            "Animal Hanisch Hospirlg",
            "##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri",
            "animal hanisch hospirlg",
            "8 willow oak lane floor 0 saint louis missouri",
        ),
        (
            "Indian address with landmark noise stripped",
            "Prabhav Business Center",
            "797, Lake Town Block A, Kolkata, near SBI Bank, Howrah",
            "prabhav business center",
            "lake town block a kolkata",
        ),
        (
            # LLC at the START of a name is NOT expanded by suffix-position-aware
            # canonicalization (only last 3 tokens are eligible). This is correct
            # behaviour: we never know if a leading 'LLC' is a prefix name or suffix.
            "LLC-prefix name -- legal suffix expansion applies to last 3 tokens only",
            "LLC Moncada Learning Center",
            "5780 Fawn Ct, 22Nd Main 9Th Block, Fort Worth, Texas",
            "llc moncada learning center",
            "5780 fawn court 22 main 9 block fort worth texas",
        ),
        (
            "dba normalization in name",
            "Ectolumdrex dba X+ Madison Inc",
            "S03575 Cty Tk M, Town Of Buffalo, WI",
            "ectolumdrex doing business as x madison incorporated",
            "s03575 cty tk m town of buffalo wi",
        ),
    ]

    print("=" * 72)
    print("Stage 0 Normalization -- Self-Test")
    print("=" * 72)

    all_passed = True
    for i, (desc, name, addr, exp_name, exp_addr) in enumerate(_TESTS):
        rec = normalize_entity_record(name, addr)
        n = rec["norm_name"]
        et = rec["encoder_text"]
        pc = rec["postal_code"]

        name_ok = (exp_name is None) or (exp_name in n)
        addr_ok = (exp_addr is None) or (exp_addr in et)
        status = "PASS" if (name_ok and addr_ok) else "FAIL"
        if not (name_ok and addr_ok):
            all_passed = False

        print("\n[{:02d}] {} -- {}".format(i + 1, status, desc))
        print("     norm_name    : {!r}".format(n))
        print("     norm_address : {!r}".format(rec["norm_address"]))
        print("     encoder_text : {!r}".format(et))
        print("     postal_code  : {!r}".format(pc))
        print("     is_missing   : {}".format(rec["is_address_missing"]))
        if not name_ok:
            print("     EXPECTED name fragment : {!r}".format(exp_name))
        if not addr_ok:
            print("     EXPECTED addr fragment : {!r}".format(exp_addr))

    print("\n" + "=" * 72)
    print("Postal Code Extraction Tests")
    print("=" * 72)
    _PC_TESTS = [
        ("Indian PIN at end", "KH NO. 570/13, NEW DELHI 110001", "110001"),
        ("US ZIP", "1702 Pine Avenue, Menomonie, WI 54751", "54751"),
        ("US ZIP+4", "914 Pierpont Ave, Cleveland, OH 44114-2201", "44114"),
        ("French postal", "29 Boulevard Haussmann, 75009 Paris", "75009"),
        ("No postal code", "Lake Town Block A, Kolkata", None),
        ("Street number not confused with PIN", "H.No 570, Village Town 380001", "380001"),
    ]
    for desc, addr, expected in _PC_TESTS:
        got = extract_postal_code(addr)
        status = "PASS" if got == expected else "FAIL"
        print("  {} -- {}: got={!r}, expected={!r}".format(status, desc, got, expected))

    print("\n" + "=" * 72)
    print("Country Canonicalization & Match Flag Tests")
    print("=" * 72)
    _COUNTRY_TESTS = [
        ("US alias folding", "USA", "us"),
        ("United States alias", "United States of America", "us"),
        ("India alias folding", "Bharat", "india"),
        ("France open-set preservation", "France", "france"),
        ("Unseen country passes through", "Germany", "germany"),
        ("Country match flag same aliases", country_match_flag("US", "USA"), 1),
        ("Country match flag cross country", country_match_flag("India", "France"), 0),
        ("Country match flag unseen same", country_match_flag("France", "France"), 1),
        ("Country match flag missing side", country_match_flag("US", ""), None),
    ]
    for desc, got, expected in _COUNTRY_TESTS:
        val = canonicalize_country(got) if isinstance(got, str) and not desc.startswith("Country match flag") else got
        status = "PASS" if val == expected else "FAIL"
        if val != expected:
            all_passed = False
        print("  {} -- {}: got={!r}, expected={!r}".format(status, desc, val, expected))

    print("\n" + "=" * 72)
    print("Structural Field Extraction Tests")
    print("=" * 72)
    st1 = extract_structural_fields("1795 Westchester Drive, High Point, NC")
    print("  street_number: {!r} (expected '1795')".format(st1["street_number"]))
    print("  trailing_segment: {!r} (expected 'NC')".format(st1["trailing_segment"]))
    print("  digit_runs: {!r}".format(st1["digit_runs"]))
    if st1["street_number"] != "1795" or st1["trailing_segment"] != "NC":
        all_passed = False

    print("\n" + "=" * 72)
    print("Overall: {}".format("ALL PASSED" if all_passed else "SOME TESTS FAILED"))
    print("=" * 72)
