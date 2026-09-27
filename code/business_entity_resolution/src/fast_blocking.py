"""
High-Performance Normalized Blocking Engine (FastNormalizedBlocker)
===================================================================
Optimized for pre-normalized Stage 0 records:
1. Zero-copy / fast columnar ingestion via Polars.
2. Integer dictionary encoding (uint32) for all candidate IDs.
3. Country-partitioned inverted index:
   country -> channel -> key -> posting list (uint32).
4. Full MultiChannel parity:
   - Channel 1: Exact Name, First-2-Tokens, Acronym & Composite Structural Keys
   - Channel 2: Character 3-Gram & 4-Gram Sub-Linear TF-IDF Retrieval
   - Channel 3: Token Inverted Index with Sub-Linear TF-IDF
   - Channel 4: Address Structural Key (postal + street)
   - Channel 4b: Address Inverted Index (Tokens with Sub-Linear TF-IDF & Multilingual Stopwords)
5. Strict Exact-Priority Sorting:
   is_exact takes strict priority (-1 vs 0), followed by blocker consensus count (-b_cnt),
   best score, best rank, and deterministic candidate ID.
6. Zero heap churn and fast execution.
"""
import time
import math
import heapq
import csv
import re
import io
import json
import operator
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set, Sequence, Iterable
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Configuration matching docs/04_stage1_blocking.md and blocking.py v5
# ---------------------------------------------------------------------------
TOP_K_SPARSE = 50
TOP_K_DENSE = 50
MAX_CANDIDATES = 25
SIMILARITY_FLOOR = 0.30
MAX_TOKEN_DOC_FREQ = 0.02
MAX_TOKEN_DOC_COUNT = 2500
MAX_NGRAM_DOC_FREQ = 0.10
MAX_NGRAM_DOC_COUNT = 5000
MAX_ADDR_TOKEN_DOC_COUNT = 5000
MIN_TOKEN_LEN = 3

EXACT_BLOCKERS = frozenset({
    "exact_name", "first_2_tokens", "acronym_match",
    "exact_name_postal", "exact_name_street", "exact_name_trailing",
    "name_lead_postal", "name_lead_street_trailing",
})

_WORD_RE = re.compile(r"[a-z0-9]+")

_ADDRESS_STOPWORDS = {
    "road", "street", "st", "rd", "ave", "avenue", "lane", "ln", "dr", "drive",
    "blvd", "boulevard", "floor", "fl", "unit", "suite", "ste", "apt", "apartment",
    "no", "near", "opp", "opposite", "behind", "court", "ct", "way", "parkway",
    "pkwy", "city", "state", "county", "block", "house", "building", "bldg",
    "first", "second", "third", "main", "cross", "north", "south",
    "east", "west", "c/o", "co", "india", "usa", "us",
    # French street/locality generics
    "rue", "bd", "route", "chemin", "allee", "place", "impasse", "quai", "cours",
    "passage", "square", "france", "paris",
    # Common Indian locality generics
    "nagar", "marg", "colony", "sector", "plot", "bengal", "delhi", "mumbai",
}

_itemgetter_0 = operator.itemgetter(0)
_itemgetter_1 = operator.itemgetter(1)


def _tokenize(text: str) -> List[str]:
    """Extract lowercased word tokens with minimum length."""
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= MIN_TOKEN_LEN]


def get_char_ngrams(s: str) -> List[str]:
    """Fast character 3-gram and 4-gram extraction."""
    L = len(s)
    if L < 3:
        return []
    res = []
    # 3-grams
    for i in range(L - 2):
        res.append(s[i:i+3])
    # 4-grams
    if L >= 4:
        for i in range(L - 3):
            res.append(s[i:i+4])
    return res


class FastNormalizedBlocker:
    """
    High-performance multi-channel blocker operating directly on Stage 0 normalized tables.
    """

    def __init__(
        self,
        top_k_sparse: int = TOP_K_SPARSE,
        max_candidates: int = MAX_CANDIDATES,
        similarity_floor: float = SIMILARITY_FLOOR,
        max_ngram_doc_count: int = MAX_NGRAM_DOC_COUNT,
        max_token_doc_count: int = MAX_TOKEN_DOC_COUNT,
        max_addr_token_doc_count: int = MAX_ADDR_TOKEN_DOC_COUNT,
    ):
        self.top_k_sparse = top_k_sparse
        self.max_candidates = max_candidates
        self.similarity_floor = similarity_floor
        self.max_ngram_doc_count = max_ngram_doc_count
        self.max_token_doc_count = max_token_doc_count
        self.max_addr_token_doc_count = max_addr_token_doc_count

        # Integer dictionary encoding: int_id <-> str_id
        self.id_to_str: List[str] = []
        self.id_to_source: List[str] = []

        # Partitions by canonical country: country -> channel_index
        self.countries: Set[str] = set()
        self.c_exact_name: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_first_2_tokens: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_acronym: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_name_postal: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_name_street: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_name_trailing: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_lead_postal: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_lead_street_trail: Dict[str, Dict[Tuple[str, str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_addr_struct: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))

        self.c_ngrams: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_tokens: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.c_addr_tokens: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

        self.c_ngram_doc_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.c_token_doc_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.c_addr_token_doc_counts: Dict[str, Counter[str]] = defaultdict(Counter)

        # Precomputed IDF tables: country -> term -> float weight
        self.c_ngram_idf: Dict[str, Dict[str, float]] = {}
        self.c_token_idf: Dict[str, Dict[str, float]] = {}
        self.c_addr_token_idf: Dict[str, Dict[str, float]] = {}

        # Candidate lengths for cosine normalization: int_id -> length
        self.cand_ngram_lens: List[int] = []
        self.cand_token_lens: List[int] = []
        self.cand_addr_lens: List[int] = []

    def index_normalized_file(self, file_path: Path | str) -> int:
        """
        Fast ingestion and indexing using Polars zero-copy columnar reader.
        """
        file_path = Path(file_path)
        df = pl.read_csv(
            file_path,
            separator="\t",
            columns=[
                "entity_id", "country_canonical", "norm_name", "norm_address",
                "postal_code", "street_number", "trailing_segment", "is_address_missing"
            ],
            schema_overrides={
                "entity_id": pl.String,
                "country_canonical": pl.String,
                "norm_name": pl.String,
                "norm_address": pl.String,
                "postal_code": pl.String,
                "street_number": pl.String,
                "trailing_segment": pl.String,
                "is_address_missing": pl.Int32,
            }
        )

        n_rows = df.height
        eids = df["entity_id"].to_list()
        countries = df["country_canonical"].fill_null("").to_list()
        norm_names = df["norm_name"].fill_null("").to_list()
        norm_addrs = df["norm_address"].fill_null("").to_list()
        postals = df["postal_code"].fill_null("").to_list()
        streets = df["street_number"].fill_null("").to_list()
        trailin = df["trailing_segment"].fill_null("").to_list()
        miss_addrs = df["is_address_missing"].fill_null(0).to_list()

        start_id = len(self.id_to_str)
        self.id_to_str.extend(eids)
        self.id_to_source.extend([eid[:2] if len(eid) >= 2 else "" for eid in eids])

        max_ngram_cnt = self.max_ngram_doc_count
        max_tok_cnt = self.max_token_doc_count
        max_addr_tok_cnt = self.max_addr_token_doc_count

        for i in range(n_rows):
            cid = start_id + i
            c_country = countries[i]
            self.countries.add(c_country)
            n_name = norm_names[i]
            n_addr = norm_addrs[i]
            postal = postals[i]
            street = streets[i]
            trail = trailin[i]
            is_miss = bool(miss_addrs[i])

            # Name Tokens
            tokens = n_name.split()
            tok_len = max(1, len(tokens))
            self.cand_token_lens.append(tok_len)

            # Name Ngrams
            ngrams = get_char_ngrams(n_name)
            ng_len = max(1, len(ngrams))
            self.cand_ngram_lens.append(ng_len)

            # Address Tokens (Channel 4b)
            if n_addr and not is_miss:
                addr_toks = [t for t in _tokenize(n_addr) if len(t) >= 3 and t not in _ADDRESS_STOPWORDS]
            else:
                addr_toks = []
            addr_len = max(1, len(addr_toks))
            self.cand_addr_lens.append(addr_len)

            first_word = tokens[0] if tokens else ""

            # 1. Exact Name
            if n_name:
                self.c_exact_name[c_country][n_name].append(cid)

            # 2. First-2-Tokens & Acronym
            if len(tokens) >= 2:
                f2 = f"{tokens[0]} {tokens[1]}"
                self.c_first_2_tokens[c_country][f2].append(cid)
                if len(tokens[0]) > 0 and len(tokens[1]) > 0:
                    acr = f"{tokens[0][0]}{tokens[1][0]}"
                    self.c_acronym[c_country][acr].append(cid)

            # 3. Composite Structural Keys
            if n_name and postal:
                self.c_name_postal[c_country][(n_name, postal)].append(cid)
            if n_name and street:
                self.c_name_street[c_country][(n_name, street)].append(cid)
            if n_name and trail:
                self.c_name_trailing[c_country][(n_name, trail)].append(cid)

            if first_word and postal and len(first_word) >= 3:
                self.c_lead_postal[c_country][(first_word, postal)].append(cid)
            if first_word and street and trail and len(first_word) >= 3:
                self.c_lead_street_trail[c_country][(first_word, street, trail)].append(cid)

            # 4. Address Structural Key
            if not is_miss and postal and street:
                self.c_addr_struct[c_country][(postal, street)].append(cid)

            # 5. Character N-Grams
            ng_counts = Counter(ngrams)
            idx_ng = self.c_ngrams[c_country]
            cnt_ng = self.c_ngram_doc_counts[c_country]
            for g in ng_counts:
                c = cnt_ng[g] + 1
                cnt_ng[g] = c
                if c <= max_ngram_cnt:
                    idx_ng[g].append(cid)
                elif c == max_ngram_cnt + 1:
                    idx_ng.pop(g, None)

            # 6. Word Tokens
            tok_set = set(tokens)
            idx_tok = self.c_tokens[c_country]
            cnt_tok = self.c_token_doc_counts[c_country]
            for t in tok_set:
                if len(t) < 3:
                    continue
                c = cnt_tok[t] + 1
                cnt_tok[t] = c
                if c <= max_tok_cnt:
                    idx_tok[t].append(cid)
                elif c == max_tok_cnt + 1:
                    idx_tok.pop(t, None)

            # 7. Address Tokens (Channel 4b)
            if addr_toks:
                addr_tok_set = set(addr_toks)
                idx_addr = self.c_addr_tokens[c_country]
                cnt_addr = self.c_addr_token_doc_counts[c_country]
                for at in addr_tok_set:
                    c = cnt_addr[at] + 1
                    cnt_addr[at] = c
                    if c <= max_addr_tok_cnt:
                        idx_addr[at].append(cid)
                    elif c == max_addr_tok_cnt + 1:
                        idx_addr.pop(at, None)

        return n_rows

    def build_idf_tables(self) -> None:
        """Precompute partition-specific IDF tables."""
        _log = math.log
        N_total = len(self.id_to_str)
        for country in self.countries:
            idx_ng = self.c_ngrams[country]
            cnt_ng = self.c_ngram_doc_counts[country]
            ng_idf = {}
            for g, doc_cnt in cnt_ng.items():
                if g not in idx_ng or doc_cnt == 0:
                    continue
                ng_idf[g] = _log(1.0 + (N_total - doc_cnt + 0.5) / (doc_cnt + 0.5))
            self.c_ngram_idf[country] = ng_idf

            idx_tok = self.c_tokens[country]
            cnt_tok = self.c_token_doc_counts[country]
            tok_idf = {}
            for t, doc_cnt in cnt_tok.items():
                if t not in idx_tok or doc_cnt == 0:
                    continue
                tok_idf[t] = _log(1.0 + (N_total - doc_cnt + 0.5) / (doc_cnt + 0.5))
            self.c_token_idf[country] = tok_idf

            idx_addr = self.c_addr_tokens[country]
            cnt_addr = self.c_addr_token_doc_counts[country]
            addr_idf = {}
            for at, doc_cnt in cnt_addr.items():
                if at not in idx_addr or doc_cnt == 0:
                    continue
                addr_idf[at] = _log(1.0 + (N_total - doc_cnt + 0.5) / (doc_cnt + 0.5))
            self.c_addr_token_idf[country] = addr_idf

    def query_record(
        self,
        eid: str,
        country: str,
        norm_name: str,
        norm_address: str,
        postal_code: str,
        street_number: str,
        trailing_segment: str,
        is_address_missing: bool,
    ) -> List[Tuple[str, str, str, str, int, int, float, str]]:
        """
        Query candidate pool for a single S1 record with integer accumulators and full channel support.
        """
        max_cand = self.max_candidates
        top_k = self.top_k_sparse
        sim_floor = self.similarity_floor
        _sqrt = math.sqrt
        _log = math.log

        # Country partition dispatch
        c_country = country if country in self.countries else ""

        # Flat candidate hits: int_id -> [best_score, best_rank, hit_exact, blockers_list]
        hits: Dict[int, List[Any]] = {}

        def record_hit(cid: int, blocker: str, score: float, rank: int, exact: bool = False):
            entry = hits.get(cid)
            if entry is None:
                hits[cid] = [score, rank, exact, [blocker]]
            else:
                entry[3].append(blocker)
                if score > entry[0]:
                    entry[0] = score
                if rank < entry[1]:
                    entry[1] = rank
                if exact:
                    entry[2] = True

        # Channel 1: Exact Name
        if norm_name:
            cands = self.c_exact_name[c_country].get(norm_name, [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "exact_name", 1.0, r, True)

        # Channel 1: First 2 tokens & Acronym
        tokens = norm_name.split()
        if len(tokens) >= 2:
            f2 = f"{tokens[0]} {tokens[1]}"
            cands = self.c_first_2_tokens[c_country].get(f2, [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "first_2_tokens", 0.90, r, True)

            acr = f"{tokens[0][0]}{tokens[1][0]}"
            cands = self.c_acronym[c_country].get(acr, [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "acronym_match", 0.85, r, True)

        first_word = tokens[0] if tokens else ""

        # Composite structural
        if norm_name and postal_code:
            cands = self.c_name_postal[c_country].get((norm_name, postal_code), [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "exact_name_postal", 1.0, r, True)

        if norm_name and street_number:
            cands = self.c_name_street[c_country].get((norm_name, street_number), [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "exact_name_street", 0.95, r, True)

        if norm_name and trailing_segment:
            cands = self.c_name_trailing[c_country].get((norm_name, trailing_segment), [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "exact_name_trailing", 0.95, r, True)

        if first_word and postal_code and len(first_word) >= 3:
            cands = self.c_lead_postal[c_country].get((first_word, postal_code), [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "name_lead_postal", 0.90, r, True)

        if first_word and street_number and trailing_segment and len(first_word) >= 3:
            cands = self.c_lead_street_trail[c_country].get((first_word, street_number, trailing_segment), [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "name_lead_street_trailing", 0.90, r, True)

        # -------------------------------------------------------------
        # Tier 1 (Fast & High-Precision): Address Structural & Address Tokens
        # -------------------------------------------------------------
        # Channel 4: Address Structural Key
        if not is_address_missing and postal_code and street_number:
            cands = self.c_addr_struct[c_country].get((postal_code, street_number), [])
            for r, cid in enumerate(cands[:max_cand], 1):
                record_hit(cid, "address_structural", 0.85, r, False)

        # Channel 4b: Address Inverted Index (Tokens with Sub-Linear TF-IDF)
        if not is_address_missing and norm_address:
            addr_idf_table = self.c_addr_token_idf.get(c_country)
            if addr_idf_table:
                s1_addr_tokens = [t for t in _tokenize(norm_address) if len(t) >= 3 and t not in _ADDRESS_STOPWORDS]
                if s1_addr_tokens:
                    s1_addr_counts = Counter(s1_addr_tokens)
                    s1_addr_weights = {}
                    for tok, count in s1_addr_counts.items():
                        idf = addr_idf_table.get(tok)
                        if idf is not None:
                            s1_addr_weights[tok] = (1.0 + _log(count)) * idf

                    if s1_addr_weights:
                        if len(s1_addr_weights) > 10:
                            top_items = heapq.nlargest(10, s1_addr_weights.items(), key=_itemgetter_1)
                            s1_addr_weights = dict(top_items)

                        addr_scores: Dict[int, float] = defaultdict(float)
                        idx_addr = self.c_addr_tokens[c_country]
                        for tok, w_s1 in s1_addr_weights.items():
                            postings = idx_addr.get(tok)
                            if postings:
                                for cid in postings:
                                    addr_scores[cid] += w_s1

                        if addr_scores:
                            s1_a_norm = _sqrt(sum(w * w for w in s1_addr_weights.values()))
                            if s1_a_norm > 0:
                                if len(addr_scores) > top_k:
                                    top_addr_cands = heapq.nlargest(top_k, addr_scores.items(), key=_itemgetter_1)
                                else:
                                    top_addr_cands = addr_scores.items()

                                cand_addr_lens = self.cand_addr_lens
                                for r, (cid, raw_score) in enumerate(top_addr_cands, 1):
                                    c_len = cand_addr_lens[cid] if cid < len(cand_addr_lens) else 3
                                    sim = raw_score / (s1_a_norm * _sqrt(c_len))
                                    if sim > 1.0:
                                        sim = 1.0
                                    if sim >= 0.20:
                                        record_hit(cid, "address_tokens", sim, r, exact=(sim >= 0.75))

        # -------------------------------------------------------------
        # Tiered Early-Exit Gate:
        # If high-confidence matches are already found, skip expensive fuzzy token & character n-gram channels
        # -------------------------------------------------------------
        has_exact = any(h[2] for h in hits.values())
        skip_fuzzy = (has_exact and len(hits) >= 10) or len(hits) >= max_cand

        # -------------------------------------------------------------
        # Tier 2 (Fuzzy Fallback): Token & Character N-Gram Inverted Indexes
        # -------------------------------------------------------------
        if not skip_fuzzy:
            # Channel 3: Token Inverted Index
            if tokens:
                tok_idf_table = self.c_token_idf.get(c_country)
                if tok_idf_table:
                    tok_counts = Counter(tokens)
                    s1_tok_weights = {}
                    for t, cnt in tok_counts.items():
                        idf = tok_idf_table.get(t)
                        if idf is not None:
                            s1_tok_weights[t] = (1.0 + _log(cnt)) * idf

                    if s1_tok_weights:
                        if len(s1_tok_weights) > 10:
                            top_items = heapq.nlargest(10, s1_tok_weights.items(), key=_itemgetter_1)
                            s1_tok_weights = dict(top_items)

                        cand_tok_scores: Dict[int, float] = defaultdict(float)
                        idx_tok = self.c_tokens[c_country]
                        for t, w in s1_tok_weights.items():
                            postings = idx_tok.get(t)
                            if postings:
                                for cid in postings:
                                    cand_tok_scores[cid] += w

                        if cand_tok_scores:
                            s1_norm = _sqrt(sum(w * w for w in s1_tok_weights.values()))
                            if s1_norm > 0:
                                top_tok_cands = heapq.nlargest(top_k * 3, cand_tok_scores.items(), key=_itemgetter_1)
                                scored_tok_cands = []
                                cand_tok_lens = self.cand_token_lens
                                for cid, raw_score in top_tok_cands:
                                    c_len = cand_tok_lens[cid] if cid < len(cand_tok_lens) else 3
                                    sim = raw_score / (s1_norm * _sqrt(c_len))
                                    if sim >= sim_floor:
                                        scored_tok_cands.append((min(1.0, sim), cid))

                                scored_tok_cands.sort(key=_itemgetter_0, reverse=True)
                                for r, (sim, cid) in enumerate(scored_tok_cands[:top_k], 1):
                                    record_hit(cid, "token_inverted", sim, r, False)

            # Channel 2: Character N-Grams
            ngrams = get_char_ngrams(norm_name)
            idf_table = self.c_ngram_idf.get(c_country)
            if ngrams and idf_table:
                ng_counts = Counter(ngrams)
                s1_weights = {}
                for g, cnt in ng_counts.items():
                    idf = idf_table.get(g)
                    if idf is not None:
                        s1_weights[g] = (1.0 + _log(cnt)) * idf

                if s1_weights:
                    if len(s1_weights) > 15:
                        top_items = heapq.nlargest(15, s1_weights.items(), key=_itemgetter_1)
                        s1_weights = dict(top_items)

                    cand_scores: Dict[int, float] = defaultdict(float)
                    idx_ng = self.c_ngrams[c_country]
                    for g, w in s1_weights.items():
                        postings = idx_ng.get(g)
                        if postings:
                            for cid in postings:
                                cand_scores[cid] += w

                    if cand_scores:
                        s1_norm = _sqrt(sum(w * w for w in s1_weights.values()))
                        if s1_norm > 0:
                            top_cands = heapq.nlargest(top_k * 3, cand_scores.items(), key=_itemgetter_1)
                            scored_cands = []
                            cand_lens = self.cand_ngram_lens
                            for cid, raw_score in top_cands:
                                c_len = cand_lens[cid] if cid < len(cand_lens) else 25
                                sim = raw_score / (s1_norm * _sqrt(c_len))
                                if sim >= sim_floor:
                                    scored_cands.append((min(1.0, sim), cid))

                            scored_cands.sort(key=_itemgetter_0, reverse=True)
                            for r, (sim, cid) in enumerate(scored_cands[:top_k], 1):
                                record_hit(cid, "char_ngram", sim, r, False)

        if not hits:
            return []

        # Strict Exact-Priority Sorting:
        # 1. Exact match (-1 vs 0) -> strict priority!
        # 2. Number of distinct blockers (-b_cnt)
        # 3. Best score (-best_score)
        # 4. Best rank (best_rank)
        # 5. Candidate string ID (cid_str)
        id_strs = self.id_to_str
        id_srcs = self.id_to_source
        sortable = []
        for cid, (best_score, best_rank, hit_exact, b_list) in hits.items():
            if len(b_list) == 1:
                b_cnt = 1
                is_ex = hit_exact or (b_list[0] in EXACT_BLOCKERS)
            else:
                uniq_b = set(b_list)
                b_cnt = len(uniq_b)
                is_ex = hit_exact or bool(uniq_b & EXACT_BLOCKERS)

            cid_str = id_strs[cid]
            sort_key = (
                -1 if is_ex else 0,
                -b_cnt,
                -best_score,
                best_rank,
                cid_str
            )
            sortable.append((sort_key, cid, b_list, best_score, best_rank))

        sortable.sort(key=_itemgetter_0)
        top_selected = sortable[:max_cand]

        results = []
        for sort_key, cid, b_list, best_score, best_rank in top_selected:
            cid_str = id_strs[cid]
            c_src = id_srcs[cid]
            uniq_b = sorted(set(b_list))
            prov = ",".join(uniq_b)
            score = round(best_score, 4)
            results.append((
                eid, cid_str, c_src, prov, len(uniq_b), best_rank, score, country
            ))

        return results


def run_fast_blocking(
    source1_paths: Sequence[Path | str],
    candidate_sources: Sequence[Path | str],
    output_dir: Path | str,
    ground_truth_path: Optional[Path | str] = None,
    top_k_sparse: int = TOP_K_SPARSE,
    max_candidates_per_entity: int = MAX_CANDIDATES,
    similarity_floor: float = SIMILARITY_FLOOR,
) -> Dict[str, Any]:
    """
    Execute high-performance Stage 1 blocking with Polars ingestion and FastNormalizedBlocker.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocker = FastNormalizedBlocker(
        top_k_sparse=top_k_sparse,
        max_candidates=max_candidates_per_entity,
        similarity_floor=similarity_floor,
    )

    # 1. Ingest candidates
    t0_idx = time.perf_counter()
    total_indexed = 0
    for c_path in candidate_sources:
        n = blocker.index_normalized_file(c_path)
        total_indexed += n
    blocker.build_idf_tables()
    idx_duration = time.perf_counter() - t0_idx
    print(f"Indexed {total_indexed:,} candidate records across {len(blocker.countries)} country partitions in {idx_duration:.3f}s.")

    # 2. Load Ground Truth if provided
    ground_truth: Dict[str, Set[str]] = {}
    if ground_truth_path:
        gt_path = Path(ground_truth_path)
        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2 and row[1].strip():
                        ground_truth[row[0].strip()] = set(row[1].strip().split(","))

    # 3. Read Source 1 records via Polars
    t0_query = time.perf_counter()
    pairs_tsv = output_dir / "candidate_pairs.tsv"
    prov_tsv = output_dir / "candidate_provenance.tsv"

    total_s1 = 0
    total_pairs = 0
    singletons = 0
    cands_per_s1 = []

    # Recall audit tracking
    recovered_gt_pairs = 0
    total_gt_pairs = sum(len(matches) for matches in ground_truth.values()) if ground_truth else 0
    entity_hits = 0
    any_hits = 0
    country_gt: Dict[str, int] = Counter()
    country_recovered: Dict[str, int] = Counter()
    channel_recovered: Dict[str, int] = Counter()

    with open(pairs_tsv, "w", encoding="utf-8", newline="") as fp_pairs, \
         open(prov_tsv, "w", encoding="utf-8", newline="") as fp_prov:

        out_buf_pairs = io.StringIO()
        out_buf_prov = io.StringIO()
        w_pairs = csv.writer(out_buf_pairs, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w_prov = csv.writer(out_buf_prov, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")

        w_pairs.writerow(["source1_entity_id", "candidate_entity_ids"])
        w_prov.writerow([
            "source1_entity_id", "candidate_entity_id", "candidate_source",
            "blocker_provenance", "blocker_count", "best_blocker_rank",
            "best_blocker_score", "country"
        ])

        for s1_path in source1_paths:
            df = pl.read_csv(
                s1_path,
                separator="\t",
                columns=[
                    "entity_id", "country_canonical", "norm_name", "norm_address",
                    "postal_code", "street_number", "trailing_segment", "is_address_missing"
                ],
                schema_overrides={
                    "entity_id": pl.String,
                    "country_canonical": pl.String,
                    "norm_name": pl.String,
                    "norm_address": pl.String,
                    "postal_code": pl.String,
                    "street_number": pl.String,
                    "trailing_segment": pl.String,
                    "is_address_missing": pl.Int32,
                }
            )

            eids = df["entity_id"].to_list()
            countries = df["country_canonical"].fill_null("").to_list()
            names = df["norm_name"].fill_null("").to_list()
            addrs = df["norm_address"].fill_null("").to_list()
            postals = df["postal_code"].fill_null("").to_list()
            streets = df["street_number"].fill_null("").to_list()
            trails = df["trailing_segment"].fill_null("").to_list()
            missings = df["is_address_missing"].fill_null(0).to_list()

            for i in range(df.height):
                eid = eids[i]
                c_country = countries[i]
                hits = blocker.query_record(
                    eid=eid,
                    country=c_country,
                    norm_name=names[i],
                    norm_address=addrs[i],
                    postal_code=postals[i],
                    street_number=streets[i],
                    trailing_segment=trails[i],
                    is_address_missing=bool(missings[i]),
                )

                total_s1 += 1
                n_cands = len(hits)
                cands_per_s1.append(n_cands)

                if n_cands == 0:
                    singletons += 1
                    w_pairs.writerow([eid, ""])
                else:
                    cand_ids = [h[1] for h in hits]
                    w_pairs.writerow([eid, ",".join(cand_ids)])
                    total_pairs += n_cands
                    for h in hits:
                        w_prov.writerow(h)

                if total_s1 % 5000 == 0:
                    fp_pairs.write(out_buf_pairs.getvalue())
                    fp_prov.write(out_buf_prov.getvalue())
                    out_buf_pairs.seek(0); out_buf_pairs.truncate(0)
                    out_buf_prov.seek(0); out_buf_prov.truncate(0)

                # Recall audit
                if eid in ground_truth:
                    true_cands = ground_truth[eid]
                    c_gt_count = len(true_cands)
                    country_gt[c_country] += c_gt_count
                    if hits:
                        found_cands = {h[1] for h in hits}
                        matches_found = true_cands & found_cands
                        rec_count = len(matches_found)
                        recovered_gt_pairs += rec_count
                        country_recovered[c_country] += rec_count
                        if rec_count > 0:
                            any_hits += 1
                        if rec_count == c_gt_count:
                            entity_hits += 1

                        for h in hits:
                            cid = h[1]
                            if cid in true_cands:
                                for b in h[3].split(","):
                                    channel_recovered[b] += 1

        # Final flush
        fp_pairs.write(out_buf_pairs.getvalue())
        fp_prov.write(out_buf_prov.getvalue())

    query_duration = time.perf_counter() - t0_query

    cands_arr = np.array(cands_per_s1) if cands_per_s1 else np.array([0])
    summary: Dict[str, Any] = {
        "engine": "FastNormalizedBlocker",
        "total_source1_entities": total_s1,
        "total_candidate_pairs": total_pairs,
        "singletons_with_zero_candidates": singletons,
        "indexing_duration_s": round(idx_duration, 4),
        "query_duration_s": round(query_duration, 4),
        "throughput_queries_per_s": round(total_s1 / query_duration, 2) if query_duration > 0 else 0,
        "candidates_per_s1": {
            "mean": round(float(np.mean(cands_arr)), 2),
            "median": int(np.median(cands_arr)),
            "p90": int(np.percentile(cands_arr, 90)),
            "p95": int(np.percentile(cands_arr, 95)),
            "max": int(np.max(cands_arr)),
        }
    }

    if ground_truth:
        gt_entities = len(ground_truth)
        pair_recall = recovered_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
        entity_recall = entity_hits / gt_entities if gt_entities > 0 else 0.0
        any_hit_rate = any_hits / gt_entities if gt_entities > 0 else 0.0

        country_rec = {
            c: round(country_recovered[c] / country_gt[c], 4)
            for c in country_gt if country_gt[c] > 0
        }
        channel_rec = {
            ch: round(channel_recovered[ch] / total_gt_pairs, 4)
            for ch in channel_recovered
        }

        summary["recall_audit"] = {
            "total_ground_truth_pairs": total_gt_pairs,
            "recovered_ground_truth_pairs": recovered_gt_pairs,
            "pair_recall": round(pair_recall, 4),
            "entity_full_recall": round(entity_recall, 4),
            "any_hit_rate": round(any_hit_rate, 4),
            "recall_by_country": country_rec,
            "recall_by_channel": channel_rec,
        }

    with open(output_dir / "blocking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fast Normalized Blocking Engine")
    parser.add_argument("--source1", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--top-k-sparse", type=int, default=TOP_K_SPARSE)
    parser.add_argument("--similarity-floor", type=float, default=SIMILARITY_FLOOR)
    args = parser.parse_args()

    run_fast_blocking(
        source1_paths=args.source1,
        candidate_sources=args.candidate_sources,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        top_k_sparse=args.top_k_sparse,
        max_candidates_per_entity=args.max_candidates,
        similarity_floor=args.similarity_floor,
    )


if __name__ == "__main__":
    main()

