#!/usr/bin/env python3
"""Fetch and record the evidence behind every prior-art claim in SELVEDGE.

The mandate forbids fabricated citations. This script is the reproducible
half of `docs/PRIOR-ART-AND-NOVELTY.md`: it contacts each declared source,
records the HTTP status, the retrieval time, the returned title and the first
part of the abstract (or page description), and writes one JSONL record per
source. The ledger's VERIFIED / PARTIAL / UNSUPPORTED / PROPOSED status is a
human judgement made **from these recorded bytes**, not from memory.

Methods:

  arxiv_id      authoritative metadata from the arXiv Atom API for one id
  arxiv_search  exact-title search; all candidate titles are recorded, so a
                near-miss cannot silently become a citation
  page          HTML landing page; title, meta description and a text snippet
  github        repository landing page; title, description and default branch

Usage:
  scripts/court --net python -- python3 scripts/prior_art_fetch.py \
      --out results/prior-art/prior-art-fetch.jsonl

Exit status is 0 even when a source is missing: a missing source is a
finding, and the record says so. Only a total failure to write the output is
an error.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree

RETRIEVED: str | None = None
USER_AGENT = "SELVEDGE-prior-art/1 (+research citation verification)"
ATOM = "{http://www.w3.org/2005/Atom}"

# Each entry: key, method, target, and what the strategy note claims it says.
SOURCES = [
    # --- mandate list, direct URLs -----------------------------------------
    ("dejavu", "page", "https://proceedings.mlr.press/v202/liu23am.html",
     "contextual activation sparsity, learned predictor, FFN skipping"),
    ("powerinfer", "page", "https://doi.org/10.1145/3694715.3695964",
     "hot/cold neuron split, activation prediction, heterogeneous sparse inference"),
    ("attribution-2026", "page", "https://pmc.ncbi.nlm.nih.gov/articles/PMC13570667/",
     "corrected gradient attribution and group-dependence caveats"),
    ("seap", "page", "https://ojs.aaai.org/index.php/AAAI/article/view/40463",
     "task-domain routing and semantic core-neuron prediction"),
    ("sparseinfer", "page", "https://proceedings.mlr.press/v306/wang26hn.html",
     "semantic core-neuron prediction"),
    ("simoe", "page", "https://aclanthology.org/2025.acl-long.816/",
     "dense-to-sparse upcycling with additional training"),
    ("upit", "page", "https://aclanthology.org/2025.acl-long.637/",
     "dense-to-sparse upcycling of experts"),
    ("peer", "arxiv_id", "2407.04153", "million tiny experts, product-key retrieval"),
    ("monet", "page",
     "https://proceedings.iclr.cc/paper_files/paper/2025/hash/"
     "382c35d1a57c07055dfcba58dd39e012-Abstract-Conference.html",
     "fine-grained indexing over many experts"),
    ("mose", "page", "https://proceedings.mlr.press/v306/tastan26a.html",
     "variable width / dynamic expert computation"),
    ("dagmoe", "page", "https://proceedings.mlr.press/v306/feng26v.html",
     "expert-output composition as a DAG"),
    ("probmoe", "page", "https://proceedings.mlr.press/v306/zhao26ap.html",
     "dynamic routing / probabilistic expert selection"),
    ("memory-layers", "page", "https://proceedings.mlr.press/v267/berges25a.html",
     "learnable key-value memory lookup distinct from procedural computation"),
    ("mote", "arxiv_id", "2506.14435", "mixture of ternary experts, low-bit experts"),
    ("puzzlemoe", "page", "https://proceedings.mlr.press/v306/zhao26ay.html",
     "redundancy-aware expert storage"),
    ("moe-infinity", "github", "https://github.com/EfficientMoE/MoE-Infinity",
     "SSD offloading, cache policy, expert prefetch"),
    ("specmd", "arxiv_id", "2602.03921", "speculative / model-level offloading?"),
    ("specprefetch", "arxiv_id", "2607.24787", "predictive weight prefetching"),
    ("seqmoe", "arxiv_id", "2609.12978", "sequence-level expert caching / router-cache coupling"),
    ("domoe", "page", "https://ijcai.org/proceedings/2026/657",
     "domain-level routing tables / task-conditioned placement"),
    ("task-coactivation", "page", "https://www.alphaxiv.org/abs/2606.01007",
     "task-aware coactivation grouping and placement"),
    ("prefetch-steered", "arxiv_id", "2610.34721",
     "prefetch-steered routing, changing routing for residency"),
    ("moe-core", "arxiv_id", "2610.01950", "coordinated residency, quality/latency of routing"),
    ("olmoe", "arxiv_id", "2409.02060", "open sparse MoE baseline"),
    ("huginn", "arxiv_id", "2502.05171", "recurrent depth, latent recurrence"),
    ("critical-recurrent", "arxiv_id", "2507.02199",
     "critical analysis of recurrent-depth reasoning claims"),
    ("vllm-prefix", "page", "https://docs.vllm.ai/en/v0.13.0/design/prefix_caching/",
     "KV reuse keyed by exact token prefix"),
    ("gguf", "github", "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md",
     "GGUF file layout"),
    ("candle", "github", "https://github.com/huggingface/candle",
     "Rust inference primitives"),
    ("lm-eval", "github", "https://github.com/EleutherAI/lm-evaluation-harness",
     "independent evaluation harness"),
    # --- WS5 mechanisms, exact-title arXiv searches ------------------------
    ("spax", "arxiv_search", "SpAx", "sparse activation with ?"),
    ("r-sparse", "arxiv_search", "R-Sparse", "rank-aware activation sparsity"),
    ("larosa", "arxiv_search", "LaRoSA", "layerwise rotated sparse activation"),
    ("sirius", "arxiv_search", "Sirius", "contextual sparsity with correction"),
    ("blockffn", "arxiv_search", "BlockFFN", "chunk-level activation sparsity"),
    ("neuroprefetcher", "arxiv_search", "NeuroPrefetcher", "learned weight prefetching"),
    ("tomoe", "arxiv_search", "ToMoE", "task-oriented MoE?"),
    ("drop-upcycling", "arxiv_search", "Drop-Upcycling",
     "drop-based dense-to-MoE upcycling"),
    ("cluster-upcycling", "arxiv_search", "cluster upcycling mixture of experts",
     "cluster-aware expert initialisation"),
    ("ders", "arxiv_search", "DeRS", "delta-compressed shared-base experts"),
    ("d2-moe", "arxiv_search", "D2-MoE", "delta / dual expert decomposition"),
    ("maskerade", "arxiv_search", "MASKerade", "sequence-level masked expert bank"),
    ("mor", "arxiv_search", "Mixture of Recursions", "adaptive recursive depth"),
    ("wina", "arxiv_search", "WINA", "weight-informed neuron activation"),
    ("norsa", "arxiv_search", "NorSA", "normalised sparse activation"),
    ("sga-sparse", "arxiv_search", "SGA-Sparse", "native gate as router"),
    ("lopa", "arxiv_search", "LoPA", "low-rank parameter adaptation"),
    ("model-casting", "arxiv_search", "Model Casting", "casting dense to sparse?"),
    ("hope", "arxiv_search", "HOPE", "low-rank pairwise interaction for experts"),
    ("seqtok", "arxiv_search", "SeqTopK", "sequence-level top-k budget allocation"),
    ("cmoe", "arxiv_search", "CMoE", "conditional / clustered MoE"),
    ("spon", "arxiv_search", "SPON", "spontaneous anchor vectors"),
    ("engram", "arxiv_search", "Engram", "conditional learned memory"),
    ("fast-feedforward", "arxiv_id", "2308.14711",
     "fast feedforward networks, conditionally executed nodes"),
    # --- round 2: alternate sources and all-field searches ------------------
    ("powerinfer-github", "github", "https://github.com/SJTU-IPADS/PowerInfer",
     "hot/cold neuron split, activation prediction, heterogeneous sparse inference"),
    ("powerinfer-acm", "page", "https://dl.acm.org/doi/10.1145/3694715.3695964",
     "PowerInfer SOSP 2024 landing page"),
    ("larosa-2", "arxiv_all", 'all:"layerwise rotated sparse activation"',
     "LaRoSA layerwise rotated sparse activation"),
    ("sirius-2", "arxiv_all", 'all:"Sirius" AND all:"contextual sparsity"',
     "Sirius contextual sparsity with correction"),
    ("spax-2", "arxiv_all", 'all:"SpAx"', "SpAx sparse activation"),
    ("cluster-upcycling-2", "arxiv_all",
     'all:"upcycling" AND all:"expert" AND all:"clustering"',
     "cluster-aware expert initialisation for upcycled MoE"),
    ("d2-moe-2", "arxiv_all", 'all:"D2-MoE"', "D-squared MoE delta experts"),
    ("norsa-2", "arxiv_all", 'all:"NorSA"', "NorSA activation sparsity"),
    ("seqtok-2", "arxiv_all", 'all:"SeqTopK"', "sequence-level top-k budget"),
    ("hope-2", "arxiv_all",
     'all:"HOPE" AND all:"mixture of experts"', "HOPE pairwise interaction"),
    ("spon-2", "arxiv_all", 'all:"SPON" AND all:"anchor"',
     "SPON spontaneous anchor vectors"),
    ("cmoe-2", "arxiv_all", 'all:"CMoE" AND all:"mixture of experts"',
     "CMoE clustered/conditional MoE"),
    ("engram-primary", "arxiv_all",
     'all:"Engram" AND all:"conditional memory"',
     "Engram conditional memory, primary source"),
    ("engram-title", "arxiv_all", 'ti:"Engram"',
     "Engram primary source, title search"),
    ("tomoe-detail", "arxiv_id", "2501.15316",
     "ToMoE dense-to-MoE dynamic structural pruning"),
    ("ders-detail", "arxiv_id", "2503.01359",
     "DeRS upcycled MoE compression"),
    ("r-sparse-detail", "arxiv_id", "2504.19449",
     "R-Sparse rank-aware activation sparsity"),
    ("mor-detail", "arxiv_id", "2507.10524",
     "Mixture-of-Recursions adaptive recursive depth"),
    ("larosa-3", "arxiv_all", 'all:"Layerwise Rotated" AND all:"sparse activation"',
     "LaRoSA layerwise rotated sparse activation"),
    ("sirius-3", "arxiv_all", 'all:"Sirius" AND all:"sparsity"',
     "Sirius sparsity with correction"),
    ("hope-title", "arxiv_all", 'ti:"HOPE"', "HOPE interaction method"),
    ("spon-title", "arxiv_all", 'ti:"SPON"', "SPON anchor vectors"),
    ("d2-moe-3", "arxiv_all", 'all:"delta experts" AND all:"mixture of experts"',
     "delta-parameterised experts"),
    ("cluster-upcycling-3", "arxiv_all",
     'all:"cluster" AND all:"upcycling" AND all:"mixture of experts"',
     "cluster-aware upcycling"),
    ("seqtok-3", "arxiv_all",
     'all:"sequence-level" AND all:"top-k" AND all:"experts"',
     "SeqTopK sequence-level expert budget"),
    # --- round 3: resolve the ids found in round 2 --------------------------
    ("spax-id", "arxiv_id", "2610.02598", "SpAx activation sparsity + weight approximation"),
    ("cluster-id", "arxiv_id", "2604.13508", "cluster-aware upcycling"),
    ("d2-id", "arxiv_id", "2602.09316", "D2-MoE heterogeneous inter-group experts"),
    ("seqtok-id", "arxiv_id", "2511.06494", "sequence-level expert routing"),
    ("spon-id", "arxiv_id", "2512.12744", "SPON spontaneous/resting neurons"),
    ("larosa-id", "arxiv_id", "2507.01299", "La RoSA layerwise rotated sparse activation"),
    ("sirius-id", "arxiv_id", "2409.03856", "Sirius contextual sparsity with correction"),
    ("cmoe-upcycle", "arxiv_all", 'all:"CMoE" AND all:"dense"',
     "CMoE dense-to-MoE upcycling"),
    ("engram-deepseek", "arxiv_all", 'all:"Engram" AND all:"DeepSeek"',
     "DeepSeek Engram primary source"),
    ("temporally-extended", "arxiv_id", "2604.20156",
     "temporally extended MoE, sequence-level expert stability"),
    ("pooling-engram-cxl", "arxiv_id", "2603.10087",
     "Engram conditional memory offloaded to CXL"),
    ("olmoe-active", "html_grep", "https://arxiv.org/html/2409.02060v2||1.3B",
     "OLMoE active parameters are about 1.3B"),
    ("powerinfer-s2", "s2", "10.1145/3694715.3695964",
     "PowerInfer abstract from Semantic Scholar"),
    # --- round 4: full abstracts from the landing pages ---------------------
    ("dejavu-abs", "page_abstract", "https://proceedings.mlr.press/v202/liu23am.html",
     "Deja Vu abstract"),
    ("sparseinfer-abs", "page_abstract", "https://proceedings.mlr.press/v306/wang26hn.html",
     "SparseInfer abstract"),
    ("mose-abs", "page_abstract", "https://proceedings.mlr.press/v306/tastan26a.html",
     "MoSE abstract"),
    ("dagmoe-abs", "page_abstract", "https://proceedings.mlr.press/v306/feng26v.html",
     "DAG-MoE abstract"),
    ("probmoe-abs", "page_abstract", "https://proceedings.mlr.press/v306/zhao26ap.html",
     "ProbMoE abstract"),
    ("puzzlemoe-abs", "page_abstract", "https://proceedings.mlr.press/v306/zhao26ay.html",
     "PuzzleMoE abstract"),
    ("memory-layers-abs", "page_abstract", "https://proceedings.mlr.press/v267/berges25a.html",
     "Memory Layers at Scale abstract"),
    ("simoe-abs", "page_abstract", "https://aclanthology.org/2025.acl-long.816/",
     "SIMoE abstract"),
    ("upit-abs", "page_abstract", "https://aclanthology.org/2025.acl-long.637/",
     "UpIT abstract"),
    ("seap-abs", "page_abstract", "https://ojs.aaai.org/index.php/AAAI/article/view/40463",
     "SEAP abstract"),
    ("attribution-abs", "page_abstract",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC13570667/", "attribution paper abstract"),
    ("monet-abs", "page_abstract",
     "https://proceedings.iclr.cc/paper_files/paper/2025/hash/"
     "382c35d1a57c07055dfcba58dd39e012-Abstract-Conference.html", "Monet abstract"),
    ("domoe-abs", "page_abstract", "https://ijcai.org/proceedings/2026/657",
     "DoMoE abstract"),
    ("task-coactivation-abs", "page_abstract", "https://www.alphaxiv.org/abs/2606.01007",
     "task-aware grouping abstract"),
    ("gguf-raw", "raw_grep",
     "https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md||GGUF",
     "GGUF spec text exists and defines the format"),
    ("vllm-prefix-grep", "html_grep",
     "https://docs.vllm.ai/en/v0.13.0/design/prefix_caching/||prefix",
     "vLLM prefix caching is keyed on the token prefix"),
    ("hope-3", "arxiv_all", 'ti:"HOPE" AND all:"expert"', "HOPE expert interaction"),
    ("norsa-3", "arxiv_all", 'all:"NorSA" OR ti:"NorSA"', "NorSA"),
    ("cmoe-primary", "arxiv_all", 'all:"CMoE" AND all:"upcycling"',
     "CMoE primary upcycling paper"),
    ("engram-primary-2", "arxiv_all",
     'all:"Engram" AND all:"N-gram" AND all:"memory"', "Engram primary source"),
    ("d2moe-grep", "html_grep", "https://arxiv.org/abs/2602.09316||D2-MoE",
     "whether D2-MoE is the acronym of the RFID-MoE paper"),
    # --- round 5: remaining attribution checks ------------------------------
    ("vllm-prefix-hash", "html_grep",
     "https://docs.vllm.ai/en/v0.13.0/design/prefix_caching/||block hash",
     "prefix caching keyed by block hash of the token prefix"),
    ("attribution-groups", "html_grep",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC13570667/||group",
     "group-level attribution and group dependence"),
    ("engram-hash", "arxiv_all", 'all:"Engram" AND all:"hash"',
     "Engram primary source via hashing term"),
    ("mor-abstract", "arxiv_id", "2507.10524", "MoR abstract"),
    ("si-specmd-detail", "arxiv_id", "2602.03921", "SpecMD full abstract"),
    ("domoe-search", "arxiv_all", 'all:"DoMoE" AND all:"expert prediction"',
     "DoMoE primary abstract"),
    ("monet-search", "arxiv_all", 'ti:"Monet" AND all:"mixture of experts"',
     "Monet primary abstract"),
    ("task-coact-search", "arxiv_all",
     'ti:"task-aware" AND ti:"expert" AND ti:"grouping"',
     "task-aware coactivation grouping primary abstract"),
    ("domoe-grep", "html_grep", "https://ijcai.org/proceedings/2026/657||offloading",
     "DoMoE abstract body"),
    ("task-coact-arxiv", "arxiv_all", 'ti:"Task-Aware Grouping"',
     "task-aware grouping primary abstract"),
    ("monet-id", "arxiv_id", "2412.04139", "Monet full abstract"),
    ("moe-infinity-readme", "raw_grep",
     "https://raw.githubusercontent.com/EfficientMoE/MoE-Infinity/main/README.md||prefetch",
     "MoE-Infinity offloading and prefetch features"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get(url: str, accept: str = "*/*") -> tuple[int, bytes, str]:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": accept,
    })
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return response.status, response.read(), response.geturl()
    except urllib.error.HTTPError as error:
        body = b""
        try:
            body = error.read()
        except Exception:  # noqa: BLE001 - the status is the evidence
            pass
        return error.code, body, url
    except Exception as error:  # noqa: BLE001 - network failure is a finding
        return 0, str(error).encode("utf-8", "replace"), url


def text_of(fragment: bytes | str) -> str:
    if isinstance(fragment, bytes):
        fragment = fragment.decode("utf-8", "replace")
    fragment = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", fragment)
    fragment = re.sub(r"(?s)<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(fragment)).strip()


def meta(html_text: str, names: list[str]) -> str:
    for name in names:
        pattern = (r'<meta[^>]+(?:name|property)=["\']%s["\'][^>]+content=["\']([^"\']*)'
                   % re.escape(name))
        match = re.search(pattern, html_text, flags=re.I)
        if match:
            return html.unescape(match.group(1)).strip()
        pattern = (r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+(?:name|property)=["\']%s["\']'
                   % re.escape(name))
        match = re.search(pattern, html_text, flags=re.I)
        if match:
            return html.unescape(match.group(1)).strip()
    return ""


def html_grep_record(key: str, target: str, claim: str) -> dict:
    url, pattern = target.split("||", 1)
    status, body, final = get(url)
    raw = body.decode("utf-8", "replace")
    text = text_of(raw)
    match = re.search(pattern, text)
    start = max(0, match.start() - 220) if match else 0
    return {
        "key": key,
        "claim": claim,
        "method": "html_grep",
        "url": url,
        "final_url": final,
        "http_status": status,
        "pattern": pattern,
        "matched": bool(match),
        "context": text[start:start + 520] if match else text[:400],
        "bytes": len(body),
        "error": "" if status == 200 else raw[:200],
        "retrieved_utc": RETRIEVED,
    }


def s2_record(key: str, doi: str, claim: str) -> dict:
    url = ("https://api.semanticscholar.org/graph/v1/paper/DOI:" + doi
           + "?fields=title,abstract,year,venue,authors")
    status, body, final = get(url, accept="application/json")
    payload: dict = {}
    if status == 200:
        try:
            payload = json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            payload = {}
    return {
        "key": key,
        "claim": claim,
        "method": "s2",
        "url": url,
        "final_url": final,
        "http_status": status,
        "title": (payload.get("title") or "")[:300],
        "abstract": (payload.get("abstract") or "")[:1600],
        "year": payload.get("year"),
        "venue": payload.get("venue"),
        "error": "" if status == 200 else body[:200].decode("utf-8", "replace"),
        "retrieved_utc": RETRIEVED,
    }


def page_abstract_record(key: str, url: str, claim: str) -> dict:
    status, body, final = get(url)
    raw = body.decode("utf-8", "replace")
    abstract = ""
    for pattern in (
        r'(?is)<div[^>]+id="abstract"[^>]*>(.*?)</div>',
        r'(?is)<div[^>]+class="[^"]*acl-abstract[^"]*"[^>]*>(.*?)</div>',
        r'(?is)<section[^>]+class="[^"]*abstract[^"]*"[^>]*>(.*?)</section>',
        r'(?is)<blockquote[^>]+class="[^"]*abstract[^"]*"[^>]*>(.*?)</blockquote>',
        r'(?is)<div[^>]+class="abstract"[^>]*>(.*?)</div>',
    ):
        match = re.search(pattern, raw)
        if match:
            candidate = text_of(match.group(1))
            if len(candidate) > len(abstract):
                abstract = candidate
    if len(abstract) < 200:
        fallback = meta(raw, ["citation_abstract", "description", "og:description"])
        if len(fallback) > len(abstract):
            abstract = fallback
    title = meta(raw, ["citation_title"]) or ""
    if not title:
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
        title = text_of(match.group(1)) if match else ""
    if "Abstract" in abstract[:40]:
        abstract = abstract.split("Abstract", 1)[1].strip()
    return {
        "key": key,
        "claim": claim,
        "method": "page_abstract",
        "url": url,
        "final_url": final,
        "http_status": status,
        "title": title[:300],
        "abstract": abstract[:2000],
        "error": "" if status == 200 else raw[:200],
        "retrieved_utc": RETRIEVED,
    }


def raw_grep_record(key: str, target: str, claim: str) -> dict:
    url, pattern = target.split("||", 1)
    status, body, final = get(url, accept="text/plain")
    raw = body.decode("utf-8", "replace")
    match = re.search(pattern, raw)
    start = max(0, match.start() - 60) if match else 0
    return {
        "key": key,
        "claim": claim,
        "method": "raw_grep",
        "url": url,
        "final_url": final,
        "http_status": status,
        "pattern": pattern,
        "matched": bool(match),
        "bytes": len(body),
        "context": raw[start:start + 400].replace("\n", " ") if match else raw[:300],
        "error": "" if status == 200 else raw[:200],
        "retrieved_utc": RETRIEVED,
    }


def page_record(key: str, url: str, claim: str) -> dict:
    status, body, final = get(url)
    raw = body.decode("utf-8", "replace") if status else body.decode("utf-8", "replace")
    title = ""
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    if match:
        title = text_of(match.group(1))
    description = meta(raw, ["citation_title"])
    abstract = meta(raw, ["citation_abstract", "description", "og:description", "twitter:description"])
    if not title:
        title = description
    if not abstract:
        abstract = text_of(raw)[:1200]
    return {
        "key": key,
        "claim": claim,
        "method": "page",
        "url": url,
        "final_url": final,
        "http_status": status,
        "bytes": len(body),
        "title": title[:300],
        "meta_title": description[:300],
        "abstract": abstract[:1600],
        "error": "" if status == 200 else raw[:200],
        "retrieved_utc": RETRIEVED,
    }


def github_record(key: str, url: str, claim: str) -> dict:
    status, body, final = get(url)
    raw = body.decode("utf-8", "replace")
    title = ""
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    if match:
        title = text_of(match.group(1))
    description = meta(raw, ["og:description", "description"])
    return {
        "key": key,
        "claim": claim,
        "method": "github",
        "url": url,
        "final_url": final,
        "http_status": status,
        "bytes": len(body),
        "title": title[:300],
        "meta_title": description[:300],
        "abstract": description[:1600],
        "error": "" if status == 200 else raw[:200],
        "retrieved_utc": RETRIEVED,
    }


def atom_entries(xml_bytes: bytes) -> list[dict]:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as error:
        return [{"parse_error": str(error)}]
    entries = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = (entry.findtext(f"{ATOM}id") or "").strip()
        title = re.sub(r"\s+", " ", (entry.findtext(f"{ATOM}title") or "")).strip()
        summary = re.sub(r"\s+", " ", (entry.findtext(f"{ATOM}summary") or "")).strip()
        published = (entry.findtext(f"{ATOM}published") or "").strip()
        authors = [author.findtext(f"{ATOM}name") or ""
                   for author in entry.findall(f"{ATOM}author")]
        entries.append({
            "arxiv_id": entry_id.rsplit("/abs/", 1)[-1],
            "title": title,
            "abstract": summary,
            "published": published,
            "authors": authors,
            "error": "arxiv api error entry" if title.lower().startswith("error") else "",
        })
    return entries


def arxiv_id_record(key: str, identifier: str, claim: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={identifier}&max_results=1"
    status, body, final = get(url)
    entries = atom_entries(body) if status == 200 else []
    first = entries[0] if entries else {}
    return {
        "key": key,
        "claim": claim,
        "method": "arxiv_id",
        "url": url,
        "final_url": final,
        "http_status": status,
        "requested_id": identifier,
        "title": first.get("title", ""),
        "arxiv_id": first.get("arxiv_id", ""),
        "published": first.get("published", ""),
        "authors": first.get("authors", []),
        "abstract": first.get("abstract", "")[:1600],
        "error": first.get("error", "") if entries else (body[:200].decode("utf-8", "replace")
                                                        if status != 200 else "no entry"),
        "retrieved_utc": RETRIEVED,
    }


def arxiv_search_record(key: str, phrase: str, claim: str) -> dict:
    query = f'ti:"{phrase}"'
    return arxiv_query_record(key, query, claim)


def arxiv_query_record(key: str, query: str, claim: str) -> dict:
    url = ("https://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(query) + "&max_results=5")
    status, body, final = get(url)
    entries = atom_entries(body) if status == 200 else []
    candidates = [{"arxiv_id": item.get("arxiv_id", ""),
                   "title": item.get("title", ""),
                   "published": item.get("published", ""),
                   "abstract": (item.get("abstract") or "")[:600]}
                  for item in entries]
    return {
        "key": key,
        "claim": claim,
        "method": "arxiv_search",
        "query": query,
        "url": url,
        "final_url": final,
        "http_status": status,
        "candidates": candidates,
        "found": len(candidates),
        "title": candidates[0]["title"] if candidates else "",
        "arxiv_id": candidates[0]["arxiv_id"] if candidates else "",
        "abstract": candidates[0]["abstract"] if candidates else "",
        "error": "" if status == 200 else body[:200].decode("utf-8", "replace"),
        "retrieved_utc": RETRIEVED,
    }


def main(argv: list[str]) -> int:
    global RETRIEVED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/prior-art/prior-art-fetch.jsonl")
    parser.add_argument("--only", default="", help="comma-separated keys")
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--retrieved", default=None)
    args = parser.parse_args(argv)

    RETRIEVED = args.retrieved or now()
    wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    records = []
    for key, method, target, claim in SOURCES:
        if wanted and key not in wanted:
            continue
        if method == "arxiv_id":
            record = arxiv_id_record(key, target, claim)
            summary = f"{record['http_status']} {record.get('arxiv_id','')} {record['title'][:90]}"
        elif method == "arxiv_search":
            record = arxiv_search_record(key, target, claim)
            titles = " | ".join(c["title"][:60] for c in record["candidates"]) or "(none)"
            summary = f"{record['http_status']} found={record['found']} {titles}"
        elif method == "arxiv_all":
            record = arxiv_query_record(key, target, claim)
            record["method"] = "arxiv_all"
            titles = " | ".join(c["title"][:60] for c in record["candidates"]) or "(none)"
            summary = f"{record['http_status']} found={record['found']} {titles}"
        elif method == "github":
            record = github_record(key, target, claim)
            summary = f"{record['http_status']} {record['title'][:90]}"
        elif method == "html_grep":
            record = html_grep_record(key, target, claim)
            summary = f"{record['http_status']} matched={record['matched']} {record.get('pattern','')}"
        elif method == "s2":
            record = s2_record(key, target, claim)
            summary = f"{record['http_status']} {record.get('venue','')} {record.get('title','')[:80]}"
        elif method == "page_abstract":
            record = page_abstract_record(key, target, claim)
            summary = (f"{record['http_status']} {record['title'][:60]} :: "
                       f"abstract={len(record['abstract'])} chars")
        elif method == "raw_grep":
            record = raw_grep_record(key, target, claim)
            summary = f"{record['http_status']} matched={record['matched']} {record.get('pattern','')}"
        else:
            record = page_record(key, target, claim)
            summary = (f"{record['http_status']} {record['meta_title'][:60]} :: "
                       f"{record['title'][:70]}")
        records.append(record)
        print(f"{key:22s} {summary}", flush=True)
        time.sleep(args.sleep)

    with open(args.out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"prior-art: wrote {len(records)} records to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
