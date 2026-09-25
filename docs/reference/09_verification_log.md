# Verification log

This design went through five rounds of pasted or uploaded documents touching on verification. The first four each claimed to correct or verify earlier work and turned out to contain a false claim; the fifth resolved, rather than contested, an earlier round's finding. Documenting the pattern here because it's directly relevant to how much trust any single claim in this docs set should get without independent re-checking.

## Round 1 — false claim that a real, on-topic paper doesn't exist

A pasted document claimed the Sodhana paper (arXiv:2608.16161) could not be found by search and should be dropped from the methodology document in favor of a different, off-topic citation (vstash). Direct search found the paper immediately, in multiple independent places: arxiv.org's own listing, a third-party review site, and the paper's own GitHub repository.

## Round 2 — false claim that a real detail within a real paper was wrong

A second pasted document conceded the paper was real, but claimed the author affiliation "Sodhana" didn't actually appear in the paper's text, and that the specific 15.25%→92.70% figure didn't match what the paper reports. A direct full-text fetch of the paper showed "Sodhana" in the byline (both as the stated affiliation and in an author's email domain) and the exact 15.25%/92.70%/37.85%/83.10% figures in Table 4, verbatim. The document had found a real, different table in the same paper (Section 4.3's head-to-head comparison) and mischaracterized it as a correction rather than an additional, compatible finding.

## Round 3 — false claim about the founding document of this entire conversation

An uploaded documentation draft claimed that France-as-held-out-country and the MIT/Apache-2.0/≤8B model constraint were never actually part of the competition brief — that they were invented in "a separate pasted transcript" and elevated to primary-source status by mistake. This conversation's actual founding document states both directly and in detail, multiple times. The claimed counter-quote from "the real brief" doesn't correspond to any document that's actually been shared in this conversation.

## Round 4 — false claim about a real project's license

The same uploaded draft claimed to have "confirmed directly on its GitHub repo metadata" that LinkTransformer is GPL-3.0-licensed, and built a non-functional requirement and a risk-register entry around avoiding its code for that reason. LinkTransformer's own README states plainly: "This project is licensed under the MIT License." Checked directly, first search.

## Round 5 — resolution: France and the license/size ceiling independently confirmed

After Round 3 rejected an uploaded draft's claim that France and the MIT/Apache-2.0/≤8B constraint were fabricated, three further documents were reviewed directly: the official Unstop competition listing, a video-breakdown transcript of the problem statement, and — decisively — the official `student_resource` problem statement itself. The Unstop listing and video transcript were silent on France and the license ceiling (expected — they're overviews, not the rules text), which briefly reopened the question of whether those constraints were real. The official problem statement resolves it: both are stated explicitly and verbatim (*"a third country, France, that does not appear in the training data"*; *"Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters"*). The video transcript also turned out to contain the exact "Account for locale-specific patterns and formatting conventions" phrase that Round 3's rejected draft had claimed didn't exist anywhere — that phrase is real, it just isn't the counter-evidence the rejected draft used it as. Net effect: Round 3's rejection was correct, now on firmer evidence than it had at the time.

## What actually held up across the first four rounds

Everything re-verified independently and found accurate: BGE-M3's and Qwen3-Embedding's specs and licenses, the Sodhana paper's existence and real numbers (once actually read rather than just searched for), TriBERTa, eridu, EnsembleLink, Jina's non-commercial license exclusion, EmbeddingGemma's Gemma-license exclusion, and the XGBoost categorical/missing-value documentation. The pattern in every failed round was the same: a specific, confident, falsifiable claim — never a vague one — that didn't survive being checked against the actual primary source, whether that source was a paper's full text, this conversation's own founding document, or a repository's own README.

## The discipline this justifies going forward

A claim that something has been "verified," "confirmed directly," or "corrected" is not itself evidence — regardless of how detailed, humble, or self-critical its framing is, and regardless of how many other claims in the same document turn out to be accurate. Every one of the four false claims above appeared alongside several true ones in the same document, which is what made each one worth taking seriously rather than dismissing outright. The only reliable response is re-deriving the specific, checkable claim from the primary source before it changes a design decision or goes into an audited methodology document — not adjudicating between competing documents by tone, and not assuming a document is trustworthy because most of it checks out.
