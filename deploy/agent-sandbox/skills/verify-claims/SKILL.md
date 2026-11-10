# Factual claim verification

Read `/workspace/CONTRACT.md`, `/workspace/contract.json`, and `/workspace/claims.json` first.

For each listed claim and each listed field, independently compare the exact field text with the frozen implementation under `/workspace/head/` and `/workspace/base/`. You may read the files you need; do not assume the generation agent's citations prove the claim. The `cited` entries are starting points only. Use `/workspace/PR.md` only as additional factual context, never as implementation evidence.

Return exactly one verdict for every listed field:
- `supported`: the cited source code directly supports every factual assertion in the field.
- `contradicted`: the frozen sources contradict at least one factual assertion.
- `unverifiable`: the frozen sources do not establish the field's factual assertions.
- `non_factual`: the field is empty or makes no factual assertion.

Every non-empty field marked `supported` must include at least one exact `head` or `base` code citation; PR-only citations do not qualify. Citations may also use `version` `pr`; for `pr`, use `path` `PR.md`. A `head` or `base` citation must quote exact text from that source file. A `pr` citation must quote exact text from `/workspace/PR.md`. Every quote must occur exactly once in its source. Do not invent or paraphrase quotes. Use at most 32 citations per field. Give a concise, factual reason for every verdict.

Do not edit repository files. Write the result to `/out/result.json`, run `node /skills/check_output.mjs`, correct any errors, and repeat until it prints `OK`. Reply `DONE` only.
