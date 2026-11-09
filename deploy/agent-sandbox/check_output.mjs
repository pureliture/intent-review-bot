import fs from "node:fs";
import path from "node:path";

const workspace = path.resolve(process.env.REVIEW_BOT_WORKSPACE_DIR ?? "/workspace");
const output = path.resolve(process.env.REVIEW_BOT_OUTPUT_DIR ?? "/out");
const errors = [];

function pointer(parts) {
  return `/${parts.map((part) => String(part).replaceAll("~", "~0").replaceAll("/", "~1")).join("/")}`;
}

function error(parts, code, detail = "") {
  errors.push(`${pointer(parts)}: ${code}${detail ? ` ${detail}` : ""}`);
}

function readJson(file, parts, missingCode, invalidCode, maxBytes = Number.MAX_SAFE_INTEGER) {
  let text;
  try {
    const metadata = fs.lstatSync(file);
    if (!metadata.isFile()) {
      error(parts, "agent_output_file_invalid");
      return null;
    }
    if (metadata.size > maxBytes) {
      error(parts, "agent_output_too_large");
      return null;
    }
    text = fs.readFileSync(file, "utf8");
  } catch (cause) {
    error(parts, cause.code === "ENOENT" ? missingCode : invalidCode);
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (cause) {
    const message = String(cause.message).replaceAll(/\s+/g, " ");
    const match = message.match(/position\s+(\d+)/i);
    const offset = match ? Number(match[1]) : (/unexpected end|end of json input/i.test(message) ? text.length : 0);
    const context = text.slice(Math.max(0, offset - 20), Math.min(text.length, offset + 20));
    error(parts, "agent_output_json_invalid",
      `message=${JSON.stringify(message)} offset=${offset} context=${JSON.stringify(context)}`);
    return null;
  }
}

function object(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(value, keys, parts) {
  if (!object(value)) {
    error(parts, "agent_output_object_invalid");
    return false;
  }
  const expected = [...keys].sort();
  const actual = Object.keys(value).sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    error(parts, "agent_output_keys_invalid",
      `expected=${JSON.stringify(expected)} actual=${JSON.stringify(actual)}`);
    return false;
  }
  return true;
}

function nonemptyString(value, parts, allowEmpty = false) {
  if (typeof value !== "string" || (!allowEmpty && value.trim().length === 0)) {
    error(parts, "agent_output_string_invalid");
    return false;
  }
  return true;
}

function safeSourcePath(version, source) {
  if (typeof source !== "string" || source.length === 0 || source.includes("\0")
      || source.includes("\\") || path.posix.isAbsolute(source)
      || source.split("/").some((part) => part === "" || part === "." || part === "..")) {
    return null;
  }
  const versionRoot = path.resolve(workspace, version);
  const file = path.resolve(versionRoot, ...source.split("/"));
  if (file !== versionRoot && !file.startsWith(`${versionRoot}${path.sep}`)) {
    return null;
  }
  try {
    const realRoot = fs.realpathSync(versionRoot);
    const realFile = fs.realpathSync(file);
    if (realFile !== realRoot && !realFile.startsWith(`${realRoot}${path.sep}`)) {
      return null;
    }
    return realFile;
  } catch {
    return null;
  }
}

function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (!object(value)) return JSON.stringify(value);
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
}

function checkCitations(citations, parts, contract, allowEmpty) {
  const keys = contract.required_keys.citation;
  if (!Array.isArray(citations)) {
    error(parts, "agent_output_citations_invalid");
    return;
  }
  if (citations.length === 0 && !allowEmpty) {
    error(parts, "agent_output_citations_missing");
    return;
  }
  if (citations.length > contract.max_items.citations_per_item) {
    error(parts, "agent_output_citations_too_many");
    return;
  }
  citations.forEach((citation, index) => {
    const at = [...parts, index];
    if (!exactKeys(citation, keys, at)) return;
    if (!contract.citation_rules.versions.includes(citation.version)) {
      error([...at, "version"], "agent_output_citation_version_invalid");
      return;
    }
    if (typeof citation.path !== "string" || typeof citation.quote !== "string" || !citation.quote) {
      error(at, "agent_output_citation_shape_invalid");
      return;
    }
    const file = safeSourcePath(citation.version, citation.path);
    if (!file) {
      error([...at, "path"], "agent_output_citation_source_invalid");
      return;
    }
    let source;
    try {
      source = fs.readFileSync(file);
    } catch {
      error([...at, "path"], "agent_output_citation_source_missing");
      return;
    }
    const quote = Buffer.from(citation.quote, "utf8");
    const first = source.indexOf(quote);
    if (first < 0) {
      error([...at, "quote"], "agent_output_quote_absent");
    } else if (source.lastIndexOf(quote) !== first) {
      error([...at, "quote"], "agent_output_quote_ambiguous");
    }
  });
}

function hasCodeFiles() {
  const pending = ["head", "base"].map((version) => path.join(workspace, version));
  while (pending.length > 0) {
    const current = pending.pop();
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch (cause) {
      if (cause.code === "ENOENT") continue;
      throw cause;
    }
    for (const entry of entries) {
      if (entry.isFile()) return true;
      if (entry.isDirectory()) pending.push(path.join(current, entry.name));
    }
  }
  return false;
}

function checkSummaryCitations(result, contract) {
  if (!object(result) || !Array.isArray(result.summary_citations)
      || (result.summary_citations.length === 0 && hasCodeFiles())) {
    error(["summary_citations"], "agent_output_summary_citations_missing");
    return false;
  }
  checkCitations(result.summary_citations, ["summary_citations"], contract, !hasCodeFiles());
  return true;
}

function checkArchitecture(result, contract) {
  if (!checkSummaryCitations(result, contract)) return;
  if (!exactKeys(result, contract.required_keys.envelope, [])) return;
  nonemptyString(result.summary, ["summary"]);
  if (!Array.isArray(result.findings)) {
    error(["findings"], "agent_output_array_invalid");
    return;
  }
  if (result.findings.length > contract.max_items.findings) {
    error(["findings"], "agent_output_too_many_items");
  }
  const seen = new Set();
  result.findings.forEach((finding, index) => {
    const at = ["findings", index];
    if (!exactKeys(finding, contract.required_keys.finding, at)) return;
    for (const key of ["statement", "reasoning"]) nonemptyString(finding[key], [...at, key]);
    nonemptyString(finding.failure_condition, [...at, "failure_condition"], true);
    if (!contract.enums.verdict.includes(finding.verdict)) {
      error([...at, "verdict"], "agent_output_enum_invalid");
    }
    if (!contract.enums.basis.includes(finding.basis)) {
      error([...at, "basis"], "agent_output_enum_invalid");
    }
    if (!Array.isArray(finding.target_ids)) {
      error([...at, "target_ids"], "agent_output_targets_invalid");
    } else {
      if (finding.target_ids.length > contract.max_items.target_ids_per_finding
          || finding.target_ids.some((target) => !contract.allowed_target_ids.includes(target))
          || new Set(finding.target_ids).size !== finding.target_ids.length
          || (finding.target_ids.length === 0
            && finding.basis !== contract.citation_rules.empty_target_ids_basis)) {
        error([...at, "target_ids"], "agent_output_targets_invalid");
      }
    }
    checkCitations(finding.citations, [...at, "citations"], contract, false);
    const identity = stable({ statement: finding.statement, citations: finding.citations });
    if (seen.has(identity)) error(at, "agent_output_duplicate");
    seen.add(identity);
  });
}

function checkIntent(result, contract) {
  if (!checkSummaryCitations(result, contract)) return;
  if (!exactKeys(result, contract.required_keys.envelope, [])) return;
  nonemptyString(result.summary, ["summary"]);
  if (!Array.isArray(result.questions)) {
    error(["questions"], "agent_output_array_invalid");
  } else if (result.questions.length > contract.max_items.questions) {
    error(["questions"], "agent_output_too_many_items");
  } else {
    result.questions.forEach((question, index) => nonemptyString(question, ["questions", index]));
  }
  if (!Array.isArray(result.perspectives)) {
    error(["perspectives"], "agent_output_array_invalid");
    return;
  }
  if (result.perspectives.length > contract.max_items.perspectives) {
    error(["perspectives"], "agent_output_too_many_items");
  }
  const seenPerspectives = new Set();
  const seenItems = new Set();
  result.perspectives.forEach((perspective, index) => {
    const at = ["perspectives", index];
    if (!exactKeys(perspective, contract.required_keys.perspective, at)) return;
    if (!contract.allowed_perspective_ids.includes(perspective.perspective_id)
        || seenPerspectives.has(perspective.perspective_id)) {
      error([...at, "perspective_id"], "agent_output_perspective_invalid");
    }
    seenPerspectives.add(perspective.perspective_id);
    if (!Array.isArray(perspective.assessments)
        || (perspective.assessments.length === 0 && contract.citation_rules.empty_assessments_allowed !== true)
        || perspective.assessments.length > contract.max_items.assessments_per_perspective) {
      error([...at, "assessments"], "agent_output_assessments_invalid");
      return;
    }
    perspective.assessments.forEach((assessment, assessmentIndex) => {
      const itemAt = [...at, "assessments", assessmentIndex];
      if (!exactKeys(assessment, contract.required_keys.assessment, itemAt)) return;
      if (!contract.enums.verdict.includes(assessment.verdict)) {
        error([...itemAt, "verdict"], "agent_output_enum_invalid");
      }
      for (const key of ["statement", "reasoning"]) nonemptyString(assessment[key], [...itemAt, key]);
      nonemptyString(assessment.failure_condition, [...itemAt, "failure_condition"], true);
      const allowEmpty = assessment.verdict === contract.citation_rules.empty_citations_verdict;
      checkCitations(assessment.citations, [...itemAt, "citations"], contract, allowEmpty);
      const identity = stable({ statement: assessment.statement, citations: assessment.citations });
      if (seenItems.has(identity)) error(itemAt, "agent_output_duplicate");
      seenItems.add(identity);
    });
  });
}

function main() {
  let contract;
  try {
    contract = JSON.parse(fs.readFileSync(path.join(workspace, "contract.json"), "utf8"));
  } catch {
    error(["workspace", "contract.json"], "agent_contract_invalid");
    return;
  }
  if (!object(contract) || !["architecture", "intent"].includes(contract.mode)
      || !object(contract.required_keys) || !object(contract.enums)
      || !object(contract.citation_rules) || !object(contract.max_items)) {
    error(["workspace", "contract.json"], "agent_contract_invalid");
    return;
  }
  const result = readJson(path.join(output, "result.json"), [],
    "agent_output_missing", "agent_output_unreadable", contract.max_output_bytes);
  if (result !== null) {
    if (contract.mode === "architecture") checkArchitecture(result, contract);
    else checkIntent(result, contract);
  }
  if (errors.length === 0) {
    process.stdout.write("OK\n");
  } else {
    process.stdout.write(`${errors.join("\n")}\n`);
    process.exitCode = 1;
  }
}

main();
