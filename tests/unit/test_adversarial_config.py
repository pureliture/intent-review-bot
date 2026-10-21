"""Adversarial and empirical stress test suite for review_bot.config.

Covers:
- Billion Laughs / YAML bomb attacks (anchor expansion DoS)
- Circular and undefined aliases
- Deep nesting and recursion depth attacks
- Duplicate keys across diverse structures, comments, quoting, and formatting
- Untrusted target repository security restrictions (.review-bot.yml)
- Security invariants (target build scripts, public CI leak, github_app allowlists)
- Schema validation, range constraints, and environment variable regex
- Unicode quirks, BOM, null bytes, surrogates, and Korean text
- Canonical SHA-256 hash determinism and sensitivity
"""
from __future__ import annotations

import io

import pytest
import yaml

from review_bot.config import (
    ConfigAliasError,
    ConfigDuplicateKeyError,
    ConfigSecurityError,
    ConfigSyntaxError,
    ConfigValidationError,
    load_config,
    load_review_config_from_str,
    load_target_repository_config,
    parse_yaml_strict,
)

EXPECTED_CANONICAL_HASH = (
    "e95b9a13fe79183e7c9ca1ceea0db96338c4561749904858222ca791b245cd71"
)


# ==============================================================================
# Category 1: YAML Bomb / Billion Laughs & Anchor Expansion
# ==============================================================================


class TestYamlBombAndAnchors:
    """Tests designed to exploit exponential entity expansion (Billion Laughs) and anchors."""

    def test_billion_laughs_sequence_bomb(self) -> None:
        bomb = """
a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d]
root: *e
"""
        with pytest.raises(ConfigAliasError) as exc_info:
            parse_yaml_strict(bomb)
        assert "YAML anchors are forbidden" in str(exc_info.value)

    def test_billion_laughs_mapping_bomb(self) -> None:
        bomb = """
a: &a {k1: lol, k2: lol}
b: &b {k1: *a, k2: *a}
c: &c {k1: *b, k2: *b}
d: &d {k1: *c, k2: *c}
root: *d
"""
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(bomb)

    def test_flow_style_anchor_and_alias(self) -> None:
        flow = "{ a: &a 1, b: *a }"
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(flow)

    def test_flow_sequence_anchor(self) -> None:
        flow = "[ &a 1, 2, *a ]"
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(flow)

    def test_merge_key_alias(self) -> None:
        merge_yaml = """
base: &base
  schema_version: 2
derived:
  <<: *base
  mode: offline
"""
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(merge_yaml)

    def test_circular_alias_self_referential(self) -> None:
        circ = "&a [*a]"
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(circ)

    def test_circular_alias_mutual(self) -> None:
        circ = """
a: &a { b: *b }
b: &b { a: *a }
"""
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(circ)

    def test_forward_undefined_alias(self) -> None:
        undef = "key: *nonexistent_anchor\n"
        with pytest.raises(ConfigAliasError) as exc_info:
            parse_yaml_strict(undef)
        assert "YAML aliases are forbidden" in str(exc_info.value)

    def test_anchor_on_mapping_key(self) -> None:
        key_anchor = "? &key_anchor my_key : value\n"
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(key_anchor)

    def test_anchor_on_document_root(self) -> None:
        root_anchor = """&root_anchor
schema_version: 2
codebase:
  product: test
"""
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(root_anchor)

    def test_anchor_on_list_item(self) -> None:
        item_anchor = """
items:
  - &item 123
  - 456
"""
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(item_anchor)

    def test_long_anchor_name(self) -> None:
        long_name = "a" * 1000
        payload = f"key: &{long_name} value\n"
        with pytest.raises(ConfigAliasError):
            parse_yaml_strict(payload)


# ==============================================================================
# Category 2: Deep Nesting, Quadratic Growth & Resource Exhaustion
# ==============================================================================


class TestDeepNestingAndResourceExhaustion:
    """Tests designed to test deep nesting, recursion limits, and DoS resilience."""

    def test_deeply_nested_flow_mapping_exceeding_recursion(self) -> None:
        # Nesting 2,000 deep exceeds Python's default recursion limit (1000)
        deep_yaml = '{"k": ' * 2000 + "1" + "}" * 2000
        with pytest.raises(ConfigSyntaxError) as exc_info:
            parse_yaml_strict(deep_yaml)
        assert "recursion depth exceeded" in str(exc_info.value).lower()

    def test_deeply_nested_flow_sequence_exceeding_recursion(self) -> None:
        deep_seq = "[" * 2000 + "1" + "]" * 2000
        with pytest.raises(ConfigSyntaxError) as exc_info:
            parse_yaml_strict(deep_seq)
        assert "recursion depth exceeded" in str(exc_info.value).lower()

    def test_large_number_of_forbidden_keys(self, valid_config_yaml: str) -> None:
        # Generate 1,000 extra unknown keys
        extra_keys = "\n".join(f"unexpected_key_{i}: {i}" for i in range(1000))
        payload = f"{valid_config_yaml}\n{extra_keys}\n"
        with pytest.raises(ConfigValidationError) as exc_info:
            load_review_config_from_str(payload)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_extreme_integer_value_no_crash(self, valid_config_yaml: str) -> None:
        # 100-digit integer should be parsed cleanly as int or validated
        huge_int = "9" * 100
        payload = valid_config_yaml.replace("cpd_minimum_tokens: 100", f"cpd_minimum_tokens: {huge_int}")
        cfg, _ = load_review_config_from_str(payload)
        assert cfg.architecture.cpd_minimum_tokens == int(huge_int)


# ==============================================================================
# Category 3: Duplicate Key Permutations & Quirks
# ==============================================================================


class TestDuplicateKeysAcrossStructures:
    """Tests duplicate key detection across comments, whitespace, quoting, types, and nesting."""

    def test_duplicate_keys_separated_by_comments_and_newlines(self) -> None:
        payload = """
schema_version: 2
# Multi-line comment block
# Line 2 of comment
# Line 3 of comment

schema_version: 2
"""
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key 'schema_version' found at line 7" in str(exc_info.value)

    def test_duplicate_keys_in_deeply_nested_dictionary(self) -> None:
        payload = """
architecture:
  isolation:
    cpus: 2
    memory_mib: 2048
    # comment
    cpus: 4
"""
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key 'cpus' found at line 7" in str(exc_info.value)

    def test_duplicate_keys_in_flow_mapping(self) -> None:
        payload = "codebase: {product: review_bot, product: duplicate_bot}"
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key 'product'" in str(exc_info.value)

    def test_duplicate_keys_in_sequence_of_mappings(self) -> None:
        payload = """
items:
  - id: 1
    id: 2
"""
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key 'id'" in str(exc_info.value)

    def test_duplicate_unquoted_vs_double_quoted(self) -> None:
        payload = """
product: alpha
"product": beta
"""
        with pytest.raises(ConfigDuplicateKeyError):
            parse_yaml_strict(payload)

    def test_duplicate_single_quoted_vs_double_quoted(self) -> None:
        payload = """
'product': alpha
"product": beta
"""
        with pytest.raises(ConfigDuplicateKeyError):
            parse_yaml_strict(payload)

    def test_duplicate_with_hex_escape_sequence(self) -> None:
        # \x5f is underscore '_'
        payload = """
"schema\\x5fversion": 2
schema_version: 2
"""
        with pytest.raises(ConfigDuplicateKeyError):
            parse_yaml_strict(payload)

    def test_duplicate_with_unicode_escape_sequence(self) -> None:
        # \u005f is underscore '_'
        payload = """
"schema\\u005fversion": 2
schema_version: 2
"""
        with pytest.raises(ConfigDuplicateKeyError):
            parse_yaml_strict(payload)

    def test_duplicate_stripped_folded_key(self) -> None:
        payload = """
? >-
  product
: 1
product: 2
"""
        with pytest.raises(ConfigDuplicateKeyError):
            parse_yaml_strict(payload)

    def test_duplicate_boolean_synonyms_yaml11(self) -> None:
        # In YAML 1.1, 'yes' and 'true' both evaluate to boolean True
        payload = """
yes: first
true: second
"""
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key 'True'" in str(exc_info.value)

    def test_duplicate_false_synonyms_yaml11(self) -> None:
        payload = """
no: 1
false: 2
"""
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key 'False'" in str(exc_info.value)

    def test_duplicate_null_synonyms(self) -> None:
        payload = """
null: 1
~: 2
"""
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key 'None'" in str(exc_info.value)

    def test_duplicate_zero_integers(self) -> None:
        payload = """
0: first
0: second
"""
        with pytest.raises(ConfigDuplicateKeyError) as exc_info:
            parse_yaml_strict(payload)
        assert "Duplicate key '0'" in str(exc_info.value)

    def test_unhashable_complex_key(self) -> None:
        payload = "? [1, 2] : sequence_key_value\n"
        with pytest.raises(ConfigSyntaxError) as exc_info:
            parse_yaml_strict(payload)
        assert "Found unhashable key" in str(exc_info.value)


# ==============================================================================
# Category 4: Untrusted Target Repository Security (.review-bot.yml)
# ==============================================================================


class TestTargetRepositorySecurity:
    """Tests security boundaries of target repo configuration load_target_repository_config."""

    def test_forbid_run_target_build_scripts_override(self) -> None:
        payload = "run_target_build_scripts: true\n"
        with pytest.raises(ConfigValidationError) as exc_info:
            load_target_repository_config(payload)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_forbid_publish_private_data_override(self) -> None:
        payload = "publish_private_data_to_public_ci: true\n"
        with pytest.raises(ConfigValidationError) as exc_info:
            load_target_repository_config(payload)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_forbid_app_id_env_override(self) -> None:
        payload = "app_id_env: MALICIOUS_ENV\n"
        with pytest.raises(ConfigValidationError):
            load_target_repository_config(payload)

    def test_forbid_service_block(self) -> None:
        payload = "service:\n  receiver: fastapi_uvicorn\n"
        with pytest.raises(ConfigValidationError):
            load_target_repository_config(payload)

    def test_forbid_isolation_runtime_override(self) -> None:
        payload = "isolation:\n  runtime: host\n"
        with pytest.raises(ConfigValidationError):
            load_target_repository_config(payload)

    def test_forbid_model_profile_override(self) -> None:
        payload = "model:\n  profile: untrusted-model\n"
        with pytest.raises(ConfigValidationError):
            load_target_repository_config(payload)

    def test_forbid_execution_mode_override(self) -> None:
        payload = "execution:\n  mode: github_app\n"
        with pytest.raises(ConfigValidationError):
            load_target_repository_config(payload)

    def test_forbid_tool_version_and_digest(self) -> None:
        payload = "architecture:\n  tool_version: '999.0'\n"
        with pytest.raises(ConfigValidationError):
            load_target_repository_config(payload)

    def test_target_repo_yaml_bomb_rejected(self) -> None:
        bomb = "a: &a ['lol']\nb: [*a, *a]\n"
        with pytest.raises(ConfigAliasError):
            load_target_repository_config(bomb)

    def test_target_repo_duplicate_keys_rejected(self) -> None:
        dup = "analysis_profile: p1\nanalysis_profile: p2\n"
        with pytest.raises(ConfigDuplicateKeyError):
            load_target_repository_config(dup)

    def test_target_repo_non_mapping_root_rejected(self) -> None:
        with pytest.raises(ConfigSyntaxError):
            load_target_repository_config("- list_item_1\n- list_item_2\n")

    def test_target_repo_invalid_type_source_roots(self) -> None:
        # String instead of list
        with pytest.raises(ConfigValidationError):
            load_target_repository_config("source_roots: 'src/main/java'\n")

    def test_target_repo_invalid_element_type_in_source_roots(self) -> None:
        # Integers instead of strings
        with pytest.raises(ConfigValidationError):
            load_target_repository_config("source_roots: [123, 456]\n")

    def test_target_repo_invalid_type_analysis_profile(self) -> None:
        with pytest.raises(ConfigValidationError):
            load_target_repository_config("analysis_profile: 12345\n")

    def test_target_repo_valid_minimal(self) -> None:
        valid = """
analysis_profile: java21-jdk-only-v1
source_roots:
  - src/main/java
architecture_rule_paths:
  - rules/arch.xml
"""
        cfg = load_target_repository_config(valid)
        assert cfg.analysis_profile == "java21-jdk-only-v1"
        assert cfg.source_roots == ["src/main/java"]
        assert cfg.architecture_rule_paths == ["rules/arch.xml"]

    def test_target_repo_empty_defaults(self) -> None:
        cfg = load_target_repository_config("")
        assert cfg.analysis_profile == "java21-jdk-only-v1"
        assert cfg.source_roots == ["src/main/java"]
        assert cfg.architecture_rule_paths == []


# ==============================================================================
# Category 5: Operator Security Invariants & Schema Constraints
# ==============================================================================


class TestOperatorSecurityInvariantsAndSchema:
    """Tests security invariants in operator configuration."""

    def test_reject_run_target_build_scripts_true(self, valid_config_yaml: str) -> None:
        exploit = valid_config_yaml.replace(
            "run_target_build_scripts: false", "run_target_build_scripts: true"
        )
        with pytest.raises(ConfigSecurityError) as exc_info:
            load_review_config_from_str(exploit)
        assert "run_target_build_scripts MUST be false for security" in str(exc_info.value)

    def test_reject_publish_private_data_true(self, valid_config_yaml: str) -> None:
        exploit = valid_config_yaml.replace(
            "publish_private_data_to_public_ci: false",
            "publish_private_data_to_public_ci: true",
        )
        with pytest.raises(ConfigSecurityError) as exc_info:
            load_review_config_from_str(exploit)
        assert "publish_private_data_to_public_ci MUST be false for security" in str(
            exc_info.value
        )

    @pytest.mark.parametrize(
        "bad_env_var",
        [
            "lower_case_env",
            "ENV-WITH-HYPHEN",
            "123_STARTS_WITH_DIGIT",
            "ENV WITH SPACES",
            "ENV;CMD_INJECTION",
            "$DOLLAR_VAR",
            "",
        ],
    )
    def test_reject_invalid_env_var_patterns(
        self, valid_config_yaml: str, bad_env_var: str
    ) -> None:
        mutated = valid_config_yaml.replace(
            "app_id_env: REVIEW_BOT_APP_ID", f'app_id_env: "{bad_env_var}"'
        )
        with pytest.raises(ConfigValidationError):
            load_review_config_from_str(mutated)

    @pytest.mark.parametrize(
        "field_replacement",
        [
            ("worker_concurrency: 1", "worker_concurrency: 0"),
            ("task_timeout_seconds: 900", "task_timeout_seconds: 0"),
            ("webhook_max_bytes: 10485760", "webhook_max_bytes: 0"),
            ("cpd_minimum_tokens: 100", "cpd_minimum_tokens: 9"),
            ("tool_timeout_seconds: 120", "tool_timeout_seconds: 0"),
            ("memory_mib: 2048", "memory_mib: 127"),
            ("pids: 128", "pids: 15"),
            ("max_findings_per_task: 20", "max_findings_per_task: 0"),
            ("max_findings_per_task: 20", "max_findings_per_task: 101"),
            ("max_questions_per_task: 20", "max_questions_per_task: 0"),
            ("max_questions_per_task: 20", "max_questions_per_task: 101"),
            ("max_comment_characters: 60000", "max_comment_characters: 999"),
            ("max_comment_characters: 60000", "max_comment_characters: 100001"),
        ],
    )
    def test_reject_range_violations(
        self, valid_config_yaml: str, field_replacement: tuple[str, str]
    ) -> None:
        old_val, new_val = field_replacement
        mutated = valid_config_yaml.replace(old_val, new_val)
        with pytest.raises(ConfigValidationError):
            load_review_config_from_str(mutated)

    def test_reject_github_app_mode_without_allowlists(
        self, valid_config_yaml: str
    ) -> None:
        # D1 replaces the historical mandatory-list rule with installation eligibility.
        gh_app_yaml = valid_config_yaml.replace("mode: offline", "mode: github_app")
        config, _ = load_review_config_from_str(gh_app_yaml)
        assert config.service.installation_allowlist == []
        assert config.service.repository_allowlist == []


# ==============================================================================
# Category 6: Unicode, Encodings & Formatting Quirks
# ==============================================================================


class TestUnicodeAndFormattingQuirks:
    """Tests UTF-8 BOM, Korean strings, null bytes, surrogates, and multi-documents."""

    def test_utf8_bom_prefix_handled_gracefully(self, valid_config_yaml: str) -> None:
        bom_yaml = f"\ufeff{valid_config_yaml}"
        cfg, cfg_hash = load_review_config_from_str(bom_yaml)
        assert cfg.schema_version == 2
        assert cfg_hash == EXPECTED_CANONICAL_HASH

    def test_null_byte_in_yaml_rejected(self) -> None:
        payload = "schema_version: 2\0\ncodebase: {}"
        with pytest.raises(ConfigSyntaxError) as exc_info:
            parse_yaml_strict(payload)
        assert "special characters are not allowed" in str(exc_info.value)

    def test_surrogate_character_in_yaml_rejected(self) -> None:
        payload = 'product: "\ud800"'
        with pytest.raises(ConfigSyntaxError):
            parse_yaml_strict(payload)

    def test_korean_text_in_comments_and_values(self, valid_config_yaml: str) -> None:
        korean_yaml = (
            "# 한국어 설정 주석\n"
            + valid_config_yaml.replace(
                "<!-- review-bot:architecture:v1 -->",
                "<!-- 리뷰봇:아키텍처:v1 -->",
            )
        )
        cfg, _ = load_review_config_from_str(korean_yaml)
        assert cfg.publish.a_marker == "<!-- 리뷰봇:아키텍처:v1 -->"

    def test_multi_document_stream_rejected(self) -> None:
        multi_doc = "schema_version: 2\n---\nschema_version: 2\n"
        with pytest.raises(ConfigSyntaxError) as exc_info:
            parse_yaml_strict(multi_doc)
        assert "expected a single document" in str(exc_info.value)

    def test_empty_string_rejected_by_load_review_config_from_str(self) -> None:
        with pytest.raises(ConfigValidationError):
            load_review_config_from_str("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            load_config("   \n\t  \n  ")

    def test_comment_only_file_rejected(self) -> None:
        with pytest.raises(ConfigValidationError):
            load_review_config_from_str("# Just a comment\n# Another comment\n")


# ==============================================================================
# Category 7: Canonical Hash Invariance & Sensitivity
# ==============================================================================


class TestCanonicalHashInvarianceAndSensitivity:
    """Verifies that compute_config_hash is deterministic, permutation-invariant, and sensitive."""

    def test_key_reordering_produces_identical_hash(self, valid_config_yaml: str) -> None:
        cfg1, hash1 = load_review_config_from_str(valid_config_yaml)

        # Parse to dict, reverse all keys, re-dump
        raw_dict = yaml.safe_load(valid_config_yaml)
        reversed_dict = {k: raw_dict[k] for k in reversed(list(raw_dict.keys()))}
        reversed_yaml = yaml.dump(reversed_dict)

        cfg2, hash2 = load_review_config_from_str(reversed_yaml)
        assert hash1 == hash2 == EXPECTED_CANONICAL_HASH
        assert cfg1.config_hash == cfg2.config_hash == EXPECTED_CANONICAL_HASH

    def test_hash_sensitivity_on_single_field_change(self, valid_config_yaml: str) -> None:
        _, hash_orig = load_review_config_from_str(valid_config_yaml)

        # Mutate retention days
        mutated = valid_config_yaml.replace("retention_days: 30", "retention_days: 31")
        _, hash_mut = load_review_config_from_str(mutated)
        assert hash_orig != hash_mut
        assert len(hash_mut) == 64

    def test_convenience_loader_with_string_and_stream(self, valid_config_yaml: str) -> None:
        cfg_from_str = load_config(valid_config_yaml)
        assert cfg_from_str.schema_version == 2

        stream = io.StringIO(valid_config_yaml)
        cfg_from_stream = load_config(stream)
        assert cfg_from_stream.schema_version == 2
        assert cfg_from_str.config_hash == cfg_from_stream.config_hash
