"""Real immutable manifests bound A targets in a mixed-build PR."""
from review_bot.adapters.git import GitRepositoryReader
from review_bot.application.delta import compute_delta
from review_bot.application.partition_review import prepare_partition_architecture_task, prepare_originals
from review_bot.application.snapshot import assemble_snapshot_bundle
from review_bot.domain.models_v1 import CpdDuplication, CpdOccurrence
from tests.conftest import SyntheticGitRepo

import pytest


def _snapshot(root, repo, base):
    head = repo.commit("synthetic head")
    reader = GitRepositoryReader(root)
    snapshot, pr, classification, selection = assemble_snapshot_bundle(reader, "synthetic/mixed", 1, 2, 3,
        base, head, "feature/change", available_modes=("architecture", "intent"))
    return reader, pr, classification, selection


def _delta(pr, paths, *, base=False):
    findings = [{"rule": "CognitiveComplexity", "path": path, "metric": 20, "method": "run"} for path in paths]
    return compute_delta(base_findings=findings if base else [], head_findings=[] if base else findings,
        base_sha=pr.base_sha, head_sha=pr.head_sha)


def test_one_gradle_change_does_not_enable_a_targets_for_nearby_maven_changes(tmp_path):
    repo = SyntheticGitRepo(tmp_path)
    repo.write_file("build.gradle", "plugins { id 'java' }\n")
    repo.write_file("maven/pom.xml", "<project/>\n")
    paths = ("src/main/java/Gradle.java", "maven/src/main/java/Maven.java")
    for path in paths:
        repo.write_file(path, "class Example {}\n")
    base = repo.commit("synthetic base")
    for path in paths:
        repo.write_file(path, "class Example { void run() {} }\n")
    reader, pr, classification, selection = _snapshot(tmp_path, repo, base)
    assert selection.selected_modes == ("architecture", "intent")
    delta = _delta(pr, paths)
    task = prepare_partition_architecture_task(reader, pr, delta, classification)
    assert [s.paths for s in task.delta.signals] == [(paths[0],)]
    assert len(delta.signals) == 2  # the original analysis is retained
    assert "partition_evaluation_excludes_non_gradle_analysis_signals:1" in task.limitations
    assert paths[1] in {f.path for f in task.snapshot.head.files}  # original PR context is not erased
    _, meta = prepare_originals(task, "architecture")
    assert list(meta["target_paths"].values()) == [[paths[0]]]


@pytest.mark.parametrize("maven_partner", [True, False])
def test_cpd_requires_eligible_ownership_for_every_partner_and_retains_originals(tmp_path, maven_partner):
    repo = SyntheticGitRepo(tmp_path)
    repo.write_file("build.gradle.kts", "plugins { java }\n")
    first = "src/main/java/Changed.java"
    partner = "maven/src/main/java/Partner.java" if maven_partner else "src/main/java/Partner.java"
    if maven_partner:
        repo.write_file("maven/pom.xml", "<project/>\n")
    for path in (first, partner):
        repo.write_file(path, "class Example {}\n")
    base = repo.commit("synthetic base")
    repo.write_file(first, "class Example { void run() {} }\n")
    reader, pr, classification, _ = _snapshot(tmp_path, repo, base)
    clone = CpdDuplication(1, 100, tuple(CpdOccurrence(p, p, 1, 1) for p in (first, partner)), "class Example")
    delta = compute_delta(base_findings=[], head_findings=[clone], base_sha=pr.base_sha, head_sha=pr.head_sha)
    assert len(delta.signals) == 1
    task = prepare_partition_architecture_task(reader, pr, delta, classification)
    assert len(task.delta.signals) == (0 if maven_partner else 1)
    assert partner in {f.path for f in task.snapshot.head.files}
    assert len(delta.signals) == 1
    if not maven_partner:
        _, meta = prepare_originals(task, "architecture")
        assert set(next(iter(meta["target_paths"].values()))) == {first, partner}


def test_current_java_uses_head_owner_even_when_base_was_gradle_and_another_module_still_enables_a(tmp_path):
    repo = SyntheticGitRepo(tmp_path)
    repo.write_file("build.gradle", "plugins { id 'java' }\n")
    repo.write_file("other/build.gradle", "plugins { id 'java' }\n")
    paths = ("src/main/java/Migrated.java", "other/src/main/java/StillGradle.java")
    for path in paths:
        repo.write_file(path, "class Example {}\n")
    base = repo.commit("synthetic Gradle base")
    repo.write_file("pom.xml", "<project/>\n")
    for path in paths:
        repo.write_file(path, "class Example { void run() {} }\n")
    reader, pr, classification, selection = _snapshot(tmp_path, repo, base)
    assert selection.selected_modes == ("architecture", "intent")
    task = prepare_partition_architecture_task(reader, pr, _delta(pr, paths), classification)
    assert [s.paths for s in task.delta.signals] == [(paths[1],)]


def test_removed_java_uses_base_gradle_owner_when_head_replaces_build_system(tmp_path):
    repo = SyntheticGitRepo(tmp_path)
    repo.write_file("build.gradle", "plugins { id 'java' }\n")
    path = "src/main/java/Removed.java"
    repo.write_file(path, "class Removed {}\n")
    base = repo.commit("synthetic Gradle base")
    repo.delete_file(path)
    repo.delete_file("build.gradle")
    repo.write_file("pom.xml", "<project/>\n")
    reader, pr, classification, selection = _snapshot(tmp_path, repo, base)
    assert selection.selected_modes == ("architecture", "intent")
    task = prepare_partition_architecture_task(reader, pr, _delta(pr, (path,), base=True), classification)
    assert [s.paths for s in task.delta.signals] == [(path,)]


def test_plain_java_without_a_gradle_owner_has_no_a_targets_even_for_operator_preparation(tmp_path):
    repo = SyntheticGitRepo(tmp_path)
    repo.write_file("pom.xml", "<project/>\n")
    path = "src/main/java/Maven.java"
    repo.write_file(path, "class Example {}\n")
    base = repo.commit("synthetic Maven base")
    repo.write_file(path, "class Example { void run() {} }\n")
    reader, pr, classification, selection = _snapshot(tmp_path, repo, base)
    assert selection.selected_modes == ("intent",)
    task = prepare_partition_architecture_task(reader, pr, _delta(pr, (path,)), classification)
    assert not task.delta.signals
    assert not prepare_originals(task, "architecture")[1]["targets"]
