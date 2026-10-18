"""Real Git manifests, rather than PR text or selector fallback, bound PMD/CPD eligibility."""
from unittest.mock import Mock

import pytest

from review_bot.adapters.git import GitRepositoryReader
from review_bot.application.selection import select_modes
from review_bot.application.snapshot import assemble_snapshot_bundle
from review_bot.ports import ModeSelectorPort
from tests.conftest import SyntheticGitRepo


@pytest.mark.parametrize("manifests,path,expected", [
    (("build.gradle",), "src/main/java/App.java", ("architecture", "intent")),
    (("build.gradle.kts",), "src/main/java/App.java", ("architecture", "intent")),
    (("pom.xml",), "src/main/java/App.java", ("intent",)),
    ((), "src/main/java/App.java", ("intent",)),
    (("gradlew",), "src/main/java/App.java", ("intent",)),
    (("build.gradle", "maven/pom.xml"), "maven/src/main/java/App.java", ("intent",)),
    (("pom.xml", "gradle/build.gradle.kts"), "gradle/src/main/java/App.java", ("architecture", "intent")),
    (("pom.xml", "build.gradle"), "src/main/java/App.java", ("intent",)),
    (("build.gradle",), "worker/app.py", ("intent",)),
    (("build.gradle",), "README.md", ("intent",)),
    (("build.gradle",), "build.gradle", ("intent",)),
])
@pytest.mark.parametrize("selector_mode", ["default", "all", "error"])
def test_only_java_in_gradle_modules_can_select_architecture(tmp_path, manifests, path, expected, selector_mode):
    repo = SyntheticGitRepo(tmp_path)
    for manifest in manifests:
        repo.write_file(manifest, "// Existing immutable build marker\n")
    repo.write_file(path, "class App {}\n" if path.endswith(".java") else "old\n")
    base = repo.commit("base")
    repo.write_file(path, "class App { int value = 1; }\n" if path.endswith(".java") else "new\n")
    head = repo.commit("head")
    selector = None
    if selector_mode != "default":
        selector = Mock(spec=ModeSelectorPort)
        if selector_mode == "error":
            selector.select_modes.side_effect = RuntimeError("bounded selector failure")
        else:
            selector.select_modes.side_effect = lambda snapshot, pr, modes: select_modes(snapshot.snapshot_id, modes)
    snapshot, pr, _, selection = assemble_snapshot_bundle(
        GitRepositoryReader(tmp_path), "org/repo", 1, 2, 3, base, head, "feature/change",
        title="Refactor architecture, pom.xml, build.gradle and App.java", body="Untrusted metadata cannot enable A.",
        available_modes=("architecture", "intent"), selector_uses_model=selector is not None, mode_selector=selector,
    )
    assert selection.selected_modes == expected
    assert selection.available_modes == ("architecture", "intent")
    assert selection.snapshot_id == snapshot.snapshot_id
    assert pr.changed_files == (path,)


@pytest.mark.parametrize("action,expected", [
    ("delete", ("architecture", "intent")),
    ("rename_out", ("architecture", "intent")),
    ("migrate_to_maven", ("intent",)),
])
def test_removed_java_uses_base_but_current_java_uses_head_build_system(tmp_path, action, expected):
    repo = SyntheticGitRepo(tmp_path)
    repo.write_file("build.gradle", "plugins { id 'java' }\n")
    repo.write_file("src/main/java/App.java", "class App {}\n")
    base = repo.commit("Gradle Java base")
    if action == "rename_out":
        repo._git("mv", "src/main/java/App.java", "README.md")
    elif action == "delete":
        repo.delete_file("src/main/java/App.java")
    else:
        repo.delete_file("build.gradle")
        repo.write_file("pom.xml", "<project/>\n")
        repo.write_file("src/main/java/App.java", "class App { int changed; }\n")
    head = repo.commit("change")
    *_, selection = assemble_snapshot_bundle(GitRepositoryReader(tmp_path), "org/repo", 1, 2, 3,
        base, head, "feature/change", available_modes=("architecture", "intent"))
    assert selection.selected_modes == expected


def test_unavailable_build_metadata_cannot_enable_architecture_or_suppress_intent(tmp_path):
    repo = SyntheticGitRepo(tmp_path)
    repo.write_file("build.gradle", "plugins { id 'java' }\n")
    repo.write_file("src/main/java/App.java", "class App {}\n")
    base = repo.commit("base")
    repo.write_file("src/main/java/App.java", "class App { int changed; }\n")
    head = repo.commit("head")
    reader = GitRepositoryReader(tmp_path)
    failing = Mock(wraps=reader)

    def read(sha, paths=None):
        if paths and "build.gradle" in paths:
            raise OSError("build metadata unavailable")
        return reader.read_tree_blobs(sha, paths)

    failing.read_tree_blobs.side_effect = read
    *_, selection = assemble_snapshot_bundle(failing, "org/repo", 1, 2, 3, base, head,
        "feature/change", available_modes=("architecture", "intent"))
    assert selection.selected_modes == ("intent",)
    assert "metadata-unavailable" in selection.policy_version
