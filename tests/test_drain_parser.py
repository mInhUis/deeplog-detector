"""Unit tests for src/parser/drain_parser.py.

Uses a small, mocked dataset of exactly 5 CloudTrail-style log strings to
verify the core Drain algorithm logic without touching real data or
performing heavy I/O.

Mock Dataset (5 log strings):
    Log 0 — "ListBuckets from 10.0.0.1 by root"           (Template A)
    Log 1 — "ListBuckets from 192.168.1.100 by root"       (Template A)
    Log 2 — "ListBuckets from 172.16.0.50 by root"         (Template A)
    Log 3 — "DescribeInstances in us-east-1 by admin"      (Template B)
    Log 4 — "CreateUser backup for account 999888777666"    (Template C)

Expected behaviour:
    • Logs 0-2 share identical structure (only the IP differs) →
      fit_transform must assign them the *same* integer ID.
    • Logs 3 and 4 are structurally distinct from each other and from
      logs 0-2 → each must receive a *unique* ID.
    • The returned template dict must contain exactly 3 entries (K = 3).

Data shapes verified by these tests:
    log_keys  : list[int]       — length 5  (one ID per input message)
    templates : dict[int, str]  — length 3  (one entry per unique cluster)
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from src.parser.drain_parser import DrainParser, LogCluster, WILDCARD

# ======================================================================== #
#  Mock data                                                                #
# ======================================================================== #

MOCK_LOGS: Final[list[str]] = [
    # Template A — 3 logs, same structure, different IPs
    "ListBuckets from 10.0.0.1 by root",
    "ListBuckets from 192.168.1.100 by root",
    "ListBuckets from 172.16.0.50 by root",
    # Template B — unique action / structure
    "DescribeInstances in us-east-1 by admin",
    # Template C — unique action / structure
    "CreateUser backup for account 999888777666",
]


# ======================================================================== #
#  Fixtures                                                                 #
# ======================================================================== #

@pytest.fixture()
def parser() -> DrainParser:
    """Return a fresh DrainParser with default hyper-parameters."""
    return DrainParser(depth=4, sim_threshold=0.4, max_children=100)


@pytest.fixture()
def parsed_result(parser: DrainParser) -> tuple[list[int], dict[int, str]]:
    """Run fit_transform on the mock dataset and cache the result."""
    return parser.fit_transform(MOCK_LOGS)


# ======================================================================== #
#  Core contract: fit_transform output shapes                               #
# ======================================================================== #

class TestFitTransformContract:
    """Verify the fundamental output shapes and types of fit_transform.

    Expected output shapes:
        log_keys  : list[int]       of length N  (N = len(MOCK_LOGS) = 5)
        templates : dict[int, str]  of length K  (K = number of unique
                    clusters ≤ N)
    """

    def test_log_keys_length_equals_input(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """log_keys must have exactly one entry per input message."""
        keys, _ = parsed_result
        assert len(keys) == len(MOCK_LOGS)

    def test_log_keys_are_ints(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Every element in log_keys must be a plain int."""
        keys, _ = parsed_result
        for key in keys:
            assert isinstance(key, int)

    def test_templates_is_dict_of_int_to_str(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """templates must map int → str."""
        _, templates = parsed_result
        for k, v in templates.items():
            assert isinstance(k, int)
            assert isinstance(v, str)

    def test_exactly_three_unique_templates(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """The 5 mock logs should collapse to exactly 3 clusters (K = 3)."""
        _, templates = parsed_result
        assert len(templates) == 3

    def test_every_key_has_a_template(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Every ID in log_keys must appear in the templates dict."""
        keys, templates = parsed_result
        for key in keys:
            assert key in templates


# ======================================================================== #
#  Template grouping: identical templates → same ID                         #
# ======================================================================== #

class TestTemplateGrouping:
    """Assert that structurally identical logs receive the same cluster ID."""

    def test_three_identical_templates_share_id(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Logs 0, 1, 2 differ only in their IP address; their IDs must match."""
        keys, _ = parsed_result
        assert keys[0] == keys[1] == keys[2]

    def test_group_a_id_differs_from_group_b(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Template A (ListBuckets) must not share an ID with Template B."""
        keys, _ = parsed_result
        assert keys[0] != keys[3]

    def test_group_a_id_differs_from_group_c(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Template A (ListBuckets) must not share an ID with Template C."""
        keys, _ = parsed_result
        assert keys[0] != keys[4]

    def test_group_b_id_differs_from_group_c(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Template B and C are structurally different → distinct IDs."""
        keys, _ = parsed_result
        assert keys[3] != keys[4]

    def test_exactly_three_unique_ids_in_sequence(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """The full key sequence must contain exactly 3 distinct values."""
        keys, _ = parsed_result
        assert len(set(keys)) == 3


# ======================================================================== #
#  Template content & wildcard quality                                      #
# ======================================================================== #

class TestTemplateContent:
    """Verify that extracted templates contain correct wildcards."""

    def test_group_a_template_has_wildcard_for_ip(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """The IP position in Template A must be generalised to '<*>'."""
        keys, templates = parsed_result
        tpl: str = templates[keys[0]]
        assert WILDCARD in tpl

    def test_group_a_template_preserves_static_tokens(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Static tokens (ListBuckets, from, by, root) must survive."""
        keys, templates = parsed_result
        tpl: str = templates[keys[0]]
        for word in ("ListBuckets", "from", "by", "root"):
            assert word in tpl

    def test_group_b_template_preserves_action(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Template B must retain 'DescribeInstances'."""
        keys, templates = parsed_result
        tpl: str = templates[keys[3]]
        assert "DescribeInstances" in tpl

    def test_group_c_template_preserves_action(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Template C must retain 'CreateUser'."""
        keys, templates = parsed_result
        tpl: str = templates[keys[4]]
        assert "CreateUser" in tpl

    def test_group_c_template_wildcards_account_id(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """The 12-digit account ID in Template C must become '<*>'."""
        keys, templates = parsed_result
        tpl: str = templates[keys[4]]
        assert "999888777666" not in tpl
        assert WILDCARD in tpl


# ======================================================================== #
#  Reverse-mapping integrity (modularity requirement)                       #
# ======================================================================== #

class TestReverseMapping:
    """Ensure the template dict supports clean reverse-mapping to text.

    Per CLAUDE.md's modularity constraint, DeepLog's integer output must be
    cleanly reverse-mapped to text templates before the Llama-3 prompt stage.
    """

    def test_all_cluster_ids_are_mappable(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Every unique ID in log_keys must resolve to a non-empty string."""
        keys, templates = parsed_result
        for uid in set(keys):
            assert uid in templates
            assert len(templates[uid]) > 0

    def test_templates_are_human_readable(
        self, parsed_result: tuple[list[int], dict[int, str]]
    ) -> None:
        """Templates must contain at least one real word (not all wildcards)."""
        _, templates = parsed_result
        for tpl in templates.values():
            non_wildcard: list[str] = [
                t for t in tpl.split() if t != WILDCARD
            ]
            assert len(non_wildcard) > 0


# ======================================================================== #
#  Pre-processing unit tests                                                #
# ======================================================================== #

class TestPreprocess:
    """Verify regex-based variable masking on individual strings."""

    def test_ipv4_masked(self, parser: DrainParser) -> None:
        """IPv4 addresses must be replaced with '<*>'."""
        result: str = parser._preprocess("request from 10.0.0.1 ok")
        assert "10.0.0.1" not in result
        assert WILDCARD in result

    def test_uuid_masked(self, parser: DrainParser) -> None:
        """UUIDs must be replaced with '<*>'."""
        raw: str = "eventID 3038ebd2-c98a-4c65-9b6e-e22506292313 logged"
        result = parser._preprocess(raw)
        assert "3038ebd2" not in result

    def test_aws_access_key_masked(self, parser: DrainParser) -> None:
        """AWS access key IDs (AKIA...) must be masked."""
        result = parser._preprocess("key AKIA1ZBTOEKWKVHP6GHZ used")
        assert "AKIA1ZBTOEKWKVHP6GHZ" not in result

    def test_bare_integers_masked(self, parser: DrainParser) -> None:
        """Standalone integers must be replaced with '<*>'."""
        result = parser._preprocess("account 811596193553 region 1")
        assert "811596193553" not in result
        # "1" at the end should also be masked
        tokens: list[str] = result.split()
        assert all(t == WILDCARD or not t.isdigit() for t in tokens)

    def test_no_false_positive_on_words(self, parser: DrainParser) -> None:
        """Pure alphabetic words must never be masked."""
        result = parser._preprocess("ListBuckets from root")
        assert "ListBuckets" in result
        assert "from" in result
        assert "root" in result


# ======================================================================== #
#  Edge cases                                                               #
# ======================================================================== #

class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_empty_input_returns_empty(self, parser: DrainParser) -> None:
        """An empty list must produce empty outputs, not crash."""
        keys, templates = parser.fit_transform([])
        assert keys == []
        assert templates == {}

    def test_single_message(self, parser: DrainParser) -> None:
        """A single message must produce exactly one cluster."""
        keys, templates = parser.fit_transform(["solo event happened"])
        assert len(keys) == 1
        assert len(templates) == 1

    def test_all_identical_messages(self, parser: DrainParser) -> None:
        """N identical messages must all receive the same cluster ID."""
        msgs: list[str] = ["identical log line"] * 10
        keys, templates = parser.fit_transform(msgs)
        assert len(set(keys)) == 1
        assert len(templates) == 1

    def test_cluster_size_tracks_matches(self, parser: DrainParser) -> None:
        """LogCluster.size must reflect how many messages matched."""
        parser.fit_transform(MOCK_LOGS)
        # The cluster for Template A should have size == 3
        group_a_cluster: LogCluster = parser._clusters[0]
        assert group_a_cluster.size == 3

    def test_deterministic_across_runs(self) -> None:
        """Two fresh parsers on the same input must produce identical keys."""
        keys_a, _ = DrainParser().fit_transform(MOCK_LOGS)
        keys_b, _ = DrainParser().fit_transform(MOCK_LOGS)
        assert keys_a == keys_b
