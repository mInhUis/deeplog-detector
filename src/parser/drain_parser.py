"""Drain: An Online Log Parsing Approach with Fixed-Depth Tree.

Implementation based on:
    He, P., Zhu, J., Zheng, Z., & Lyu, M.R. (2017).
    "Drain: An Online Log Parsing Approach with Fixed Depth Tree."
    IEEE International Conference on Web Services (ICWS).

**Why Drain?**
Cloud environments generate millions of semi-structured log lines whose
text differs only in variable fields (IP addresses, resource IDs, request
IDs).  Drain uses a fixed-depth prefix tree to cluster these lines into
*templates* in a single pass — O(n) overall — making it ideal for the
massive CloudTrail payloads in this pipeline.

Data Flow (see Architecture & Workflow Decisions in CLAUDE.md):
    Raw log strings (list[str])
        → DrainParser.fit_transform()
        → (log_key_sequence: list[int], templates: dict[int, str])
        → DeepLog LSTM input

Input Shape:
    log_messages : list[str]  — N raw log event strings.

Output Shape:
    log_keys     : list[int]  — length-N integer sequence (one ID per
                                input message, in the same order).
    templates    : dict[int, str] — K entries  (K ≤ N unique clusters).
                   Maps each integer Log Key ID back to its human-readable
                   text template, enabling the reverse-mapping required by
                   the Llama-3 prompt stage.

Data-Structure Optimisation (per CLAUDE.md):
    • Pre-compiled regex patterns — compiled once, reused for every message.
    • Dict-based tree nodes — O(1) child lookup by token value at each depth.
    • Direct list-indexing for token comparison — avoids iterator overhead.
    • ``__slots__`` / ``dataclass(slots=True)`` on Node and LogCluster —
      minimises per-object memory overhead for trees with millions of nodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

# ======================================================================== #
#  Pre-compiled regex patterns for variable masking                         #
# ======================================================================== #
# Order: most specific first → avoids partial-match interference.
_VARIABLE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),                              # IPv4
    re.compile(
        r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    ),                                                                         # UUID
    re.compile(r"arn:aws:[a-zA-Z0-9\-]+::[0-9]+:[a-zA-Z0-9/\-_:]+"),         # AWS ARN
    re.compile(r"AKIA[0-9A-Z]{16}"),                                           # Access key
    re.compile(r"AIDA[0-9A-Z]{16}"),                                           # IAM user ID
    re.compile(r"AROA[0-9A-Z]{16}"),                                           # Role ID
    re.compile(r"ASIA[0-9A-Z]{16}"),                                           # Temp cred
    re.compile(r"\b[0-9a-fA-F]{12,}\b"),                                       # Long hex ID
    re.compile(r"\b0x[0-9a-fA-F]+\b"),                                         # Hex literal
    re.compile(r"\b\d+\.\d+\.\d+\b"),                                          # Version x.y.z
    re.compile(r"\b\d+\b"),                                                    # Remaining ints
)

WILDCARD: Final[str] = "<*>"


# ======================================================================== #
#  Data structures                                                          #
# ======================================================================== #

@dataclass(slots=True)
class LogCluster:
    """A cluster of log messages that share an identical template pattern.

    Attributes:
        template_tokens: Canonical template as a token list; variable
                         positions are replaced with ``'<*>'``.
                         Shape: list[str] of length T (token count).
        cluster_id:      Unique integer Log Key ID for this cluster.
        size:            Running count of messages matched to this cluster.
    """

    template_tokens: list[str]
    cluster_id: int
    size: int = 1

    @property
    def template_str(self) -> str:
        """Return the template as a single space-joined string."""
        return " ".join(self.template_tokens)


@dataclass(slots=True)
class _Node:
    """Internal tree node for the fixed-depth parse tree.

    Uses a :class:`dict` for O(1) child lookup by token value.

    Attributes:
        children: Token-string → child ``_Node`` mapping.
        clusters: ``LogCluster`` list stored **only** at leaf nodes.
    """

    children: dict[str, _Node] = field(default_factory=dict)
    clusters: list[LogCluster] = field(default_factory=list)


# ======================================================================== #
#  Drain Parser                                                             #
# ======================================================================== #

class DrainParser:
    """Fixed-Depth Tree parser for structured log template extraction.

    Converts raw log strings into integer *Log Key* IDs and a reverse-mapping
    dictionary of text templates, suitable for feeding into DeepLog's LSTM.

    Parameters:
        depth:          Number of tree levels before the leaf/cluster layer.
                        Higher values give finer-grained token routing but
                        increase memory usage.  Default ``4``.
        sim_threshold:  Minimum token-match ratio ``[0.0, 1.0]`` required to
                        assign a message to an existing cluster.  Lower values
                        yield fewer, more generalised templates.  Default
                        ``0.4``.
        max_children:   Maximum children per internal node.  When exceeded,
                        new tokens are routed to a ``<*>`` wildcard child,
                        preventing tree explosion on high-cardinality tokens.
                        Default ``100``.

    Example::

        >>> parser = DrainParser()
        >>> keys, tpls = parser.fit_transform([
        ...     "ListBuckets from 10.0.0.1",
        ...     "ListBuckets from 192.168.1.1",
        ...     "GetObject file.txt",
        ... ])
        >>> keys
        [0, 0, 1]
        >>> tpls
        {0: 'ListBuckets from <*>', 1: 'GetObject file.txt'}
    """

    __slots__ = (
        "depth",
        "sim_threshold",
        "max_children",
        "_root",
        "_clusters",
        "_next_id",
    )

    def __init__(
        self,
        depth: int = 4,
        sim_threshold: float = 0.4,
        max_children: int = 100,
    ) -> None:
        """Initialise the parser with tunable hyper-parameters.

        Args:
            depth:          Fixed tree depth (excluding the root length layer).
            sim_threshold:  Cluster similarity acceptance threshold.
            max_children:   Per-node child limit before wildcard fallback.
        """
        self.depth: int = depth
        self.sim_threshold: float = sim_threshold
        self.max_children: int = max_children

        # Root: dict keyed by token-count (message length) → first _Node
        self._root: dict[int, _Node] = {}
        self._clusters: list[LogCluster] = []
        self._next_id: int = 0

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    def fit_transform(
        self,
        log_messages: list[str],
    ) -> tuple[list[int], dict[int, str]]:
        """Parse log messages, extract templates, and return Log Key IDs.

        For each message the method:

        1. **Pre-processes** — masks known variable patterns (IPs, UUIDs,
           ARNs, numeric IDs) with ``'<*>'`` using pre-compiled regexes.
        2. **Tokenises** — splits on whitespace.
        3. **Tree-searches** — traverses the fixed-depth tree by token count
           then by token values at successive positions.
        4. **Cluster-matches** — compares with existing leaf clusters via
           token-level similarity; creates a new cluster if no match.

        Args:
            log_messages: Raw log strings, one per event.
                          Shape: ``list[str]`` of length **N**.

        Returns:
            A ``(log_keys, templates)`` tuple where:

            - ``log_keys``  — ``list[int]`` of length **N**.
              Ordered integer sequence; ``log_keys[i]`` is the cluster ID
              assigned to ``log_messages[i]``.
            - ``templates`` — ``dict[int, str]`` of length **K** (K ≤ N).
              Maps each cluster ID to its text template string.
        """
        log_keys: list[int] = []

        for message in log_messages:
            preprocessed: str = self._preprocess(message)
            tokens: list[str] = preprocessed.split()
            cluster: LogCluster = self._tree_search(tokens)
            log_keys.append(cluster.cluster_id)

        templates: dict[int, str] = {
            c.cluster_id: c.template_str for c in self._clusters
        }

        return log_keys, templates

    # ================================================================== #
    #  Pre-processing                                                      #
    # ================================================================== #

    @staticmethod
    def _preprocess(message: str) -> str:
        """Mask known variable patterns with the ``<*>`` wildcard.

        Pre-compiled regexes are applied in specificity order (IPv4 before
        bare integers) to avoid partial-match collisions.

        Args:
            message: A single raw log string.

        Returns:
            The message with all matched variables replaced by ``'<*>'``.
        """
        result: str = message
        for pattern in _VARIABLE_PATTERNS:
            result = pattern.sub(WILDCARD, result)
        return result

    # ================================================================== #
    #  Tree traversal                                                      #
    # ================================================================== #

    def _tree_search(self, tokens: list[str]) -> LogCluster:
        """Traverse the parse tree to find or create a matching cluster.

        Routing order:

        1. **Length layer** — ``_root[len(tokens)]`` (O(1) dict lookup).
        2. **Token layers** — up to ``depth − 1`` levels, each choosing
           a child by exact token match or ``<*>`` fallback.
        3. **Leaf** — scan resident clusters for similarity ≥ threshold.
        4. If no match, create a new ``LogCluster``.

        Args:
            tokens: Whitespace-split pre-processed token list.
                    Shape: ``list[str]`` of length T.

        Returns:
            The matched or newly created :class:`LogCluster`.
        """
        token_count: int = len(tokens)

        # --- Level 1: length bucket (O(1) lookup) ---
        if token_count not in self._root:
            self._root[token_count] = _Node()
        current_node: _Node = self._root[token_count]

        # --- Levels 2 .. depth: token-position routing ---
        # Traverse min(depth-1, token_count) levels so short messages
        # are never indexed beyond their actual length.
        traverse_depth: int = min(self.depth - 1, token_count)

        for i in range(traverse_depth):
            token: str = tokens[i]

            if token in current_node.children:
                # Exact token match — O(1) dict hit
                current_node = current_node.children[token]
            elif WILDCARD in current_node.children:
                # Wildcard fallback
                current_node = current_node.children[WILDCARD]
            else:
                # No match — create a new child node
                if len(current_node.children) < self.max_children:
                    new_node = _Node()
                    current_node.children[token] = new_node
                    current_node = new_node
                else:
                    # Node is at capacity → route through wildcard bucket
                    if WILDCARD not in current_node.children:
                        current_node.children[WILDCARD] = _Node()
                    current_node = current_node.children[WILDCARD]

        # --- Leaf layer: cluster matching ---
        return self._match_or_create(current_node, tokens)

    # ================================================================== #
    #  Cluster matching & template update                                  #
    # ================================================================== #

    def _match_or_create(
        self,
        leaf: _Node,
        tokens: list[str],
    ) -> LogCluster:
        """Find the best-matching cluster at a leaf, or create a new one.

        Iterates over existing clusters in the leaf, computes token-level
        similarity, and accepts the best match if it exceeds
        ``sim_threshold``.  On acceptance the template is generalised
        in-place.

        Args:
            leaf:   Leaf ``_Node`` reached by tree traversal.
            tokens: Pre-processed token list for the current message.
                    Shape: ``list[str]`` of length T.

        Returns:
            The matched or newly created :class:`LogCluster`.
        """
        best_cluster: LogCluster | None = None
        best_sim: float = -1.0

        for cluster in leaf.clusters:
            sim: float = self._compute_similarity(
                cluster.template_tokens, tokens
            )
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None and best_sim >= self.sim_threshold:
            # Generalise the template where tokens differ
            self._update_template(best_cluster, tokens)
            best_cluster.size += 1
            return best_cluster

        # No satisfactory match — mint a new cluster
        new_cluster = LogCluster(
            template_tokens=list(tokens),   # copy to avoid aliasing
            cluster_id=self._next_id,
        )
        self._next_id += 1
        leaf.clusters.append(new_cluster)
        self._clusters.append(new_cluster)
        return new_cluster

    @staticmethod
    def _compute_similarity(
        template: list[str],
        tokens: list[str],
    ) -> float:
        """Compute token-level similarity between a template and new tokens.

        Similarity = (number of exactly matching non-wildcard positions)
                     / (total token count).

        Both wildcard-to-wildcard and wildcard-to-value matches are
        counted; only true mismatches reduce the score.

        Args:
            template: Existing cluster template (may contain ``'<*>'``).
                      Shape: ``list[str]`` of length T.
            tokens:   New pre-processed log tokens.
                      Shape: ``list[str]`` of length T.

        Returns:
            ``float`` in ``[0.0, 1.0]``.  Returns ``0.0`` when lengths
            differ (messages of different token counts are never similar
            by definition in Drain).
        """
        if len(template) != len(tokens):
            return 0.0

        total: int = len(template)
        if total == 0:
            return 0.0

        match_count: int = sum(
            1 for t, m in zip(template, tokens) if t == m
        )
        return match_count / total

    @staticmethod
    def _update_template(cluster: LogCluster, tokens: list[str]) -> None:
        """Generalise a cluster's template by replacing divergent positions.

        Any position where the existing template token and the new token
        disagree is replaced with ``'<*>'``, monotonically increasing
        the template's generality.

        Args:
            cluster: The matched cluster — mutated **in-place**.
            tokens:  New pre-processed log tokens.
        """
        tpl: list[str] = cluster.template_tokens
        for i, (t_tok, m_tok) in enumerate(zip(tpl, tokens)):
            if t_tok != m_tok:
                tpl[i] = WILDCARD
