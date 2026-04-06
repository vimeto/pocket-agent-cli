"""Simulated web search tool for HotpotQA benchmark.

Returns pre-cached HotpotQA context paragraphs ranked by simple TF-IDF
keyword matching. Configurable network latency via NetworkSimulator.
"""

import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from ..network.network_simulator import NetworkConfig, NetworkSimulator


class SimulatedWebSearch:
    """Simulated web search that returns pre-cached HotpotQA context.

    For each question, the supporting paragraphs are known. When the agent
    calls web_search(query), we return the most relevant paragraphs using
    simple TF-IDF keyword matching.

    Configurable network latency via NetworkSimulator.

    Attributes:
        paragraphs: All context paragraphs for this question.
        network_sim: Optional network simulator for latency.
        search_count: Number of searches performed.
        search_log: Detailed log of each search call.
    """

    def __init__(
        self,
        context_paragraphs: List[Dict[str, str]],
        network_config: Optional[NetworkConfig] = None,
        seed: int = 42,
    ):
        """Initialize the simulated web search.

        Args:
            context_paragraphs: List of paragraph dicts with 'title' and 'text'.
            network_config: Optional network configuration for latency simulation.
            seed: Random seed for network simulator reproducibility.
        """
        self.paragraphs = context_paragraphs
        self.network_sim = (
            NetworkSimulator(network_config, seed=seed) if network_config else None
        )
        self.search_count = 0
        self.search_log: List[Dict[str, Any]] = []

        # Pre-compute IDF over the paragraph corpus
        self._doc_tokens: List[Counter] = []
        self._idf: Dict[str, float] = {}
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self) -> None:
        """Build TF-IDF index over context paragraphs."""
        n_docs = len(self.paragraphs)
        if n_docs == 0:
            return

        # Tokenize each document (title + text)
        doc_freq: Counter = Counter()
        for para in self.paragraphs:
            tokens = self._tokenize(para.get("title", "") + " " + para.get("text", ""))
            tf = Counter(tokens)
            self._doc_tokens.append(tf)
            for token in set(tokens):
                doc_freq[token] += 1

        # Compute IDF
        for token, df in doc_freq.items():
            self._idf[token] = math.log((n_docs + 1) / (df + 1)) + 1  # smoothed IDF

    def _rank_paragraphs(self, query: str, top_k: int = 3) -> List[str]:
        """Rank paragraphs by TF-IDF similarity to query.

        Args:
            query: Search query string.
            top_k: Number of top results to return.

        Returns:
            List of formatted paragraph strings, most relevant first.
        """
        if not self.paragraphs:
            return []

        query_tokens = Counter(self._tokenize(query))
        if not query_tokens:
            # Return first top_k paragraphs as fallback
            return [
                f"[{p['title']}]\n{p['text']}" for p in self.paragraphs[:top_k]
            ]

        # Score each document
        scores = []
        for i, (para, doc_tf) in enumerate(zip(self.paragraphs, self._doc_tokens)):
            score = 0.0
            doc_len = sum(doc_tf.values()) or 1
            for token, query_count in query_tokens.items():
                if token in doc_tf:
                    tf = doc_tf[token] / doc_len  # normalized TF
                    idf = self._idf.get(token, 1.0)
                    score += query_count * tf * idf
            scores.append((score, i))

        # Sort by score descending
        scores.sort(key=lambda x: (-x[0], x[1]))

        results = []
        for score, idx in scores[:top_k]:
            para = self.paragraphs[idx]
            results.append(f"[{para['title']}]\n{para['text']}")

        return results

    def search(self, query: str, top_k: int = 3) -> str:
        """Search and return relevant paragraphs.

        If network_config is set, adds simulated latency for both
        the upload of the query and download of results.

        Args:
            query: Search query string.
            top_k: Number of top results to return.

        Returns:
            Formatted string of search results.
        """
        t0 = time.time()
        self.search_count += 1

        upload_delay_ms = 0.0
        download_delay_ms = 0.0

        # Simulate network latency for the search request (upload)
        if self.network_sim:
            event = self.network_sim.simulate_transfer_sync(
                len(query.encode("utf-8")), "upload"
            )
            upload_delay_ms = event.total_delay_ms

        # Rank and retrieve paragraphs
        results = self._rank_paragraphs(query, top_k)
        result_text = "\n\n".join(results)

        # Simulate download of results
        if self.network_sim:
            event = self.network_sim.simulate_transfer_sync(
                len(result_text.encode("utf-8")), "download"
            )
            download_delay_ms = event.total_delay_ms

        elapsed_ms = (time.time() - t0) * 1000

        # Log this search
        self.search_log.append({
            "search_number": self.search_count,
            "query": query,
            "num_results": len(results),
            "result_bytes": len(result_text.encode("utf-8")),
            "upload_delay_ms": upload_delay_ms,
            "download_delay_ms": download_delay_ms,
            "total_network_delay_ms": upload_delay_ms + download_delay_ms,
            "total_elapsed_ms": elapsed_ms,
        })

        return result_text

    def get_search_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics for all searches performed.

        Returns:
            Dict with search count, total network delay, etc.
        """
        total_network_ms = sum(
            entry["total_network_delay_ms"] for entry in self.search_log
        )
        total_result_bytes = sum(
            entry["result_bytes"] for entry in self.search_log
        )

        return {
            "search_count": self.search_count,
            "total_network_delay_ms": total_network_ms,
            "total_result_bytes": total_result_bytes,
            "avg_network_delay_ms": (
                total_network_ms / self.search_count if self.search_count > 0 else 0
            ),
            "search_log": self.search_log,
        }

    def get_network_summary(self) -> Optional[Dict[str, Any]]:
        """Get the network simulator summary, if network sim is active.

        Returns:
            Network simulator summary dict, or None.
        """
        if self.network_sim:
            return self.network_sim.get_summary()
        return None
