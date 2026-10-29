"""Qdrant projection backend (PORT-2), over Qdrant's HTTP API with explicit requests (no SDK
defaults): one collection per embedding space, cosine distance, float32 vectors; searches with
`params.exact = true` (a full scan, no HNSW), so the ranking is exact over the stored float32
vectors (`EXACT_NATIVE`). Writes wait for the operation to be applied (`wait=true`) and carry a
condition (`update_filter: blob_seq <= new`), so a late or repeated projection never replaces a
point's newer bytes; points of one request share that condition, so a batch is split by version.
The collection's configuration is read back and recorded in `describe()`."""

from __future__ import annotations

import httpx
import numpy as np

from ..errors import ContractError
from ..retrieval.data import canonical_hash


class QdrantBackend:
    name = "qdrant"

    def __init__(self, url, *, prefix="rvg", timeout=60.0, api_key=None):
        headers = {"api-key": api_key} if api_key else {}
        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=timeout, headers=headers)
        self.prefix = prefix
        self.version = self._call("GET", "/").get("version")
        self._ready, self._config = set(), {}

    def _call(self, method, path, body=None, params=None):
        response = self.client.request(method, path, json=body, params=params)
        if response.status_code >= 400:
            raise ContractError(
                "qdrant_request_failed", status=response.status_code, detail={"path": path}
            )
        data = response.json()
        return data.get("result", data)

    def collection(self, space):
        return f"{self.prefix}_{canonical_hash(space)[:16]}"

    def describe(self):
        return {
            "backend": "qdrant",
            "server_version": self.version,
            "api": "HTTP (httpx), explicit requests",
            "distance": "Cosine (score = cosine similarity, float32)",
            "search": "points/query params.exact=true (full scan)",
            "write": "wait=true with update_filter blob_seq <= new",
            "collections": self._config,
        }

    def ensure(self, space, dims):
        name = self.collection(space)
        if name in self._ready:
            return
        response = self.client.get(f"/collections/{name}")
        if response.status_code == 404:
            self._call(
                "PUT",
                f"/collections/{name}",
                {"vectors": {"size": int(dims), "distance": "Cosine"}},
            )
            for field, schema in (
                ("tenant", "keyword"),
                ("blob_seq", "integer"),
                ("observed_us", "integer"),
                ("inserted", "integer"),
            ):
                self._call(
                    "PUT",
                    f"/collections/{name}/index",
                    {"field_name": field, "field_schema": schema},
                    params={"wait": "true"},
                )
        info = self._call("GET", f"/collections/{name}")
        config = info.get("config", {})
        if config.get("params", {}).get("vectors", {}).get("size") != dims:
            raise ContractError("qdrant_collection_dimension_mismatch")
        self._config[name] = {
            "vectors": config.get("params", {}).get("vectors"),
            "hnsw_config": config.get("hnsw_config"),
            "optimizer_config": config.get("optimizer_config"),
        }
        self._ready.add(name)

    def upsert(self, space, points):
        name = self.collection(space)
        by_version = {}
        for point, vector, payload in points:
            by_version.setdefault(payload["blob_seq"], []).append(
                {
                    "id": point,
                    "vector": np.asarray(vector, dtype=np.float64).tolist(),
                    "payload": payload,
                }
            )
        for seq, group in sorted(by_version.items()):
            for start in range(0, len(group), 256):
                self._call(
                    "PUT",
                    f"/collections/{name}/points",
                    {
                        "points": group[start : start + 256],
                        "update_filter": {"must": [{"key": "blob_seq", "range": {"lte": seq}}]},
                    },
                    params={"wait": "true"},
                )

    def delete(self, space, point_ids):
        if point_ids:
            self._call(
                "POST",
                f"/collections/{self.collection(space)}/points/delete",
                {"points": list(point_ids)},
                params={"wait": "true"},
            )

    def count(self, tenant, space):
        return self._call(
            "POST",
            f"/collections/{self.collection(space)}/points/count",
            {"filter": {"must": [{"key": "tenant", "match": {"value": tenant}}]}, "exact": True},
        )["count"]

    def search(self, tenant, space, query, limit, *, include=None, exclude=None):
        must = [{"key": "tenant", "match": {"value": tenant}}]
        if include is not None:
            must.append({"has_id": list(include)})
        body = {
            "query": np.asarray(query, dtype=np.float64).tolist(),
            "limit": int(limit),
            "params": {"exact": True},
            "with_payload": ["blob_seq"],
            "filter": {
                "must": must,
                **({"must_not": [{"has_id": list(exclude)}]} if exclude else {}),
            },
        }
        result = self._call("POST", f"/collections/{self.collection(space)}/points/query", body)
        return [
            (str(p["id"]), float(p["score"]), int(p["payload"]["blob_seq"]))
            for p in result.get("points", [])
        ]

    def close(self):
        self.client.close()
