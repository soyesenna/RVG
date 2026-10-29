"""Storage ports and adapters (PORT-0): the knowledge state, vector blobs, the vector index and
the release fence, chosen per deployment and never per knowledge base or language."""

from .factory import Storage, StorageConfig, as_storage, load_config, local_storage, open_storage

__all__ = ["Storage", "StorageConfig", "as_storage", "load_config", "local_storage", "open_storage"]
