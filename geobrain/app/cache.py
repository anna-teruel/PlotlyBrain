import os
import uuid
from typing import Any

import diskcache

# Dedicated directory so session data does not collide with the background
# callback manager's own task cache.
_DATA_DIR = os.path.join(os.path.expanduser("~"), ".geobrain_cache", "session_data")

# 8 GiB ceiling so a couple of high-resolution volumes fit; diskcache evicts
# least-recently-used entries beyond this.
_cache = diskcache.Cache(_DATA_DIR, size_limit=8 * 1024**3)


def _key(session_id: str, key: str) -> str:
	return f"{session_id}::{key}"


def new_session() -> str:
	"""Create a fresh session id (no allocation needed until something is put)."""
	return uuid.uuid4().hex


def put(session_id: str, key: str, value: Any) -> None:
	"""Store ``value`` under ``key`` for the given session (visible to all processes)."""
	_cache.set(_key(session_id, key), value)


def get(session_id: str | None, key: str, default: Any = None) -> Any:
	"""Return a cached value, or ``default`` if the session/key is missing."""
	if not session_id:
		return default
	return _cache.get(_key(session_id, key), default)


def has(session_id: str | None, key: str) -> bool:
	"""Return whether ``key`` exists for ``session_id``."""
	if not session_id:
		return False
	return _key(session_id, key) in _cache


def clear(session_id: str | None) -> None:
	"""Drop everything stored for a session."""
	if not session_id:
		return
	prefix = f"{session_id}::"
	for k in list(_cache.iterkeys()):
		if isinstance(k, str) and k.startswith(prefix):
			_cache.delete(k)
