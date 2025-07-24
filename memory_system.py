import time
import json
import logging
import numpy as np
import uuid
import re
import threading
import yaml
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import OrderedDict

# Configure logging for the memory system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemorySystem.Core")

@dataclass
class ShortTermMemoryEntry:
    """Represents an individual entry in short-term memory."""
    id: str
    content: Dict[str, Any]  # Content of the entry (text, embeddings, etc.)
    timestamp: float  # When the entry was created
    access_count: int = 0  # Number of times this entry was accessed
    last_accessed: float = field(default_factory=time.time)  # Last time it was accessed
    ttl: Optional[float] = None  # Time-to-live in seconds (None = no default expiration)
    tags: Set[str] = field(default_factory=set)  # Tags for categorization
    emotional_valence: float = 0.0  # Emotional valence (-1.0 to 1.0)
    is_pinned: bool = False  # If True, entry is protected from automatic eviction
    embedding: Optional[Union[List[float], np.ndarray]] = None # Embedding vector for semantic similarity

    @property
    def age(self) -> float:
        """Returns the age of the entry in seconds."""
        return time.time() - self.timestamp

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Checks if the entry has expired based on TTL."""
        if self.is_pinned:
            return False  # Pinned entries never expire by TTL
        if self.ttl is None or self.ttl <= 0: # No TTL defined or invalid TTL, does not expire
            return False

        now = current_time or time.time()
        return (now - self.timestamp) > self.ttl

    def touch(self) -> None:
        """Updates the access counter and last accessed timestamp."""
        self.access_count += 1
        self.last_accessed = time.time()

    def pin(self) -> None:
        """Pins the entry, preventing automatic removal."""
        self.is_pinned = True

    def unpin(self) -> None:
        """Unpins the entry, allowing automatic removal."""
        self.is_pinned = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entry to a serializable dictionary."""
        result = asdict(self)
        result['tags'] = list(result['tags'])
        if self.embedding is not None and isinstance(self.embedding, np.ndarray):
            result['embedding'] = self.embedding.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShortTermMemoryEntry':
        """Creates an entry from a dictionary."""
        if 'tags' in data:
            data['tags'] = set(data['tags'])
        if 'embedding' in data and isinstance(data['embedding'], list):
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)

class ShortTermMemory:
    """
    Implements short-term memory storage for conversational context.
    Maintains entries for a configurable period before either evicting or discarding.
    """

    def __init__(self,
                 max_entries: int = 100,
                 default_ttl: Optional[float] = 3600.0,  # Default 1 hour
                 eviction_policy: str = "lru"): # "lru", "lfu", "fifo"
        """
        Initializes the short-term memory.

        Args:
            max_entries: Maximum number of entries to maintain.
            default_ttl: Default time-to-live for entries in seconds (None for no default expiration).
            eviction_policy: Policy to use when max_entries is exceeded ("lru", "lfu", "fifo").
        """
        self.logger = logging.getLogger("MemorySystem.ShortTermMemory")
        self.entries: OrderedDict[str, ShortTermMemoryEntry] = OrderedDict() # Keeps insertion order
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy.lower() # Standardize policy name

        # Index for tags (includes entities treated as tags)
        self.tag_index: Dict[str, Set[str]] = {} # tag (lowercase) -> set of entry IDs

        self.last_cleanup_timestamp = time.time() # To track manual cleanups

        self.logger.info(f"ShortTermMemory initialized (max: {max_entries}, default_ttl: {default_ttl}s, policy: {eviction_policy}).")

    def add(
            self,
            content: Dict[str, Any],
            ttl: Optional[float] = None,
            tags: Optional[Set[str]] = None,
            emotional_valence: float = 0.0,
            pin: bool = False,
            embedding: Optional[Union[List[float], np.ndarray]] = None) -> str:
        """
        Adds a new entry to short-term memory.

        Args:
            content: Content of the entry.
            ttl: Optional time-to-live (uses default_ttl if None).
            tags: Set of tags for categorization.
            emotional_valence: Associated emotional valence (-1.0 to 1.0).
            pin: If True, the entry is pinned and does not expire automatically.
            embedding: Optional embedding vector for semantic search.

        Returns:
            ID of the created entry.
        """
        if len(self.entries) >= self.max_entries:
            self._evict_entry()

        entry_id = str(uuid.uuid4())
        entry_ttl = ttl if ttl is not None else self.default_ttl
        processed_tags = {tag.lower() for tag in tags} if tags else set() # Normalize tags to lowercase

        entry = ShortTermMemoryEntry(
            id=entry_id,
            content=content,
            timestamp=time.time(),
            ttl=entry_ttl,
            tags=processed_tags,
            emotional_valence=emotional_valence,
            is_pinned=pin,
            embedding=embedding
        )

        self.entries[entry_id] = entry

        # Update tag index
        for tag in entry.tags:
            self.tag_index.setdefault(tag, set()).add(entry_id)

        self.logger.debug(f"Added entry {entry_id[:8]} to ShortTermMemory. Total entries: {len(self.entries)}.")
        return entry_id

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an entry by its ID and updates its access count.

        Args:
            entry_id: ID of the entry to retrieve.

        Returns:
            Content of the entry (dictionary) or None if not found.
        """
        entry = self.entries.get(entry_id)
        if entry:
            entry.touch() # Update access count and timestamp
            # Move accessed item to the end to maintain LRU/LFU heuristic through OrderedDict
            self.entries.move_to_end(entry_id)
            return entry.content
        self.logger.debug(f"Entry {entry_id[:8]} not found in ShortTermMemory.")
        return None

    def get_full_entry(self, entry_id: str) -> Optional[ShortTermMemoryEntry]:
        """
        Retrieves a complete entry object by its ID and updates its access count.

        Args:
            entry_id: ID of the entry to retrieve.

        Returns:
            ShortTermMemoryEntry object or None if not found.
        """
        entry = self.entries.get(entry_id)
        if entry:
            entry.touch() # Update access count and timestamp
            self.entries.move_to_end(entry_id)
            return entry
        self.logger.debug(f"Full entry {entry_id[:8]} not found in ShortTermMemory.")
        return None

    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Retrieves all entries associated with a specific tag (case-insensitive).

        Args:
            tag: Name of the tag.

        Returns:
            List of contents of entries mentioning the tag.
        """
        results = []
        normalized_tag = tag.lower()
        entry_ids = self.tag_index.get(normalized_tag, set())

        for entry_id in entry_ids:
            # Use get_full_entry to update access info
            entry = self.get_full_entry(entry_id)
            if entry and not entry.is_expired(current_time=time.time()): # Only return non-expired entries
                results.append(entry.content)
        return results

    def search_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of contents of the most recent entries.
        """
        results = []
        # OrderedDict maintains insertion order, so the last N elements are the most recent.
        # Iterate from the end (most recent) backwards.
        recent_entry_ids = list(self.entries.keys())

        # Iterate in reverse to get entries from newest to oldest
        for entry_id in reversed(recent_entry_ids):
            if len(results) >= limit: # Stop if limit reached
                break
            entry = self.get_full_entry(entry_id) # Use get_full_entry to update access info
            if entry and not entry.is_expired(current_time=time.time()):
                results.append(entry.content)
        return results

    def search_by_embedding(
            self,
            query_embedding: Union[List[float], np.ndarray],
            threshold: float = 0.7,
            limit: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches for semantically similar entries based on their embeddings.

        Args:
            query_embedding: The embedding vector to compare against.
            threshold: Minimum cosine similarity score to consider a match.
            limit: Maximum number of results to return.

        Returns:
            A list of tuples, each containing (content dictionary, similarity score),
            sorted by similarity in descending order.
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-9: # Handle near-zero query embeddings
            self.logger.warning("Query embedding is a zero vector. Cannot perform similarity search.")
            return []
        query_embedding_normalized = query_embedding / query_norm

        similar_entries: List[Tuple['ShortTermMemoryEntry', float]] = []

        for entry_id, entry in self.entries.items():
            # Check if embedding is present and entry is not expired
            if entry.embedding is not None and not entry.is_expired(current_time=time.time()):
                entry_embedding_np = np.array(entry.embedding) # Ensure numpy array

                entry_norm = np.linalg.norm(entry_embedding_np)
                if entry_norm < 1e-9: # Handle near-zero stored embeddings
                    continue
                entry_embedding_normalized = entry_embedding_np / entry_norm

                similarity = np.dot(query_embedding_normalized, entry_embedding_normalized)

                if similarity >= threshold:
                    similar_entries.append((entry, similarity))

        similar_entries.sort(key=lambda x: x[1], reverse=True) # Sort by similarity

        results = []
        for entry, similarity in similar_entries[:limit]:
            entry.touch() # Update access stats
            self.entries.move_to_end(entry.id) # Move accessed item to end for LRU
            results.append((entry.content, float(similarity))) # Convert numpy float to native float

        return results

    def pin_entry(self, entry_id: str) -> bool:
        """
        Pins an entry to prevent automatic expiration or eviction.

        Args:
            entry_id: ID of the entry to pin.

        Returns:
            True if successful, False otherwise.
        """
        entry = self.entries.get(entry_id)
        if entry:
            entry.pin() # Use the method on the dataclass instance
            self.logger.debug(f"Entry {entry_id[:8]} pinned.")
            return True
        self.logger.warning(f"Attempted to pin non-existent entry {entry_id[:8]}.")
        return False

    def unpin_entry(self, entry_id: str) -> bool:
        """
        Removes pinning from an entry, allowing it to expire or be evicted.

        Args:
            entry_id: ID of the entry.

        Returns:
            True if successful, False otherwise.
        """
        entry = self.entries.get(entry_id)
        if entry:
            entry.unpin() # Use the method on the dataclass instance
            self.logger.debug(f"Entry {entry_id[:8]} unpinned.")
            return True
        self.logger.warning(f"Attempted to unpin non-existent entry {entry_id[:8]}.")
        return False

    def remove(self, entry_id: str) -> bool:
        """
        Removes an entry from short-term memory.

        Args:
            entry_id: ID of the entry to remove.

        Returns:
            True if removed successfully, False otherwise.
        """
        entry = self.entries.pop(entry_id, None)
        if entry:
            # Clean up references in tag index
            for tag in entry.tags:
                self.tag_index.setdefault(tag, set()).discard(entry_id) # Use discard to safely remove
                if not self.tag_index.get(tag): # Clean up empty sets if empty
                    del self.tag_index[tag]

            self.logger.debug(f"Removed entry {entry_id[:8]} from ShortTermMemory.")
            return True
        self.logger.warning(f"Attempted to remove non-existent entry {entry_id[:8]}.")
        return False

    def cleanup(self) -> int:
        """
        Removes expired entries from short-term memory.

        Returns:
            Number of entries removed.
        """
        removed_count = 0
        current_time = time.time()

        # Collect IDs of expired entries that are not pinned
        expired_entry_ids = [
            entry_id for entry_id, entry in list(self.entries.items())
            if entry.is_expired(current_time=current_time)
        ]

        # Remove expired entries
        for entry_id in expired_entry_ids:
            if self.remove(entry_id): # Use self.remove to properly update indexes
                removed_count += 1

        if removed_count > 0:
            self.logger.info(f"Cleanup: removed {removed_count} expired entries from ShortTermMemory.")

        self.last_cleanup_timestamp = current_time
        return removed_count

    def _evict_entry(self) -> None:
        """
        Evicts entries when the maximum size is exceeded, according to the configured policy.
        Pinned entries are prioritized against eviction.
        """
        # This while loop handles cases where multiple entries need to be evicted
        # to bring the total count below max_entries, e.g., if max_entries was reduced dynamically
        while len(self.entries) > self.max_entries:
            # Find candidates for eviction (must not be pinned)
            eviction_candidates = []
            for entry_id, entry in self.entries.items():
                if not entry.is_pinned:
                    eviction_candidates.append(entry)

            if not eviction_candidates: # All entries are pinned, cannot evict non-pinned
                self.logger.warning("ShortTermMemory is full and all entries are pinned. Cannot evict any more.")
                break # Exit loop as no more unpinned entries can be removed

            # Select entry to evict based on policy
            entry_to_evict: Optional[ShortTermMemoryEntry] = None

            if self.eviction_policy == "fifo":
                # FIFO: oldest non-pinned by creation timestamp (first in OrderedDict)
                for entry_id_candidate in self.entries.keys():
                    if not self.entries[entry_id_candidate].is_pinned:
                        entry_to_evict = self.entries[entry_id_candidate]
                        break
            elif self.eviction_policy == "lru":
                # LRU: least recently used non-pinned (their position in OrderedDict reflects usage, oldest is at beginning)
                # Iterate from the front (least recently used) until an unpinned entry is found
                for entry_id_candidate in self.entries.keys():
                    if not self.entries[entry_id_candidate].is_pinned:
                        entry_to_evict = self.entries[entry_id_candidate]
                        break
            elif self.eviction_policy == "lfu":
                # LFU: least frequently used non-pinned (minimum access_count)
                if eviction_candidates: # Ensure there are candidates
                    entry_to_evict = min(eviction_candidates, key=lambda entry: entry.access_count)
            else:
                self.logger.error(f"Unknown eviction policy: {self.eviction_policy}. Defaulting to FIFO.")
                # Fallback to FIFO by default if policy is unknown
                for entry_id_candidate in self.entries.keys():
                    if not self.entries[entry_id_candidate].is_pinned:
                        entry_to_evict = self.entries[entry_id_candidate]
                        break

            if entry_to_evict:
                self.remove(entry_to_evict.id) # Remove the selected entry
            else:
                self.logger.error("No suitable unpinned entry found for eviction, despite exceeding max_entries. This should not happen if not all entries are pinned.")
                break

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns comprehensive statistics about the short-term memory.

        Returns:
            A dictionary with various statistics.
        """
        total_entries = len(self.entries)
        pinned_entries = sum(1 for e in self.entries.values() if e.is_pinned)

        # Filter for active (non-expired) entries before calculating averages/ages
        active_entries_list = [e for e in self.entries.values() if not e.is_expired(current_time=time.time())]
        active_entries_count = len(active_entries_list)

        avg_access_count_active = 0.0
        if active_entries_count > 0:
            avg_access_count_active = sum(e.access_count for e in active_entries_list) / active_entries_count

        current_time = time.time()
        oldest_active_age = 0.0
        newest_active_age = 0.0
        avg_active_age = 0.0

        if active_entries_list:
            active_ages = [(current_time - e.timestamp) for e in active_entries_list]
            oldest_active_age = max(active_ages)
            newest_active_age = min(active_ages)
            avg_active_age = sum(active_ages) / active_entries_count

        return {
            "total_entries": total_entries,
            "active_entries": active_entries_count,
            "expired_entries": total_entries - active_entries_count,
            "max_entries_limit": self.max_entries,
            "fill_percentage": (total_entries / self.max_entries) * 100 if self.max_entries > 0 else 0.0,
            "pinned_entries": pinned_entries,
            "default_ttl_seconds": self.default_ttl,
            "eviction_policy": self.eviction_policy,
            "avg_access_count_active": avg_access_count_active,
            "tags_indexed_count": len(self.tag_index),
            "oldest_active_entry_age_seconds": oldest_active_age,
            "newest_active_entry_age_seconds": newest_active_age,
            "avg_active_entry_age_seconds": avg_active_age,
            "last_cleanup_timestamp": self.last_cleanup_timestamp
        }

@dataclass
class LongTermEntry:
    """Represents an individual entry in long-term memory."""
    id: str
    content: Dict[str, Any]  # Content of the entry (text, embeddings, etc.)
    timestamp: float  # When the entry was created
    embedding: Optional[Union[List[float], np.ndarray]] = None  # Semantic embedding of the content
    emotional_valence: float = 0.0  # Emotional valence (-1.0 to 1.0)
    loyalty_coefficient: float = 0.5  # Loyalty coefficient (0.0 to 1.0)
    tags: Set[str] = field(default_factory=set)  # Tags for categorization
    access_count: int = 0  # Number of times this entry was accessed
    last_accessed: float = field(default_factory=time.time)  # Last time it was accessed
    importance: float = 0.5  # Importance of the entry (0.0 to 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entry to a serializable dictionary."""
        result = asdict(self)
        result['tags'] = list(result['tags']) # Convert set to list for JSON serialization
        if isinstance(self.embedding, np.ndarray):
            result['embedding'] = self.embedding.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongTermEntry':
        """Creates an entry from a dictionary."""
        if 'tags' in data:
            data['tags'] = set(data['tags'])
        if 'embedding' in data and isinstance(data['embedding'], list):
            data['embedding'] = np.array(data['embedding'])

        # Ensure all expected fields are present (using .get with defaults for robustness)
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            content=data.get('content', {}),
            timestamp=data.get('timestamp', time.time()),
            embedding=data.get('embedding'),
            emotional_valence=data.get('emotional_valence', 0.0),
            loyalty_coefficient=data.get('loyalty_coefficient', 0.5),
            tags=data.get('tags', set()),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', time.time()),
            importance=data.get('importance', 0.5)
        )

    def touch(self) -> None:
        """Updates the access counter and last accessed timestamp."""
        self.access_count += 1
        self.last_accessed = time.time()

    def decay_importance(self, factor: float = 0.05) -> None:
        """Decreases the importance of the entry."""
        self.importance = max(0.0, self.importance - factor)


class LongTermMemory:
    """
    Implements long-term memory storage for persistent information.
    Entries are indexed and can be retrieved by various criteria.
    Thread-safe access via internal RLock.
    """

    def __init__(self, storage_dir: str = "data/context_memory/long_term", max_entries: int = 10000, importance_threshold: float = 0.3):
        """
        Initializes the long-term memory.

        Args:
            storage_dir: Directory for persistent storage.
            max_entries: Maximum number of entries before eviction policy is applied.
            importance_threshold: Importance below which entries might be purged.
        """
        self.logger = logging.getLogger("MemorySystem.LongTermMemory")
        self.storage_dir = Path(storage_dir)
        self.entries: Dict[str, 'LongTermEntry'] = {}
        self.entity_index: Dict[str, Set[str]] = {}  # Entity (lowercase) -> {entry_ids}
        self.tag_index: Dict[str, Set[str]] = {}  # Tag (lowercase) -> {entry_ids}
        self.max_entries = max_entries
        self.importance_threshold = importance_threshold
        self._lock = threading.RLock() # RLock for thread-safe access

        with self._lock:
            self.storage_dir.mkdir(parents=True, exist_ok=True) # Ensure storage directory exists
            self._load_entries() # Load existing entries
            self.logger.info(f"LongTermMemory initialized ({len(self.entries)} entries loaded from {self.storage_dir}).")

    def _load_entries(self) -> None:
        """Loads entries from persistent storage."""
        # This method is called within the lock during __init__ and implicitly protected
        # by other methods calling it under their lock.
        try:
            index_file = self.storage_dir / "index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)

                # Clear current indexes before rebuilding from loaded data
                self.entries.clear()
                self.entity_index.clear()
                self.tag_index.clear()

                for entry_id in index_data.get("entries", []):
                    entry_path = self.storage_dir / f"{entry_id}.json"
                    if entry_path.exists():
                        try:
                            with open(entry_path, 'r', encoding='utf-8') as ef:
                                entry_data = json.load(ef)
                                entry = LongTermEntry.from_dict(entry_data)
                                self.entries[entry_id] = entry

                                # Rebuild indexes for the loaded entry
                                self._update_indices(entry)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Error decoding JSON for entry {entry_id} from {entry_path}: {e}. Skipping entry.")
                        except KeyError as e:
                            self.logger.error(f"Missing key in entry {entry_id} data from {entry_path}: {e}. Skipping entry.")
                        except Exception as e:
                            self.logger.error(f"Unexpected error loading specific entry {entry_id} from {entry_path}: {e}", exc_info=True)
                    else:
                        self.logger.warning(f"Entry file {entry_path} not found for ID {entry_id}. Skipping.")
            else:
                self.logger.info(f"No index.json found in {self.storage_dir}. Starting with empty memory.")

            self.logger.info(f"Loaded {len(self.entries)} entries from storage.")
        except Exception as e:
            self.logger.error(f"Error loading entries from storage: {e}", exc_info=True)

    def _save_entry(self, entry: LongTermEntry) -> None:
        """Saves an entry to persistent storage."""
        # This method is called by other public methods which are already locked.
        try:
            entry_path = self.storage_dir / f"{entry.id}.json"
            with open(entry_path, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)

            self._update_index_file()
        except Exception as e:
            self.logger.error(f"Error saving entry {entry.id} to {entry_path}: {e}", exc_info=True)

    def _update_index_file(self) -> None:
        """Updates the main index file (index.json) that lists all entry IDs."""
        # This method is called by other public methods which are already locked.
        try:
            index_data = {
                "entries": list(self.entries.keys()),
                "last_updated": time.time()
            }

            index_file = self.storage_dir / "index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error updating index file {index_file}: {e}", exc_info=True)

    def _extract_entities_from_content(self, content: Dict[str, Any]) -> Set[str]:
        """
        Extracts entities from an entry's content to use in indexing.
        This is a simplified approach, usually backed by an NLP service.
        It looks for capitalized words or phrases.
        """
        entities = set()
        text = str(content.get('text', ''))

        # Regex to find capitalized words or phrases (e.g., "New York", "AI Project")
        # Ignores single capital letters or all-caps words unless they are part of a phrase.
        # This is still a heuristic, a real NLP entity recognizer is recommended.
        matches = re.findall(r'\b[A-Z][a-z0-9]*(?:\s+[A-Z][a-z0-9]*)*\b', text)
        for match in matches:
            # Exclude common acronyms or single letters that are not proper nouns in context
            if len(match) > 1 or (len(match) == 1 and match.isupper() and match.isascii()): # Filter out single caps 'A', 'I', etc.
                 # Check if it's not entirely uppercase unless it's a known multi-word acronym
                if not match.isupper() or (match.isupper() and ' ' in match):
                    entities.add(match.lower()) # Store in lowercase for consistent indexing
        return entities

    def _update_indices(self, entry: LongTermEntry) -> None:
        """Updates the internal search indices for a given entry."""
        # This method is called by other public methods which are already locked.
        # Update tag index
        for tag in entry.tags:
            normalized_tag = tag.lower() # Standardize tags to lowercase
            self.tag_index.setdefault(normalized_tag, set()).add(entry.id)

        # Update entity index
        # Expecting entities to be present in entry.content['entities'] list or extracted
        extracted_entities = set(entry.content.get('entities', [])).union(self._extract_entities_from_content(entry.content))
        for entity in extracted_entities:
            normalized_entity = entity.lower() # Standardize entities to lowercase
            self.entity_index.setdefault(normalized_entity, set()).add(entry.id)

    def _remove_from_indices(self, entry: LongTermEntry) -> None:
        """Removes an entry's references from the internal search indices."""
        # This method is called by other public methods which are already locked.
        # Remove from tag index
        for tag in entry.tags:
            normalized_tag = tag.lower()
            if normalized_tag in self.tag_index and entry.id in self.tag_index[normalized_tag]:
                self.tag_index[normalized_tag].discard(entry.id) # Use discard to safely remove
                if not self.tag_index[normalized_tag]: # Clean up empty sets
                    del self.tag_index[normalized_tag]

        # Remove from entity index
        extracted_entities = set(entry.content.get('entities', [])).union(self._extract_entities_from_content(entry.content))
        for entity in extracted_entities:
            normalized_entity = entity.lower()
            if normalized_entity in self.entity_index and entry.id in self.entity_index[normalized_entity]:
                self.entity_index[normalized_entity].discard(entry.id) # Use discard to safely remove
                if not self.entity_index[normalized_entity]: # Clean up empty sets
                    del self.entity_index[normalized_entity]

    def add(
            self,
            content: Dict[str, Any],
            embedding: Optional[Union[List[float], np.ndarray]] = None,
            emotional_valence: float = 0.0,
            loyalty_coefficient: float = 0.5,
            tags: Optional[Set[str]] = None,
            importance: float = 0.5) -> str:
        """
        Adds a new entry to long-term memory.

        Args:
            content: Content of the entry.
            embedding: Optional semantic embedding.
            emotional_valence: Emotional valence (-1.0 to 1.0).
            loyalty_coefficient: Loyalty coefficient (0.0 to 1.0).
            tags: Set of tags.
            importance: Importance of the entry (0.0 to 1.0).

        Returns:
            ID of the created entry.
        """
        with self._lock:
            # Evict entries if capacity is reached before adding new one
            if len(self.entries) >= self.max_entries:
                self._evict_entries(count=int(self.max_entries * 0.1)) # Evict 10% of entries if full

            # Convert numpy array to list for JSON serialization if necessary
            processed_embedding = None
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    processed_embedding = embedding.tolist()
                elif isinstance(embedding, list):
                    processed_embedding = embedding
                else:
                    self.logger.warning(f"Unsupported embedding type: {type(embedding)}. Embedding will be ignored for entry content: {content.get('text', str(content))[:50]}...")

            entry_id = str(uuid.uuid4())

            entry = LongTermEntry(
                id=entry_id,
                content=content,
                timestamp=time.time(),
                embedding=processed_embedding,
                emotional_valence=emotional_valence,
                loyalty_coefficient=loyalty_coefficient,
                tags=tags if tags is not None else set(),
                importance=importance
            )

            self.entries[entry_id] = entry
            self._update_indices(entry)
            self._save_entry(entry)

            self.logger.debug(f"Added entry {entry_id[:8]} to long-term memory. Content: {content.get('text', '')[:50]}...")
            return entry_id

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an entry by its ID and updates its access count.

        Args:
            entry_id: ID of the entry to retrieve.

        Returns:
            Content of the entry (dictionary) or None if not found.
        """
        with self._lock:
            entry = self.entries.get(entry_id)
            if entry:
                entry.touch() # Update access count and timestamp
                self._save_entry(entry) # Persist the updated access info
                return entry.content
            self.logger.debug(f"Entry {entry_id[:8]} not found in LongTermMemory.")
            return None

    def get_full_entry(self, entry_id: str) -> Optional['LongTermEntry']:
        """
        Retrieves a complete entry object by its ID.

        Args:
            entry_id: ID of the entry to retrieve.

        Returns:
            LongTermEntry object or None if not found.
        """
        with self._lock:
            entry = self.entries.get(entry_id)
            if entry:
                entry.touch() # Update access count and timestamp
                self._save_entry(entry) # Persist the updated access info
                return entry
            self.logger.debug(f"Full entry {entry_id[:8]} not found in LongTermMemory.")
            return None

    def update(self, entry_id: str, **kwargs) -> bool:
        """
        Updates an existing entry with new values for specified fields.

        Args:
            entry_id: ID of the entry to update.
            **kwargs: Fields to update (e.g., content, importance, tags).

        Returns:
            True if updated successfully, False otherwise.
        """
        with self._lock:
            if entry_id not in self.entries:
                self.logger.warning(f"Attempted to update non-existent entry {entry_id[:8]}.")
                return False

            entry = self.entries[entry_id]

            # Store old tags and entities for index cleanup (before content/tags are updated)
            old_tags = entry.tags.copy()
            current_entities_in_content = set(entry.content.get('entities', []))
            old_extracted_entities = self._extract_entities_from_content(entry.content)
            old_entities = old_extracted_entities.union(current_entities_in_content)

            # Update specified fields
            for key, value in kwargs.items():
                if key == 'content':
                    entry.content.update(value) # Merge content dictionaries
                elif key == 'tags':
                    if value is not None:
                        # Remove old tags from index
                        for tag in old_tags:
                            normalized_tag = tag.lower()
                            if normalized_tag in self.tag_index and entry_id in self.tag_index[normalized_tag]:
                                self.tag_index[normalized_tag].discard(entry_id)
                                if not self.tag_index[normalized_tag]: del self.tag_index[normalized_tag]
                        entry.tags = set(value) # Assign the new set of tags
                    else:
                        entry.tags = set()
                elif key == 'embedding': # Handle embedding explicitly to ensure list conversion
                    if isinstance(value, np.ndarray):
                        entry.embedding = value.tolist()
                    elif isinstance(value, list) or value is None:
                        entry.embedding = value
                    else:
                        self.logger.warning(f"Unsupported embedding type {type(value)} for update.")
                elif hasattr(entry, key): # Direct attribute update
                    setattr(entry, key, value)
                else:
                    self.logger.warning(f"Attempted to update non-existent or unhandled field '{key}' in LongTermEntry {entry_id[:8]}.")

            # After content/tags update, re-extract new entities for new indexing
            new_extracted_entities = self._extract_entities_from_content(entry.content)
            new_content_entities = set(entry.content.get('entities', []))
            new_entities = new_extracted_entities.union(new_content_entities)

            # Update entity index: Remove old references that are no longer present
            for entity in old_entities.difference(new_entities):
                normalized_entity = entity.lower()
                if normalized_entity in self.entity_index and entry_id in self.entity_index[normalized_entity]:
                    self.entity_index[normalized_entity].discard(entry_id)
                    if not self.entity_index[normalized_entity]: del self.entity_index[normalized_entity]

            # Add new references that were not present before
            for entity in new_entities.difference(old_entities):
                normalized_entity = entity.lower()
                self.entity_index.setdefault(normalized_entity, set()).add(entry_id)

            # Update newly set tags in index
            for tag in entry.tags:
                self.tag_index.setdefault(tag.lower(), set()).add(entry_id) # Add new tags to index

            self._save_entry(entry)

            self.logger.debug(f"Updated entry {entry_id[:8]} in long-term memory.")
            return True

    def remove(self, entry_id: str) -> bool:
        """
        Removes an entry from long-term memory permanently.

        Args:
            entry_id: ID of the entry to remove.

        Returns:
            True if removed successfully, False otherwise.
        """
        with self._lock:
            entry = self.entries.pop(entry_id, None) # Remove from main dictionary
            if not entry:
                self.logger.warning(f"Attempted to remove non-existent entry {entry_id[:8]}.")
                return False

            self._remove_from_indices(entry) # Remove from search indices

            # Remove entry file from disk
            try:
                entry_path = self.storage_dir / f"{entry.id}.json"
                if entry_path.exists():
                    os.remove(entry_path)
            except Exception as e:
                self.logger.error(f"Error removing entry file {entry.id[:8]} from disk: {e}", exc_info=True)

            self._update_index_file() # Update main index file

            self.logger.debug(f"Removed entry {entry.id[:8]} from long-term memory.")
            return True

    def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for entries by tag (case-insensitive).

        Args:
            tag: Tag to search for.
            limit: Maximum number of results.

        Returns:
            List of entry contents (dictionaries).
        """
        with self._lock:
            results_content: List[Dict[str, Any]] = []
            normalized_tag = tag.lower()
            entry_ids = self.tag_index.get(normalized_tag, set())

            for entry_id in list(entry_ids)[:limit]: # Iterate over a copy and limit
                # Directly get full entry to touch and save
                entry = self.get_full_entry(entry_id)
                if entry:
                    results_content.append(entry.content)

            return results_content

    def search_by_entity(self, entity: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for entries that mention an entity (case-insensitive).

        Args:
            entity: Entity to search for.
            limit: Maximum number of results.

        Returns:
            List of entry contents (dictionaries).
        """
        with self._lock:
            results_content: List[Dict[str, Any]] = []
            normalized_entity = entity.lower()
            entry_ids = self.entity_index.get(normalized_entity, set())

            for entry_id in list(entry_ids)[:limit]: # Iterate over a copy and limit
                # Directly get full entry to touch and save
                entry = self.get_full_entry(entry_id)
                if entry:
                    results_content.append(entry.content)

            return results_content

    def search_by_embedding(
            self,
            query_embedding: Union[List[float], np.ndarray],
            threshold: float = 0.7,
            limit: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches for semantically similar entries to an embedding (cosine similarity).

        Args:
            query_embedding: Query embedding (list of floats or numpy array).
            threshold: Minimum similarity threshold (cosine similarity).
            limit: Maximum number of results.

        Returns:
            List of tuples (content dictionary, similarity score).
        """
        with self._lock:
            if isinstance(query_embedding, list):
                query_embedding_np = np.array(query_embedding)
            else:
                query_embedding_np = query_embedding

            query_norm = np.linalg.norm(query_embedding_np)
            if query_norm < 1e-9: # Handle near-zero query embeddings to avoid division by zero
                self.logger.warning("Query embedding is a zero vector. Cannot perform similarity search.")
                return []
            query_embedding_normalized = query_embedding_np / query_norm

            results: List[Tuple['LongTermEntry', float]] = [] # Store (entry_object, similarity) initially

            for entry_id, entry in self.entries.items(): # Iterate through all entries in memory
                if entry.embedding is not None:
                    entry_embedding_np = np.array(entry.embedding)

                    entry_norm = np.linalg.norm(entry_embedding_np)
                    if entry_norm < 1e-9: # Handle near-zero entry embeddings
                        continue
                    entry_embedding_normalized = entry_embedding_np / entry_norm

                    similarity = np.dot(query_embedding_normalized, entry_embedding_normalized)

                    if similarity >= threshold:
                        results.append((entry, float(similarity))) # Store entry object, cast np.float to float

            results.sort(key=lambda x: x[1], reverse=True) # Sort by similarity

            final_results = []
            for entry_obj, similarity in results[:limit]:
                entry_obj.touch() # Update access count
                self._save_entry(entry_obj) # Persist the updated access info
                final_results.append((entry_obj.content, similarity))

            return final_results

    def search_by_date_range(self,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for entries within a specific date range.

        Args:
            start_time: Start timestamp (None for no lower limit).
            end_time: End timestamp (None for no upper limit, uses current time).
            limit: Maximum number of results.

        Returns:
            List of entry contents (dictionaries).
        """
        with self._lock:
            results_content: List[Dict[str, Any]] = []

            effective_end_time = end_time if end_time is not None else time.time()
            effective_start_time = start_time if start_time is not None else 0.0

            filtered_entries = [
                (entry_id, entry) for entry_id, entry in self.entries.items()
                if effective_start_time <= entry.timestamp <= effective_end_time
            ]

            filtered_entries.sort(key=lambda x: x[1].timestamp, reverse=True)

            for entry_id, entry_obj in filtered_entries[:limit]:
                entry_obj.touch() # Update access count
                self._save_entry(entry_obj) # Persist the updated access info
                results_content.append(entry_obj.content)

            return results_content

    def _evict_entries(self, count: int = 10) -> int:
        """
        Removes the least relevant entries when the maximum entry limit is exceeded.
        Uses heuristics based on: lowest importance, least recently accessed, and lowest loyalty.
        This method is called by other public methods which are already locked.

        Args:
            count: Number of entries to remove.

        Returns:
            Number of entries removed.
        """
        if len(self.entries) <= self.max_entries - count:
            self.logger.debug(f"Current entries ({len(self.entries)}) below eviction threshold ({self.max_entries - count}). No eviction needed.")
            return 0

        entries_to_consider: List[LongTermEntry] = list(self.entries.values())

        # Prioritize for eviction (in ascending order of "undesirability"):
        # 1. Lowest importance
        # 2. Least recently accessed (higher last_accessed value means more recent, so sort by lower value)
        # 3. Lowest loyalty coefficient
        # 4. Lowest access count (although last_accessed is usually more robust)

        entries_to_consider.sort(
            key=lambda entry: (
                entry.importance,           # Primary sort: least important first
                entry.last_accessed,        # Secondary sort: least recently accessed first
                entry.loyalty_coefficient,  # Tertiary sort: lowest loyalty first
                entry.access_count          # Quaternary sort: least accessed first
            )
        )

        removed_count = 0
        entries_to_remove_ids = [entry.id for entry in entries_to_consider[:count]]

        for entry_id in entries_to_remove_ids:
            if self.remove(entry_id): # Use self.remove to properly update indices and disk
                removed_count += 1

        if removed_count > 0:
            self.logger.info(f"Evicted {removed_count} least relevant entries from long-term memory.")

        return removed_count

    def purge_low_importance(self) -> int:
        """
        Removes entries with importance below the configured `importance_threshold`.
        This is typically called as part of a garbage collection or synchronization process.
        This method is called by other public methods which are already locked.

        Returns:
            The number of entries successfully removed.
        """
        removed_count = 0
        entries_to_check = list(self.entries.values()) # Iterate over a copy

        for entry in entries_to_check:
            if entry.importance < self.importance_threshold:
                if self.remove(entry.id): # Call remove to handle proper cleanup
                    removed_count += 1

        if removed_count > 0:
            self.logger.info(f"Purged {removed_count} low-importance entries from LongTermMemory (threshold: {self.importance_threshold}).")
        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns comprehensive statistics about the long-term memory.

        Returns:
            A dictionary with various statistics.
        """
        with self._lock:
            if not self.entries:
                return {
                    "total_entries": 0,
                    "max_entries_limit": self.max_entries,
                    "storage_size_kb": 0.0,
                    "oldest_entry_datetime": None,
                    "newest_entry_datetime": None,
                    "tags_count": len(self.tag_index),
                    "entities_count": len(self.entity_index),
                    "avg_importance": 0.0,
                    "min_importance": 0.0,
                    "max_importance": 0.0,
                    "avg_loyalty": 0.0,
                    "total_access_count": 0
                }

            timestamps = [entry.timestamp for entry in self.entries.values()]
            importances = [entry.importance for entry in self.entries.values()]
            loyalties = [entry.loyalty_coefficient for entry in self.entries.values()]
            access_counts = [entry.access_count for entry in self.entries.values()]

            oldest = min(timestamps)
            newest = max(timestamps)

            storage_size_bytes = 0
            for entry_file in self.storage_dir.glob("*.json"):
                try:
                    storage_size_bytes += entry_file.stat().st_size
                except OSError: # Catch if file is removed during stats collection
                    self.logger.warning(f"Could not get size for file {entry_file.name}. Skipping.")

            stats = {
                "total_entries": len(self.entries),
                "max_entries_limit": self.max_entries,
                "current_fill_percentage": (len(self.entries) / self.max_entries) * 100 if self.max_entries > 0 else 0.0,
                "storage_size_kb": storage_size_bytes / 1024.0, # Convert bytes to KB
                "oldest_entry_datetime": datetime.fromtimestamp(oldest).isoformat(), # ISO format string
                "newest_entry_datetime": datetime.fromtimestamp(newest).isoformat(), # ISO format string
                "tags_count": len(self.tag_index),
                "entities_count": len(self.entity_index),
                "avg_importance": np.mean(importances),
                "min_importance": np.min(importances),
                "max_importance": np.max(importances),
                "avg_loyalty": np.mean(loyalties),
                "total_access_count": sum(access_counts)
            }

        return stats

class ContextManager:
    """
    Central Context Manager for a generic AI/LLM system.

    Coordinates the interaction between short-term and long-term memory, applies
    retention, synchronization, and cleanup policies, and integrates with the
    processing pipeline.
    """

    def __init__(self,
                 config_file: Optional[str] = None,
                 storage_dir: str = "data/context_memory"):
        """
        Initializes the context manager.

        Args:
            config_file: Path to a YAML or JSON configuration file.
            storage_dir: Base directory for memory storage.
        """
        self.logger = logging.getLogger("MemorySystem.ContextManager")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config(config_file)

        # Initialize Short-Term Memory
        self.short_term = ShortTermMemory(
            max_entries=self.config.get("short_term", {}).get("max_entries", 100),
            default_ttl=self.config.get("short_term", {}).get("default_ttl", 3600.0),
            eviction_policy=self.config.get("short_term", {}).get("eviction_policy", "lru")
        )

        # Initialize Long-Term Memory
        long_term_storage_path = self.storage_dir / "long_term"
        long_term_storage_path.mkdir(parents=True, exist_ok=True)
        self.long_term = LongTermMemory(
            storage_dir=str(long_term_storage_path),
            importance_threshold=self.config.get("long_term", {}).get("importance_threshold", 0.3)
        )

        # Active context state
        self.active_context = {
            "current_topic": None,
            "active_entities": set(),
            "emotional_state": 0.0,
            "context_stability": 1.0,
            "session_start_time": time.time(),
            "last_activity_time": time.time()
        }

        # Synchronization settings
        self.sync_interval = self.config.get("sync", {}).get("interval_seconds", 300)
        self.drift_threshold = self.config.get("sync", {}).get("drift_threshold", 0.3)
        self.last_sync_time = time.time()

        self.lock = threading.RLock() # Main lock for ContextManager operations

        if self.config.get("sync", {}).get("auto_sync", True):
            self.sync_stop_event = threading.Event()
            self.sync_thread = threading.Thread(target=self._background_sync_loop, daemon=True)
            self.sync_thread.start()
            self.logger.info(f"Background sync started with interval {self.sync_interval} seconds.")
        else:
            self.logger.info("Auto-sync is disabled.")
            self.sync_thread = None
            self.sync_stop_event = None

        self.logger.info("Context Manager initialized.")

    def __del__(self):
        """Ensures background sync thread is stopped on object destruction."""
        if self.sync_stop_event:
            self.sync_stop_event.set()
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)
            if self.sync_thread.is_alive():
                self.logger.warning("Background sync thread did not terminate cleanly.")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Loads context manager configuration from a file.

        Args:
            config_file: Path to a configuration file (YAML or JSON) or None for default.

        Returns:
            A dictionary containing the loaded or default configurations.
        """
        default_config = {
            "short_term": {
                "max_entries": 100,
                "default_ttl": 3600.0,
                "eviction_policy": "lru",
                "recent_search_limit": 5,
                "semantic_search_threshold": 0.6,
                "stability_check_limit": 5
            },
            "long_term": {
                "max_entries": 10000,
                "importance_threshold": 0.3,
                "semantic_search_threshold": 0.6
            },
            "sync": {
                "auto_sync": True,
                "interval_seconds": 300,
                "drift_threshold": 0.3,
                "archive_threshold": 0.5
            },
            "retention": {
                "emotion_weight": 0.4,
                "recency_weight": 0.3,
                "access_weight": 0.3,
                "explicit_loyalty": 0.8,
                "explicit_importance": 0.9,
                "forget_decay_factor": 0.7
            }
        }

        if config_file:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    if config_file.endswith(('.yaml', '.yml')):
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)

                self._deep_update(default_config, loaded_config)
                self.logger.info(f"Configuration loaded from {config_file}.")
            except FileNotFoundError:
                self.logger.warning(f"Config file not found at {config_file}. Using default configuration.")
            except (yaml.YAMLError, json.JSONDecodeError) as e:
                self.logger.error(f"Error parsing config file {config_file}: {e}. Using default configuration.")
            except Exception as e:
                self.logger.error(f"Unexpected error loading configuration from {config_file}: {e}. Using default configuration.")

        return default_config

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively updates a dictionary `d` with values from dictionary `u`.

        Args:
            d: The dictionary to be updated.
            u: The dictionary containing new values.

        Returns:
            The updated dictionary `d`.)
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def _background_sync_loop(self) -> None:
        """Background thread loop for periodic synchronization."""
        while not self.sync_stop_event.is_set():
            try:
                self.sync_stop_event.wait(self.sync_interval)
                if self.sync_stop_event.is_set():
                    break # Exit loop if stop event is set

                # Acquire the main lock before calling sync()
                with self.lock:
                    self.logger.info("Executing background sync...")
                    self.sync()
                    self.last_sync_time = time.time()
                    self.logger.info("Background sync completed.")
            except Exception as e:
                self.logger.error(f"Error during background synchronization: {e}", exc_info=True)

    def process_input(self,
                      input_text: str,
                      embedding: Optional[np.ndarray] = None,
                      entities: Optional[Set[str]] = None,
                      emotional_valence: float = 0.0,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a text input, updates the context, and returns relevant context.
        This method is the primary entry point for integrating new user/system input
        into the system's contextual awareness module.

        Args:
            input_text: The incoming text string.
            embedding: Semantic embedding of the text.
            entities: Set of detected entities (e.g., proper nouns, keywords).
            emotional_valence: Detected emotional valence (-1.0 to 1.0).
            context: Additional contextual metadata.

        Returns:
            A dictionary containing relevant context for further processing.
        """
        with self.lock:
            self.active_context["last_activity_time"] = time.time()

            memory_commands = self._extract_memory_commands(input_text)

            content_for_stm = {
                "text": input_text,
                "timestamp": time.time(),
                "entities": list(entities) if entities else [],
                "raw_input_context": context,
                "processed_commands": [cmd["type"] for cmd in memory_commands]
            }
            if embedding is not None:
                content_for_stm["embedding"] = embedding
            else:
                # If no embedding is provided, store a placeholder or None.
                # A real system would generate an embedding here if necessary.
                content_for_stm["embedding"] = None

            if memory_commands:
                for cmd in memory_commands:
                    self._handle_memory_command(cmd, content_for_stm)
                self.logger.debug(f"Processed {len(memory_commands)} explicit memory commands.")

            tags_for_stm = entities if entities is not None else set()
            if memory_commands:
                tags_for_stm.add("memory_command_present")

            self.short_term.add(
                content=content_for_stm,
                emotional_valence=emotional_valence,
                tags=tags_for_stm,
                pin=False,
                embedding=embedding
            )
            self.logger.debug(f"Input added to Short-Term Memory. Total entries: {len(self.short_term.entries)}.")

            if entities:
                self.active_context["active_entities"].update(e.lower() for e in entities)
                # Cap active entities to avoid unbounded growth
                if len(self.active_context["active_entities"]) > self.config["short_term"].get("active_entities_limit", 50):
                    current_entities_list = list(self.active_context["active_entities"])
                    self.active_context["active_entities"] = set(current_entities_list[-self.config["short_term"].get("active_entities_limit", 50):])


            current_emotion = self.active_context["emotional_state"]
            self.active_context["emotional_state"] = current_emotion * 0.7 + emotional_valence * 0.3
            self.active_context["emotional_state"] = max(-1.0, min(1.0, self.active_context["emotional_state"]))

            if embedding is not None:
                stability = self._calculate_context_stability(embedding)
                self.active_context["context_stability"] = stability

                if stability < self.drift_threshold:
                    self.logger.info(f"Context drift detected (stability: {stability:.2f}, threshold: {self.drift_threshold:.2f}). Initializing context reboot.")
                    aggressive_threshold = self.drift_threshold / 2
                    if stability < aggressive_threshold:
                        self.logger.warning(f"Significant context drift detected (stability: {stability:.2f}). Performing aggressive context reboot.")
                        self._reboot_context(preserve_pinned=True)
                    else:
                        self.logger.info(f"Mild context drift detected. Current stability: {stability:.2f}. Not initiating aggressive reboot. Context will gradually drift.")

            relevant_context_result = self._retrieve_relevant_context(content_for_stm)

            return relevant_context_result

    def _extract_memory_commands(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts explicit memory commands from the text.
        This uses regex patterns to identify commands like "remember that...",
        "forget about...", "pin ...", or "mark as important...".

        Args:
            text: The text string to analyze.

        Returns:
            A list of dictionaries, each representing an extracted command with its type,
            content, and span (start/end indices in the original text).
        """
        commands = []

        patterns = [
            (r"(?:remember|lembrar|guardar)(?:\s+that|\s+que)?\s+(.+?)(?:.|$|please|por favor)", "remember"),
            (r"(?:forget|esquecer|remover)(?:\s+about|\s+o que falamos sobre)?\s+(.+?)(?:.|$|please|por favor)", "forget"),
            (r"(?:pin|fixar)(?:\s+in\s+memory|\s+na\s+memoria)?\s+(.+?)(?:.|$|please|por favor)", "pin"),
            (r"(?:mark\s+as\s+important|marcar\s+como\s+importante)\s+(.+?)(?:.|$|please|por favor)", "important")
        ]

        for pattern_str, cmd_type in patterns:
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                content = match.group(1).strip()
                commands.append({
                    "type": cmd_type,
                    "content": content,
                    "span": match.span()
                })

        return commands

    def _handle_memory_command(self, command: Dict[str, Any], original_input_content: Dict[str, Any]) -> None:
        """
        Processes an explicit memory command.
        This function gates the effects of memory commands to the actual memory subsystems.

        Args:
            command: The command dictionary (e.g., {"type": "remember", "content": "..."}).
            original_input_content: The full content dictionary of the input that contained the command.
        """
        cmd_type = command["type"]
        cmd_content = command["content"]

        embedding_from_input = original_input_content.get("embedding")
        emotional_valence_from_input = original_input_content.get("emotional_valence", 0.0)

        retention_config = self.config.get("retention", {})

        if cmd_type == "remember":
            importance = retention_config.get("explicit_importance", 0.8)
            loyalty = retention_config.get("explicit_loyalty", 0.7)
            self.long_term.add(
                content={"text": cmd_content, "source_input": original_input_content.get("text", "")},
                embedding=embedding_from_input,
                emotional_valence=emotional_valence_from_input,
                loyalty_coefficient=loyalty,
                importance=importance,
                tags={"explicit_memory", "user_requested", "remember_command"}
            )
            self.logger.info(f"Explicit memory added: '{cmd_content}' to LTM.")

        elif cmd_type == "forget":
            self.logger.info(f"Processing 'forget' command for: '{cmd_content}'")
            removed_from_stm = 0
            for entry_id, entry_obj in list(self.short_term.entries.items()):
                if cmd_content.lower() in str(entry_obj.content.get("text", "")).lower():
                    if self.short_term.remove(entry_id):
                        removed_from_stm += 1
            self.logger.debug(f"{removed_from_stm} entries matching '{cmd_content}' removed from STM.")

            decayed_in_ltm = 0
            # Use LTM's public API to ensure locking and saving
            ltm_entries_to_decay_ids = []
            with self.long_term._lock: # Acquire LTM's lock as we're iterating its entries directly
                for entry_id, entry_obj in list(self.long_term.entries.items()):
                    if cmd_content.lower() in str(entry_obj.content.get("text", "")).lower():
                        ltm_entries_to_decay_ids.append(entry_id)

            for entry_id in ltm_entries_to_decay_ids:
                entry_obj = self.long_term.get_full_entry(entry_id) # This re-acquires LTM lock and touches/saves
                if entry_obj:
                    decay_factor = retention_config.get("forget_decay_factor", 0.7)
                    entry_obj.decay_importance(decay_factor)

                    self.long_term.update(entry_id, importance=entry_obj.importance)
                    decayed_in_ltm += 1

                    if entry_obj.importance < self.long_term.importance_threshold:
                        if self.long_term.remove(entry_id):
                            self.logger.info(f"LTM entry {entry_id[:8]} completely removed after decay due to 'forget' command.")
            self.logger.debug(f"{decayed_in_ltm} entries affected in LTM matching '{cmd_content}'.")

        elif cmd_type == "pin":
            # Add to short-term memory as pinned. If it already exists, pin it.
            # For simplicity, we add a new entry. A more complex logic would find and pin existing.
            self.short_term.add(
                content={"text": cmd_content, "source_input": original_input_content.get("text", "")},
                emotional_valence=emotional_valence_from_input,
                pin=True,
                tags={"pinned", "user_requested", "pin_command"},
                embedding=embedding_from_input
            )
            self.logger.info(f"Entry pinned in Short-Term Memory: '{cmd_content}'.")

        elif cmd_type == "important":
            importance = retention_config.get("explicit_importance", 0.95)
            loyalty = retention_config.get("explicit_loyalty", 0.9)
            self.long_term.add(
                content={"text": cmd_content, "source_input": original_input_content.get("text", "")},
                embedding=embedding_from_input,
                emotional_valence=emotional_valence_from_input,
                loyalty_coefficient=loyalty,
                importance=importance,
                tags={"important", "user_requested", "important_command"}
            )
            self.logger.info(f"Important memory added to LTM: '{cmd_content}'.")

    def _retrieve_relevant_context(self, current_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves relevant context for the current content being processed.
        This function aggregates information from both short-term and long-term memory
        based on recency, entities, and semantic similarity.

        Args:
            current_content: The current content being processed.

        Returns:
            A dictionary containing relevant context categorized by source (short_term_memories, long_term_memories).
        """
        relevant_context = {
            "short_term_memories": [],
            "long_term_memories": [],
            "active_context_metadata": {
                "active_entities": list(self.active_context["active_entities"]),
                "emotional_state": self.active_context["emotional_state"],
                "context_stability": self.active_context["context_stability"],
                "last_activity_time": self.active_context["last_activity_time"]
            }
        }

        # 1. Retrieve recent entries from Short-Term Memory
        recent_stm_entries_content = self.short_term.search_recent(limit=self.config["short_term"].get("recent_search_limit", 5))
        if recent_stm_entries_content:
            relevant_context["short_term_memories"].extend([{"source": "recent_stm", "content": entry} for entry in recent_stm_entries_content])

        # 2. Retrieve by entities
        entities_in_current_content = set(current_content.get("entities", []))
        if entities_in_current_content:
            for entity in entities_in_current_content:
                stm_entity_matches = self.short_term.search_by_tag(entity)
                for entry_content in stm_entity_matches:
                    relevant_context["short_term_memories"].append({"source": "entity_stm", "content": entry_content, "matched_entity": entity})

                ltm_entity_matches = self.long_term.search_by_entity(entity, limit=self.config["long_term"].get("entity_search_limit", 3))
                for ltm_content in ltm_entity_matches:
                    relevant_context["long_term_memories"].append({"source": "entity_ltm", "content": ltm_content, "matched_entity": entity})

        # 3. Retrieve by semantic similarity
        if "embedding" in current_content and current_content["embedding"] is not None:
            current_embedding_np = np.array(current_content["embedding"])

            stm_similar_entries = self.short_term.search_by_embedding(
                query_embedding=current_embedding_np,
                threshold=self.config["short_term"].get("semantic_search_threshold", 0.6),
                limit=self.config["short_term"].get("semantic_search_result_limit", 3)
            )
            for entry_content, similarity in stm_similar_entries:
                relevant_context["short_term_memories"].append({"source": "semantic_stm", "content": entry_content, "similarity": float(similarity)})

            ltm_similar_entries = self.long_term.search_by_embedding(
                query_embedding=current_embedding_np,
                threshold=self.config["long_term"].get("semantic_search_threshold", 0.6),
                limit=self.config["long_term"].get("semantic_search_result_limit", 3)
            )
            for entry_content, similarity in ltm_similar_entries:
                relevant_context["long_term_memories"].append({"source": "semantic_ltm", "content": entry_content, "similarity": float(similarity)})

        def deduplicate_memories(memories_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen_texts = set()
            unique_memories = []
            # Iterate in reverse to keep newer/more relevant duplicates
            for mem_entry in reversed(memories_list):
                # Using a combination of content text and source for more robust deduplication
                text_content = mem_entry["content"].get("text", str(mem_entry["content"]))
                dedup_key = (text_content, mem_entry["source"])
                if text_content and dedup_key not in seen_texts:
                    seen_texts.add(dedup_key)
                    unique_memories.append(mem_entry)
            return list(reversed(unique_memories))

        relevant_context["short_term_memories"] = deduplicate_memories(relevant_context["short_term_memories"])
        relevant_context["long_term_memories"] = deduplicate_memories(relevant_context["long_term_memories"])

        return relevant_context

    def _calculate_context_stability(self, current_embedding: np.ndarray) -> float:
        """
        Calculates the stability of the current context by comparing it with recent entries.
        This provides a quantitative measure of how consistent the current topic/theme
        is with the immediate past interactions.

        Args:
            current_embedding: The embedding of the current content.

        Returns:
            A stability value (0.0 to 1.0), where 1.0 is highly stable and 0.0 is very unstable.
        """
        if not isinstance(current_embedding, np.ndarray):
            current_embedding = np.array(current_embedding)

        current_norm = np.linalg.norm(current_embedding)
        if current_norm < 1e-9:
            return 0.0
        current_embedding_normalized = current_embedding / current_norm

        recent_entries_contents = self.short_term.search_recent(limit=self.config["short_term"].get("stability_check_limit", 5))

        if not recent_entries_contents:
            return 1.0

        similarities = []

        for entry_content in recent_entries_contents:
            entry_embedding = entry_content.get("embedding")
            if entry_embedding is not None:
                if not isinstance(entry_embedding, np.ndarray):
                    entry_embedding = np.array(entry_embedding)

                entry_norm = np.linalg.norm(entry_embedding)
                if entry_norm > 1e-9:
                    entry_embedding_normalized = entry_embedding / entry_norm

                    similarity = np.dot(current_embedding_normalized, entry_embedding_normalized)
                    similarities.append(similarity)

        if not similarities:
            return 1.0

        mean_similarity = np.mean([max(0.0, sim) for sim in similarities])
        return float(mean_similarity)

    def _reboot_context(self, preserve_pinned: bool = True) -> None:
        """
        Reinitializes the active context state, optionally preserving pinned entries.
        This is typically triggered upon detection of significant context drift.

        Args:
            preserve_pinned: If True, short-term memory entries explicitly marked as pinned
                             will be re-added to the new short-term memory instance.
        """
        self.logger.info("Rebooting active context due to significant drift.")

        pinned_entries_data: List[Dict[str, Any]] = []
        if preserve_pinned:
            for entry_obj in self.short_term.entries.values():
                if entry_obj.is_pinned:
                    pinned_entries_data.append(entry_obj.to_dict())
            self.logger.debug(f"Found {len(pinned_entries_data)} pinned entries to preserve.")

        self.active_context["active_entities"].clear()
        self.active_context["emotional_state"] = 0.0
        self.active_context["context_stability"] = 1.0
        self.active_context["current_topic"] = None
        self.active_context["session_start_time"] = time.time()
        self.active_context["last_activity_time"] = time.time()

        self.short_term = ShortTermMemory(
            max_entries=self.config.get("short_term", {}).get("max_entries", 100),
            default_ttl=self.config.get("short_term", {}).get("default_ttl", 3600.0),
            eviction_policy=self.config.get("short_term", {}).get("eviction_policy", "lru")
        )
        self.logger.debug("Short-Term Memory reinitialized.")

        if preserve_pinned and pinned_entries_data:
            re_added_count = 0
            for entry_dict in pinned_entries_data:
                try:
                    re_created_entry = ShortTermMemoryEntry.from_dict(entry_dict)

                    self.short_term.add(
                        content=re_created_entry.content,
                        ttl=re_created_entry.ttl,
                        emotional_valence=re_created_entry.emotional_valence,
                        tags=re_created_entry.tags,
                        pin=True,
                        embedding=re_created_entry.embedding
                    )

                    if re_created_entry.content.get("entities"):
                        self.active_context["active_entities"].update(e.lower() for e in re_created_entry.content["entities"])
                    re_added_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to re-add pinned entry during reboot: {e}", exc_info=True)
            self.logger.info(f"Context reboot completed. {re_added_count} pinned entries restored.")
        else:
            self.logger.info("Context reboot completed. No pinned entries to restore or preservation was disabled.")

    def sync(self) -> Dict[str, Any]:
        """
        Synchronizes short-term and long-term memories.
        This process evaluates entries in short-term memory based on importance criteria
        (emotional valence, recency, access count, pinned status) and archives relevant ones to
        long-term memory. It also triggers cleanup of expired entries in short-term memory
        and purges low-importance entries from long-term memory.

        Returns:
            A dictionary with synchronization statistics.
        """
        with self.lock: # Ensure only one sync operation at a time
            self.logger.info("Starting memory synchronization.")

            sync_stats = {
                "archived_entries": 0,
                "removed_from_stm": 0,
                "expired_from_stm": 0,
                "purged_from_ltm": 0,
                "start_time": time.time()
            }

            current_stm_entries = list(self.short_term.entries.values())

            archive_threshold = self.config.get("sync", {}).get("archive_threshold", 0.5)
            retention_weights = self.config.get("retention", {})
            emotion_weight = retention_weights.get("emotion_weight", 0.4)
            recency_weight = retention_weights.get("recency_weight", 0.3)
            access_weight = retention_weights.get("access_weight", 0.3)

            current_time = time.time()

            entries_to_archive_ids: Set[str] = set()

            # --- Phase 1: Identify entries to archive from STM to LTM ---
            for entry in current_stm_entries:
                if entry.is_pinned:
                    entries_to_archive_ids.add(entry.id)
                    continue

                emotion_factor = abs(entry.emotional_valence)

                age_seconds = current_time - entry.timestamp
                max_age_for_recency = entry.ttl if entry.ttl is not None and entry.ttl > 0 else (3600.0 * 24.0 * 7.0) # Default to 1 week if no TTL
                recency_factor = max(0.0, 1.0 - (age_seconds / max_age_for_recency))

                access_threshold = self.config.get("short_term", {}).get("access_threshold", 5)
                access_factor = min(1.0, entry.access_count / access_threshold)

                combined_importance_score = (
                    emotion_weight * emotion_factor +
                    recency_weight * recency_factor +
                    access_weight * access_factor
                )

                if combined_importance_score >= archive_threshold:
                    entries_to_archive_ids.add(entry.id)

            # --- Phase 2: Archive identified entries to LTM ---
            for entry_id_to_archive in entries_to_archive_ids:
                entry_obj = self.short_term.get_full_entry(entry_id_to_archive)
                if not entry_obj: continue

                content_for_ltm = entry_obj.content.copy()
                content_for_ltm["archived_from_stm_id"] = entry_obj.id
                content_for_ltm["stm_access_count"] = entry_obj.access_count
                content_for_ltm["stm_emotional_valence"] = entry_obj.emotional_valence
                content_for_ltm["stm_is_pinned"] = entry_obj.is_pinned
                content_for_ltm["stm_timestamp"] = entry_obj.timestamp

                ltm_initial_importance = (emotion_weight * abs(entry_obj.emotional_valence) +
                                          recency_weight * (1.0 - (current_time - entry_obj.timestamp) / max_age_for_recency) +
                                          access_weight * min(1.0, entry_obj.access_count / access_threshold))
                if entry_obj.is_pinned:
                    ltm_initial_importance = max(ltm_initial_importance, self.config["retention"].get("explicit_importance", 0.8))

                loyalty_coefficient = min(0.9, 0.3 + (entry_obj.access_count * 0.1))

                self.long_term.add( # LTM add is already thread-safe
                    content=content_for_ltm,
                    embedding=entry_obj.embedding,
                    emotional_valence=entry_obj.emotional_valence,
                    loyalty_coefficient=loyalty_coefficient,
                    importance=ltm_initial_importance,
                    tags=entry_obj.tags
                )
                sync_stats["archived_entries"] += 1
                self.logger.debug(f"Archived STM entry {entry_obj.id[:8]} (importance: {ltm_initial_importance:.2f}) to LTM.")

                # --- Phase 3: Remove archived (and non-pinned) entries from STM ---
                if not entry_obj.is_pinned:
                    if self.short_term.remove(entry_obj.id):
                        sync_stats["removed_from_stm"] += 1

            # --- Phase 4: Perform cleanup of expired entries in STM ---
            sync_stats["expired_from_stm"] = self.short_term.cleanup()
            self.logger.debug(f"Cleaned up {sync_stats['expired_from_stm']} expired entries from STM.")

            # --- Phase 5: Purge low-importance entries from LTM ---
            # LTM purge is already thread-safe
            sync_stats["purged_from_ltm"] = self.long_term.purge_low_importance()
            self.logger.debug(f"Purged {sync_stats['purged_from_ltm']} low-importance entries from LTM.")

            sync_stats["elapsed_time"] = time.time() - sync_stats["start_time"]
            self.logger.info(
                f"Memory synchronization finished: {sync_stats['archived_entries']} archived, "
                f"{sync_stats['removed_from_stm']} removed from STM (not pinned), "
                f"{sync_stats['expired_from_stm']} expired from STM, "
                f"{sync_stats['purged_from_ltm']} purged from LTM. Elapsed: {sync_stats['elapsed_time']:.2f}s."
            )

            return sync_stats

    def explicit_remember(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Provides an explicit API for the system to remember something important.
        This bypasses the normal input processing flow and directly archives
        the provided text into long-term memory with high importance.

        Args:
            text: The text content to remember.
            context: Optional additional context/metadata for the memory entry.

        Returns:
            The ID of the created long-term memory entry.
        """
        with self.lock:
            content_for_ltm = {
                "text": text,
                "timestamp": time.time(),
                "explicit_memory_flag": True,
                "external_context": context
            }

            retention_config = self.config.get("retention", {})

            embedding_from_context = context.get("embedding") if context else None
            emotional_valence_from_context = context.get("emotional_valence", 0.5)

            entry_id = self.long_term.add( # LTM add is already thread-safe
                content=content_for_ltm,
                embedding=embedding_from_context,
                emotional_valence=emotional_valence_from_context,
                loyalty_coefficient=retention_config.get("explicit_loyalty", 0.8),
                importance=retention_config.get("explicit_importance", 0.9),
                tags={"explicit_memory", "system_requested", "important_user_input"}
            )

            self.logger.info(f"Explicit memory added: '{text[:50]}...' (ID: {entry_id[:8]}).")
            return entry_id

    def explicit_forget(self, query_text: str) -> int:
        """
        Provides an explicit API for the system to forget information.
        This attempts to remove entries from short-term memory and significantly
        reduce the importance of (or remove) entries in long-term memory that match
        the provided query text.

        Args:
            query_text: The text or pattern to forget (e.g., "details about project X").

        Returns:
            The total number of entries affected (removed from STM or importance reduced/removed from LTM).
        """
        with self.lock: # Ensure the forget operation is atomic
            affected_count = 0
            self.logger.info(f"Processing explicit 'forget' command for: '{query_text[:50]}...'.")

            retention_config = self.config.get("retention", {})

            # --- Phase 1: Remove from Short-Term Memory ---
            stm_entries_to_remove_ids = []
            for entry_id, entry_obj in list(self.short_term.entries.items()):
                if query_text.lower() in str(entry_obj.content.get("text", "")).lower():
                    stm_entries_to_remove_ids.append(entry_id)

            for entry_id in stm_entries_to_remove_ids:
                if self.short_term.remove(entry_id):
                    affected_count += 1
            self.logger.debug(f"Removed {len(stm_entries_to_remove_ids)} entries from STM matching '{query_text[:20]}'.")

            # --- Phase 2: Affect Long-Term Memory (decay importance or remove) ---
            ltm_entries_to_affect_ids = []
            with self.long_term._lock: # Temporarily acquire LTM's lock to safely get IDs
                for entry_id, entry_obj in list(self.long_term.entries.items()):
                    if query_text.lower() in str(entry_obj.content.get("text", "")).lower():
                        ltm_entries_to_affect_ids.append(entry_id)

            for entry_id in ltm_entries_to_affect_ids:
                entry_obj = self.long_term.get_full_entry(entry_id) # This call is thread-safe within LTM
                if entry_obj:
                    decay_factor = retention_config.get("forget_decay_factor", 0.7)
                    entry_obj.decay_importance(decay_factor)

                    self.long_term.update(entry_id, importance=entry_obj.importance) # This call is thread-safe within LTM
                    affected_count += 1

                    if entry_obj.importance < self.long_term.importance_threshold:
                        if self.long_term.remove(entry_id): # This call is thread-safe within LTM
                            self.logger.info(f"LTM entry {entry_id[:8]} completely removed after decay due to 'forget' command.")
            self.logger.debug(f"{len(ltm_entries_to_affect_ids)} entries affected in LTM matching '{query_text[:20]}'.")

            self.logger.info(f"Explicit 'forget' command processed. Total affected entries: {affected_count}.")
            return affected_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns general statistics for the context manager, including details
        from short-term and long-term memory, and the active context state.

        Returns:
            A dictionary with comprehensive statistics.
        """
        with self.lock:
            short_term_stats = self.short_term.get_stats()
            long_term_stats = self.long_term.get_stats()

            session_duration_seconds = time.time() - self.active_context["session_start_time"]

            active_context_stats = {
                "active_entities_count": len(self.active_context["active_entities"]),
                "current_emotional_state": self.active_context["emotional_state"],
                "current_context_stability": self.active_context["context_stability"],
                "session_duration_seconds": session_duration_seconds
            }

            full_stats = {
                "short_term_memory": short_term_stats,
                "long_term_memory": long_term_stats,
                "current_active_context": active_context_stats,
                "manager_status": {
                    "last_sync_time": self.last_sync_time,
                    "next_sync_in_seconds": max(0, self.sync_interval - (time.time() - self.last_sync_time))
                },
                "timestamp_report": time.time()
            }
            return full_stats
