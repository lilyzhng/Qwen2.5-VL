"""
Custom exceptions for the ALFA 0.1 retrieval system.
"""


class VideoRetrievalError(Exception):
    """Base exception for ALFA 0.1 retrieval."""
    pass


class VideoNotFoundError(VideoRetrievalError):
    """Raised when a video file is not found."""
    pass


class VideoLoadError(VideoRetrievalError):
    """Raised when a video cannot be loaded or processed."""
    pass


class InvalidVideoFormatError(VideoRetrievalError):
    """Raised when video format is not supported."""
    pass


class DatabaseError(VideoRetrievalError):
    """Base exception for database-related errors."""
    pass


class DatabaseNotFoundError(DatabaseError):
    """Raised when database file is not found."""
    pass


class DatabaseCorruptedError(DatabaseError):
    """Raised when database is corrupted or cannot be loaded."""
    pass


class EmbeddingError(VideoRetrievalError):
    """Base exception for embedding-related errors."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when model cannot be loaded."""
    pass


class EmbeddingExtractionError(EmbeddingError):
    """Raised when embedding extraction fails."""
    pass


class SearchError(VideoRetrievalError):
    """Base exception for search-related errors."""
    pass


class InvalidQueryError(SearchError):
    """Raised when search query is invalid."""
    pass


class NoResultsError(SearchError):
    """Raised when search returns no results."""
    pass


class ConfigurationError(VideoRetrievalError):
    """Raised when configuration is invalid."""
    pass
