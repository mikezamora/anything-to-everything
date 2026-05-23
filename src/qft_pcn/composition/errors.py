"""Exception family for sub-project I (lemma library and promotion). Spec §6."""
from __future__ import annotations


class CompositionError(Exception):
    """Base of every sub-project I error."""


class LemmaHashCollision(CompositionError):
    """save() of a content hash that already maps to different content."""


class LemmaNotFound(CompositionError):
    """load/materialize/compile_constraint of an unknown lemma_id."""


class LemmaLeafCountMismatch(CompositionError):
    """use_lemma 'leaves' length != lemma n_leaves_L."""


class LemmaSpeciesMismatch(CompositionError):
    """Re-indexing onto a host window with a misaligned species pattern."""


class ConditionalLemmaRefused(CompositionError):
    """compile_constraint of a conditional lemma without allow_conditional."""


class LemmaValidationError(CompositionError):
    """Registration validation pass failed (surfaced via RegistrationResult)."""


class CompressionError(CompositionError):
    """Bond compression could not reach eps_compress without exceeding it."""
