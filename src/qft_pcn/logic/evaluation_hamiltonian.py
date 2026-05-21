"""Evaluation Hamiltonian for the QPCN logic layer (sub-project C).

See docs/superpowers/specs/2026-05-21-eval-hamiltonian-design.md.

Per spec §1.1 + §1.2: STRUCTURAL Hamiltonian, depends only on N. No
AST is touched here; the encoder produces the input MPS and the
spectrum drives reduction via imaginary-time evolution (see
factored_evolution.py). Per controller resolution: mirrors B's
TypingHamiltonian shape — .terms / .term_energy / .residuals /
.total_energy — and works through the factored-expectation primitives
in _factored_expectation.py. The dense 65536^2 operator is NEVER
materialized (manifesto Temptation 3).

Phase 1 ships the structural-collapse projector terms:
  - R_BETA       : P_APP(k) o+ P_LAM(k+1)         (spec §3.2)
  - R_ARITH_PRE  : P_BIN_arith(k) o+ P_INT(k+1)   (spec §3.3)
  - R_ARITH_POST : P_INT(k+1) o+ P_INT(k+2)       (spec §3.3)
  - R_CMP_PRE    : P_BIN_cmp(k) o+ P_INT(k+1)     (spec §3.4)
  - R_IF         : P_IF(k) o+ P_BOOL(k+1)         (spec §3.5)

The off-diagonal transition couplings that DRIVE reduction live in a
separate phase (see factored_evolution.py future work). The Phase 1
deliverable is constructibility + nonzero-on-redex + zero-on-normal-form
+ composability with TypingHamiltonian.

Avoiding the manifesto's banned identifiers: this module never refers
to the AST module, never uses a Python beta-reduction or substitution
routine, and never inspects per-program state during term construction.
The static guard test enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.qft_pcn.qft.mps import MPS
from .encoding import SPECIES
from ._factored_expectation import (
    factored_two_site_expectation,
    factored_two_site_expectation_cached,
    build_envs,
)
from ._eval_terms import (
    beta_factors,
    arith_pre_factors,
    arith_post_factors,
    cmp_pre_factors,
    if_redex_factors,
    DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH, DEFAULT_LAMBDA_IF,
)


# ---- Rule names ---------------------------------------------------------

RULE_R_BETA = "R-Beta"
RULE_R_ARITH_PRE = "R-Arith-Pre"
RULE_R_ARITH_POST = "R-Arith-Post"
RULE_R_CMP_PRE = "R-Cmp-Pre"
RULE_R_IF = "R-If"

_ALL_RULES = (
    RULE_R_BETA,
    RULE_R_ARITH_PRE,
    RULE_R_ARITH_POST,
    RULE_R_CMP_PRE,
    RULE_R_IF,
)


# ---- Exceptions ---------------------------------------------------------


class EvalHamiltonianError(Exception):
    """Base error for EvalHamiltonian operations."""


class EvalTermNotFound(EvalHamiltonianError):
    def __init__(self, term):
        super().__init__(f"term {term} not in this Hamiltonian's term list")


# ---- EvalTerm -----------------------------------------------------------


@dataclass(frozen=True)
class EvalTerm:
    """A single evaluation-rule term, addressable by (rule_id, site, arity).

    arity=2 for all current rules: each is a two-site coupling at bond
    (site, site+1).
    """
    rule_id: str
    site: int
    arity: int


# ---- Per-rule factored-expectation dispatch -----------------------------


def _two_site_energy(state, site, op_l, op_r, envs):
    if envs is not None:
        return factored_two_site_expectation_cached(
            state, site, op_l, op_r, envs[0], envs[1])
    return factored_two_site_expectation(state, site, op_l, op_r)


def _energy_r_beta(state, site, envs, lam_scale):
    op_l, op_r = beta_factors(lam_scale)
    return _two_site_energy(state, site, op_l, op_r, envs)


def _energy_r_arith_pre(state, site, envs, lam_scale):
    op_l, op_r = arith_pre_factors(lam_scale)
    return _two_site_energy(state, site, op_l, op_r, envs)


def _energy_r_arith_post(state, site, envs, lam_scale):
    op_l, op_r = arith_post_factors(lam_scale)
    return _two_site_energy(state, site, op_l, op_r, envs)


def _energy_r_cmp_pre(state, site, envs, lam_scale):
    op_l, op_r = cmp_pre_factors(lam_scale)
    return _two_site_energy(state, site, op_l, op_r, envs)


def _energy_r_if(state, site, envs, lam_scale):
    op_l, op_r = if_redex_factors(lam_scale)
    return _two_site_energy(state, site, op_l, op_r, envs)


# ---- The Hamiltonian class ---------------------------------------------


class EvalHamiltonian:
    """Sum-of-terms evaluation Hamiltonian for an N-site lattice.

    Per spec §1.2, depends only on N. The SAME instance evaluates
    correctly on any MPS encoding an AST of size <= N.

    Term enumeration (per spec §3 + Phase 1 scope):
      - R_BETA       : site k in [0, N-1)
      - R_ARITH_PRE  : site k in [0, N-1)
      - R_ARITH_POST : site k in [0, N-1)
      - R_CMP_PRE    : site k in [0, N-1)
      - R_IF         : site k in [0, N-1)

    Total: 5 * (N - 1).
    """

    SPECIES = SPECIES

    def __init__(self, N: int,
                 lambda_beta: float = DEFAULT_LAMBDA_BETA,
                 lambda_arith: float = DEFAULT_LAMBDA_ARITH,
                 lambda_if: float = DEFAULT_LAMBDA_IF):
        self.N = int(N)
        self.lambda_beta = float(lambda_beta)
        self.lambda_arith = float(lambda_arith)
        self.lambda_if = float(lambda_if)
        self.terms: list[EvalTerm] = self._enumerate_terms()
        self._terms_set = frozenset(self.terms)

    def _enumerate_terms(self) -> list[EvalTerm]:
        terms: list[EvalTerm] = []
        for rule in _ALL_RULES:
            for k in range(self.N - 1):
                terms.append(EvalTerm(rule_id=rule, site=k, arity=2))
        return terms

    def _lambda_for(self, rule_id: str) -> float:
        if rule_id == RULE_R_BETA:
            return self.lambda_beta
        if rule_id in (RULE_R_ARITH_PRE, RULE_R_ARITH_POST, RULE_R_CMP_PRE):
            return self.lambda_arith
        if rule_id == RULE_R_IF:
            return self.lambda_if
        raise EvalTermNotFound(rule_id)

    def term_energy(self, state: MPS, term: EvalTerm,
                    envs=None) -> float:
        if term not in self._terms_set:
            raise EvalTermNotFound(term)
        rule = term.rule_id
        site = term.site
        lam = self._lambda_for(rule)
        if rule == RULE_R_BETA:
            return _energy_r_beta(state, site, envs, lam)
        if rule == RULE_R_ARITH_PRE:
            return _energy_r_arith_pre(state, site, envs, lam)
        if rule == RULE_R_ARITH_POST:
            return _energy_r_arith_post(state, site, envs, lam)
        if rule == RULE_R_CMP_PRE:
            return _energy_r_cmp_pre(state, site, envs, lam)
        if rule == RULE_R_IF:
            return _energy_r_if(state, site, envs, lam)
        raise EvalTermNotFound(term)

    def total_energy(self, state: MPS) -> float:
        envs = build_envs(state)
        return sum(self.term_energy(state, t, envs=envs) for t in self.terms)

    def residuals(self, state: MPS) -> dict:
        envs = build_envs(state)
        return {(t.rule_id, t.site): self.term_energy(state, t, envs=envs)
                for t in self.terms}
