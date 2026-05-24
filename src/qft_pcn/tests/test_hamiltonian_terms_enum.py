"""Hamiltonian.terms metadata enumeration regression test.

For a two-species Hamiltonian with one Yukawa coupling, the `terms`
attribute must enumerate at least the expected one-site mass term per
species per site plus the Yukawa cross-species term per site plus the
two-site kinetic terms.
"""

from src.qft_pcn.qft.hamiltonian import (
    FieldSpecies, Hamiltonian, HamiltonianConfig,
)


def test_terms_present_for_two_species_with_yukawa():
    species = [
        FieldSpecies(name="A", cutoff=2, bare_mass=1.0, kinetic=0.5,
                     quartic=0.0, source=0.0),
        FieldSpecies(name="B", cutoff=2, bare_mass=0.8, kinetic=0.3,
                     quartic=0.1, source=0.05),
    ]
    cfg = HamiltonianConfig(
        species=species,
        yukawa_couplings={("A", "B"): 0.2},
    )
    H = Hamiltonian(cfg, N=3)

    assert hasattr(H, "terms")
    assert isinstance(H.terms, list)
    assert len(H.terms) >= 4

    kinds = {t["kind"] for t in H.terms}
    # Expected kinds for this configuration.
    expected = {"mass", "kinetic", "quartic", "source", "yukawa"}
    assert expected.issubset(kinds), (
        f"missing kinds: {expected - kinds}; got {kinds}"
    )

    # One mass term per species per site = 2 * 3 = 6.
    mass_terms = [t for t in H.terms if t["kind"] == "mass"]
    assert len(mass_terms) == 6

    # One Yukawa term per site = 3.
    yukawa_terms = [t for t in H.terms if t["kind"] == "yukawa"]
    assert len(yukawa_terms) == 3
    for t in yukawa_terms:
        assert t["species"] == ("A", "B")
        assert t["coeff"] == 0.2

    # Bond kinetic terms: (N-1) bonds * 2 species = 4.
    kinetic_terms = [t for t in H.terms if t["kind"] == "kinetic"]
    assert len(kinetic_terms) == 4
    for t in kinetic_terms:
        assert isinstance(t["site"], tuple) and len(t["site"]) == 2


def test_terms_empty_kinetic_when_kinetic_is_zero():
    species = [FieldSpecies(name="A", cutoff=2, bare_mass=1.0,
                            kinetic=0.0)]
    cfg = HamiltonianConfig(species=species)
    H = Hamiltonian(cfg, N=2)
    assert not any(t["kind"] == "kinetic" for t in H.terms)
