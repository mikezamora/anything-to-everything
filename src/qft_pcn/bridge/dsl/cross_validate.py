"""Phase 3 of validation: cross-field references.

Assumes the input has already passed phase-2 JSON Schema validation
(bridge/dsl/schema.py::validate_schema). Raises BadReferenceError with
a JSON Pointer naming the bad reference.
"""

from __future__ import annotations

from typing import Any

from ..errors import BadReferenceError


def cross_validate(dsl: dict[str, Any]) -> None:
    sites: int = dsl["sites"]
    field_names = {f["name"] for f in dsl["fields"]}

    for i, c in enumerate(dsl.get("constraints", [])):
        if c["kind"] == "local":
            if not 0 <= c["site"] < sites:
                raise BadReferenceError(
                    message=f"constraint {i} site {c['site']} out of [0, {sites-1}] (pointer /constraints/{i}/site)",
                    details={"pointer": f"/constraints/{i}/site",
                             "value": c["site"], "sites": sites},
                )
        else:  # two_site
            for j, s in enumerate(c["sites"]):
                if not 0 <= s < sites:
                    raise BadReferenceError(
                        message=f"constraint {i} sites[{j}]={s} out of [0, {sites-1}] (pointer /constraints/{i}/sites/{j})",
                        details={"pointer": f"/constraints/{i}/sites/{j}",
                                 "value": s, "sites": sites},
                    )

    for i, o in enumerate(dsl["observables"]):
        if not 0 <= o["site"] < sites:
            raise BadReferenceError(
                message=f"observable {i} site {o['site']} out of [0, {sites-1}] (pointer /observables/{i}/site)",
                details={"pointer": f"/observables/{i}/site",
                         "value": o["site"], "sites": sites},
            )
        if o["field"] not in field_names:
            raise BadReferenceError(
                message=f"observable {i} field {o['field']!r} not declared (pointer /observables/{i}/field)",
                details={"pointer": f"/observables/{i}/field",
                         "value": o["field"],
                         "known_fields": sorted(field_names)},
            )

    for site_key, fmap in (dsl.get("boundary") or {}).items():
        try:
            site = int(site_key)
        except ValueError:
            raise BadReferenceError(
                message=f"boundary key {site_key!r} is not an integer (pointer /boundary/{site_key})",
                details={"pointer": f"/boundary/{site_key}"},
            )
        if not 0 <= site < sites:
            raise BadReferenceError(
                message=f"boundary site {site} out of [0, {sites-1}] (pointer /boundary/{site_key})",
                details={"pointer": f"/boundary/{site_key}",
                         "value": site, "sites": sites},
            )
        for fname in fmap:
            if fname not in field_names:
                raise BadReferenceError(
                    message=f"boundary site {site} field {fname!r} not declared (pointer /boundary/{site_key}/{fname})",
                    details={"pointer": f"/boundary/{site_key}/{fname}",
                             "value": fname,
                             "known_fields": sorted(field_names)},
                )
