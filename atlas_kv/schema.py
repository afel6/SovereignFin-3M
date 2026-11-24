"""Vocabulary of knowledge-graph relations."""

from enum import StrEnum


class Relation(StrEnum):
    """Non-numeric relations only."""

    operates_in_sector = "operates_in_sector"
    headquartered_in = "headquartered_in"
    has_segment = "has_segment"
    offers_product = "offers_product"
    competes_with = "competes_with"
    faces_risk = "faces_risk"
    uses_technology = "uses_technology"
    has_subsidiary = "has_subsidiary"
    listed_on = "listed_on"
    supplied_by = "supplied_by"
    pays_late_to = "pays_late_to"
    shares_vendor_with = "shares_vendor_with"
