from __future__ import annotations

import pricing


def test_member_price_rounds_half_up():
    got = pricing.member_price_cents("PRODUCT-2", 50)
    assert got == 500, f"member price must round half-up (expected 500, got {got})"


def test_member_price_when_no_rounding_is_needed():
    assert pricing.member_price_cents("PRODUCT-0", 10) == 1799
