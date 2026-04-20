"""Parse Instagram's ``rur`` cookie — region + staleness + continent-match.

Format (as stored by the IG web frontend, literal backslashes):

    "RVA\\054<user_id>\\054<expires_epoch>:01<hmac_signature>"

``RVA`` / ``CLN`` / ``FRC`` / etc. is a 3-letter IG edge-PoP code. The
second field is the account's numeric ID. The third field is an epoch
timestamp — NOT the cookie's expiry, but the edge's issued freshness
marker. If this timestamp is older than ~48h the session is about to
rotate; re-extract cookies from the browser before running automation
or the first API call will force a `/login`.

Used at startup to:
  * warn when proxy country doesn't match the edge continent
  * refuse to start when ``rur`` freshness is stale (configurable)
  * decode the account's claimed home edge for diagnostic output
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


# 2026-current IG edge-PoP codes → continent. Keep loose — "continent match"
# is the actual signal the edge checks, not the exact city.
REGION_TO_CONTINENT: dict[str, str] = {
    # North America
    "ATN": "NA", "RVA": "NA", "HIL": "NA", "PRN": "NA", "ASH": "NA",
    "SJC": "NA", "LLA": "NA", "NAM": "NA",
    # Europe
    "CLN": "EU", "FRC": "EU", "FRA": "EU", "VLL": "EU", "DUB": "EU",
    "LON": "EU", "AMS": "EU", "EUR": "EU",
    # Asia-Pacific
    "NAO": "AS", "ODN": "AS", "NCG": "AS", "SIN": "AS", "HKG": "AS",
    "TYO": "AS", "SYD": "AU",
    # Africa / LatAm
    "LDC": "AF", "NBO": "AF", "GRU": "SA", "SAO": "SA",
}


# ISO 3166-1 alpha-2 country → continent. Only the common ones a user is
# likely to set via IG_COUNTRY_CODE — keep list short + legible.
COUNTRY_TO_CONTINENT: dict[str, str] = {
    "US": "NA", "CA": "NA", "MX": "NA",
    "GB": "EU", "IE": "EU", "DE": "EU", "FR": "EU", "IT": "EU", "ES": "EU",
    "NL": "EU", "BE": "EU", "SE": "EU", "NO": "EU", "DK": "EU", "FI": "EU",
    "CH": "EU", "AT": "EU", "PL": "EU", "PT": "EU", "GR": "EU", "CZ": "EU",
    "JP": "AS", "KR": "AS", "SG": "AS", "HK": "AS", "TW": "AS",
    "IN": "AS", "MY": "AS", "TH": "AS", "ID": "AS", "PH": "AS", "VN": "AS",
    "CN": "AS", "AE": "AS", "IL": "AS", "TR": "AS",
    "AU": "AU", "NZ": "AU",
    "BR": "SA", "AR": "SA", "CL": "SA", "CO": "SA", "PE": "SA",
    "ZA": "AF", "NG": "AF", "EG": "AF", "KE": "AF", "MA": "AF",
}


@dataclass(frozen=True)
class RurInfo:
    region: str                 # e.g. "RVA"
    continent: str              # e.g. "NA" (or "?" when unknown)
    user_id: str                # numeric IG user id
    issued_epoch: int | None    # epoch seconds, None if malformed
    age_hours: float | None     # now - issued, None if unknown
    raw: str                    # original cookie value

    @property
    def is_stale(self) -> bool:
        """True when the rur was issued >48h ago. Sessions nearing rotation
        routinely force a /login on the first API call."""
        return self.age_hours is not None and self.age_hours > 48.0


def parse_rur(raw: str) -> RurInfo | None:
    """Parse the rur cookie value. Returns None on malformed input.

    Accepts both forms the value can appear in:
      * Cookie-Editor JSON export: literal ``\\054`` sequences (the seller's
        export)
      * Live cookie jar from requests: actual commas (``,``) after the jar
        deserialiser replaced ``\\054``

    Both decode to the same logical tuple.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().strip('"')
    # Normalise both encodings to a common comma form
    s = s.replace(r"\054", ",").replace(chr(0x2C), ",")  # defensive: \054 → ,
    # Split on commas; first 3 fields are region, user_id, (timestamp:sig)
    parts = s.split(",")
    if len(parts) < 3:
        return None
    region = parts[0].strip().upper()
    user_id = parts[1].strip()
    remainder = ",".join(parts[2:])  # in case signature had commas (rare)
    # remainder is "<epoch>:01<hmac>"
    epoch_str = remainder.split(":", 1)[0].strip()
    issued: int | None = None
    age_h: float | None = None
    try:
        issued = int(epoch_str)
        if 10**9 <= issued <= 10**11:  # sanity: 2001..5138
            now = int(datetime.now(timezone.utc).timestamp())
            age_h = (now - issued) / 3600.0
        else:
            issued = None
    except ValueError:
        issued = None

    continent = REGION_TO_CONTINENT.get(region, "?")
    return RurInfo(
        region=region,
        continent=continent,
        user_id=user_id,
        issued_epoch=issued,
        age_hours=age_h,
        raw=raw,
    )


def continent_matches(rur: RurInfo, country_code: str | None) -> bool:
    """True when the user's declared country (IG_COUNTRY_CODE or IG_PROXY
    region) shares a continent with the rur edge. Unknowns fail open
    (``True``) so we don't block legitimate setups on unknown region codes.
    """
    if rur.continent == "?":
        return True  # unknown region — can't judge
    if not country_code:
        return True  # user hasn't declared — don't block
    cc = country_code.strip().upper()[:2]
    user_cont = COUNTRY_TO_CONTINENT.get(cc)
    if user_cont is None:
        return True  # unknown country — don't block
    return user_cont == rur.continent
