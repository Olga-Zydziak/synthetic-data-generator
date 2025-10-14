"""Domain models and enumerations for fraudforge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

import pandas as pd

__all__ = [
    "FraudType",
    "Channel",
    "Region",
    "AgeBand",
    "Currency",
    "DeviceType",
    "TransactionRecord",
]


class StrEnum(str, Enum):
    """String enumeration base class."""

    def __str__(self) -> str:
        return str(self.value)


class FraudType(StrEnum):
    """Enumeration of supported fraud types."""

    CARD_NOT_PRESENT = "CARD_NOT_PRESENT"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    SKIMMING = "SKIMMING"
    AUTHORIZED_PUSH_PAYMENT = "AUTHORIZED_PUSH_PAYMENT"

    CARD_PRESENT_CLONED = "CARD_PRESENT_CLONED"
    SYNTHETIC_IDENTITY = "SYNTHETIC_IDENTITY"
    FRIENDLY_FRAUD = "FRIENDLY_FRAUD"
    MONEY_MULE = "MONEY_MULE"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"



class Channel(StrEnum):
    """Enumeration of customer interaction channels."""

    APP = "APP"
    WEB = "WEB"
    ATM = "ATM"
    POS = "POS"
    WIRE = "WIRE"


class Region(StrEnum):
    """Enumeration of bank operating regions."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


class AgeBand(StrEnum):
    """Enumeration of customer age bands."""

    A18_25 = "A18_25"
    A26_35 = "A26_35"
    A36_50 = "A36_50"
    A50_PLUS = "A50_PLUS"


class Currency(StrEnum):
    """Enumeration of supported currencies."""

    USD = "USD"


DeviceType = Literal["mobile", "desktop", "pos", "atm"]


@dataclass(slots=True)
class TransactionRecord:
    """Structured representation of a transaction row."""

    transaction_id: str
    event_time: pd.Timestamp
    customer_id: str
    account_id: str
    age_band: AgeBand
    region: Region
    account_tenure_days: int
    channel: Channel
    device_id: str
    device_type: DeviceType
    os: str
    app_version: str
    ip: str
    merchant_id: str
    merchant_category: str
    merchant_country: str
    amount: float
    currency: Currency
    txns_last_24h: int
    avg_amount_7d: float
    chargeback_count_90d: int
    is_fraud: bool
    fraud_type: FraudType | None
    is_causal_fraud: bool
    scenario: str
    is_dirty: bool
    dirty_issues: list[str]

    alias_is_casual_fraud: ClassVar[str] = "is_casual_fraud"

    def to_dict(self) -> dict[str, object]:
        """Convert record to dictionary with alias column.

        Returns:
            dict[str, object]: Record dictionary.
        """

        data = {
            "transaction_id": self.transaction_id,
            "event_time": self.event_time,
            "customer_id": self.customer_id,
            "account_id": self.account_id,
            "age_band": self.age_band.value,
            "region": self.region.value,
            "account_tenure_days": self.account_tenure_days,
            "channel": self.channel.value,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "os": self.os,
            "app_version": self.app_version,
            "ip": self.ip,
            "merchant_id": self.merchant_id,
            "merchant_category": self.merchant_category,
            "merchant_country": self.merchant_country,
            "amount": self.amount,
            "currency": self.currency.value,
            "txns_last_24h": self.txns_last_24h,
            "avg_amount_7d": self.avg_amount_7d,
            "chargeback_count_90d": self.chargeback_count_90d,
            "is_fraud": self.is_fraud,
            "fraud_type": self.fraud_type.value if self.fraud_type else None,
            "is_causal_fraud": self.is_causal_fraud,
            "scenario": self.scenario,
            "is_dirty": self.is_dirty,
            "dirty_issues": self.dirty_issues,
        }
        data[self.alias_is_casual_fraud] = self.is_causal_fraud
        return data
