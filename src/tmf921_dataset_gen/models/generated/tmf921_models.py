from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TMFBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class IntentExpression(TMFBaseModel):
    expression_type: str = Field(alias="@type")
    iri: str
    base_type: str | None = Field(default=None, alias="@baseType")
    schema_location: str | None = Field(default=None, alias="@schemaLocation")


class JsonLdExpression(IntentExpression):
    expressionValue: dict[str, Any]


class TurtleExpression(IntentExpression):
    expressionValue: str


class EntityRef(TMFBaseModel):
    id: str | None = None
    href: str | None = None
    name: str | None = None
    referred_type: str | None = Field(default=None, alias="@referredType")
    expression_type: str | None = Field(default=None, alias="@type")


class Characteristic(TMFBaseModel):
    id: str | None = None
    name: str | None = None
    value: Any = None
    valueType: str | None = None
    expression_type: str | None = Field(default=None, alias="@type")


class PartyRefOrPartyRoleRef(TMFBaseModel):
    href: str | None = None
    id: str | None = None
    name: str | None = None
    referred_type: str | None = Field(default=None, alias="@referredType")
    expression_type: str | None = Field(default=None, alias="@type")


class RelatedPartyRefOrPartyRoleRef(TMFBaseModel):
    role: str | None = None
    partyOrPartyRole: PartyRefOrPartyRoleRef | dict[str, Any] | None = None
    expression_type: str | None = Field(default=None, alias="@type")


class Attachment(TMFBaseModel):
    id: str | None = None
    href: str | None = None
    name: str | None = None
    expression_type: str | None = Field(default=None, alias="@type")


class AttachmentRef(TMFBaseModel):
    id: str | None = None
    href: str | None = None
    name: str | None = None
    referred_type: str | None = Field(default=None, alias="@referredType")
    expression_type: str | None = Field(default=None, alias="@type")


class Hub(TMFBaseModel):
    callback: str
    query: str | None = None
    expression_type: str = Field(alias="@type")
    base_type: str | None = Field(default=None, alias="@baseType")
    schema_location: str | None = Field(default=None, alias="@schemaLocation")


class BaseEvent(TMFBaseModel):
    eventId: str | None = None
    eventTime: str | None = None
    eventType: str | None = None
    correlationId: str | None = None
    domain: str | None = None
    title: str | None = None
    description: str | None = None
    priority: str | None = None
    timeOcurred: str | None = None
    expression_type: str | None = Field(default=None, alias="@type")


class IntentCreateEvent(BaseEvent):
    event: dict[str, Any] | None = None


class IntentStatusChangeEvent(BaseEvent):
    event: dict[str, Any] | None = None


class IntentFVO(TMFBaseModel):
    expression_type: str = Field(alias="@type")
    name: str
    expression: JsonLdExpression | TurtleExpression | IntentExpression
    description: str | None = None
    priority: str | None = None
    context: str | None = None
    version: str | None = None
    intentSpecification: EntityRef | dict[str, Any] | None = None
    characteristic: list[Characteristic | dict[str, Any]] | None = None
    relatedParty: list[RelatedPartyRefOrPartyRoleRef | dict[str, Any]] | None = None
    attachment: list[Attachment | AttachmentRef | dict[str, Any]] | None = None

    @field_validator("expression", mode="before")
    @classmethod
    def validate_expression(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        expression_type = value.get("@type")
        if expression_type == "JsonLdExpression":
            return JsonLdExpression.model_validate(value)
        if expression_type == "TurtleExpression":
            return TurtleExpression.model_validate(value)
        return IntentExpression.model_validate(value)
