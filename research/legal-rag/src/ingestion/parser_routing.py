"""Parser routing and batching policies for structural ingestion."""

from __future__ import annotations

from dataclasses import dataclass

from src.common.schemas import ParserRoute


@dataclass(frozen=True, slots=True)
class DocumentStructuralProfile:
    """Signals used to route a document through parser branches."""

    has_struct_tree: bool
    born_digital: bool
    layout_complex: bool


@dataclass(frozen=True, slots=True)
class ParserAvailability:
    """Runtime availability of structural parser capabilities."""

    opendataloader_available: bool
    hybrid_backend_healthy: bool


@dataclass(frozen=True, slots=True)
class RouteBatch:
    """A homogeneous parser route batch for invocation planning."""

    route: ParserRoute
    doc_ids: list[str]


class ParserRoutingError(RuntimeError):
    """Raised when no structural parser route can satisfy the ingestion contract."""


def resolve_parser_route(profile: DocumentStructuralProfile, availability: ParserAvailability) -> ParserRoute:
    """Resolve the parser route for one document."""

    if not availability.opendataloader_available:
        raise ParserRoutingError("OpenDataLoader is required for structural ingestion but is unavailable.")

    if profile.has_struct_tree:
        return ParserRoute.TAGGED

    if profile.born_digital or profile.layout_complex:
        if availability.hybrid_backend_healthy:
            return ParserRoute.HYBRID
        return ParserRoute.LOCAL_ONLY

    return ParserRoute.LOCAL_ONLY


def plan_route_batches(routes_by_doc_id: dict[str, ParserRoute]) -> list[RouteBatch]:
    """Group documents into homogeneous route batches."""

    route_order = (
        ParserRoute.TAGGED,
        ParserRoute.HYBRID,
        ParserRoute.LOCAL_ONLY,
    )
    batches: list[RouteBatch] = []
    for route in route_order:
        doc_ids = [doc_id for doc_id, resolved_route in routes_by_doc_id.items() if resolved_route is route]
        if doc_ids:
            batches.append(RouteBatch(route=route, doc_ids=doc_ids))
    return batches
