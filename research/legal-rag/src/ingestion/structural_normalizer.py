"""Normalize raw structural sidecars into legal-friendly canonical blocks."""

from __future__ import annotations

import re

from src.common.schemas import NormalizedStructuralBlock, RawParseNode, RawParseSidecar

_GLYPH_PATTERN = re.compile(r"glyph<[^>]*>")
_COMMAND_PATTERN = re.compile(r"/([a-z]+)(?:\.[a-z]+)*")
_COMMAND_REPLACEMENTS = {
    "comma": ",",
    "period": ".",
    "colon": ":",
    "hyphen": "-",
    "space": " ",
}


def normalize_structural_sidecar(sidecar: RawParseSidecar) -> list[NormalizedStructuralBlock]:
    """Normalize parser-native nodes into canonical structural blocks."""

    blocks: list[NormalizedStructuralBlock] = []
    heading_stack: list[str] = []

    for node in sidecar.nodes:
        cleaned_text = _clean_text(node.text)
        if node.kind == "heading":
            heading_stack = _update_heading_stack(heading_stack, cleaned_text, node.heading_level)
            blocks.append(
                NormalizedStructuralBlock(
                    source_block_id=node.node_id,
                    page_number=node.page_number,
                    text=cleaned_text,
                    heading_path=list(heading_stack),
                    block_roles=["heading"],
                    linked_content_id=node.linked_content_id,
                )
            )
            continue

        blocks.append(
            NormalizedStructuralBlock(
                source_block_id=node.node_id,
                page_number=node.page_number,
                text=cleaned_text,
                heading_path=list(heading_stack),
                block_roles=_roles_for_node(node),
                linked_content_id=node.linked_content_id,
            )
        )

    return blocks


def _roles_for_node(node: RawParseNode) -> list[str]:
    """Execute `_roles_for_node`.

    Args:
        node: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if node.kind == "caption":
        return ["table-context"]
    if node.kind == "list_item":
        return ["list-item"]
    if node.kind == "paragraph":
        return ["clause"]
    if node.kind == "table":
        return ["table"]
    return [node.kind.replace("_", "-")]


def _update_heading_stack(stack: list[str], heading_text: str, heading_level: int | None) -> list[str]:
    """Execute `_update_heading_stack`.

    Args:
        stack: Input parameter.
        heading_text: Input parameter.
        heading_level: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    level = heading_level if heading_level is not None and heading_level > 0 else len(stack) + 1
    trimmed = list(stack[: max(level - 1, 0)])
    trimmed.append(heading_text)
    return trimmed


def _clean_text(value: str) -> str:
    """Execute `_clean_text`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    text = _GLYPH_PATTERN.sub("", value)

    def replace_command(match: re.Match[str]) -> str:
        """Execute `replace_command`.

        Args:
            match: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        command = match.group(1)
        return _COMMAND_REPLACEMENTS.get(command, "")

    text = _COMMAND_PATTERN.sub(replace_command, text)
    return " ".join(text.split())

