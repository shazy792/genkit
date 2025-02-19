# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from typing import Optional

import pydantic
from genkit.core import plugin_abc
from genkit.core.action import ActionKind
from genkit.core.registry import Registry
from genkit.core.typing import (
    GenerateRequest,
    GenerateResponse,
    Message,
    Role,
    TextPart,
)
from genkit.plugins.google_ai.models import models
from google import genai


def googleai_name(name: str) -> str:
    """Create a Google AI action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified Google AI action name.
    """
    return f'googleai/{name}'


class GoogleAiPluginOptions(pydantic.BaseModel):
    api_key: Optional[str] = None
    # TODO: implement all authentication methods
    # project: Optional[str] = None,
    # location: Optional[str] = None
    # TODO: implement http options
    # api_version: Optional[str] = None
    # base_url: Optional[str] = None


class GoogleAi(plugin_abc.Plugin):
    def __init__(self, plugin_params=GoogleAiPluginOptions):
        api_key = (
            plugin_params.api_key
            if plugin_params.api_key
            else os.getenv('GEMINI_API_KEY')
        )
        if not api_key:
            raise ValueError(
                'Gemini api key should be passed in plugin params '
                'or as a GEMINI_API_KEY environment variable'
            )
        self._client = genai.Client(api_key=api_key)

    def initialize(self, registry: Registry):
        for name, model in models.SUPPORTED_MODELS.items():
            model_metadata = {
                'model': {
                    'supports': model.supports.model_dump(),
                }
            }

            registry.register_action(
                kind=ActionKind.MODEL,
                name=googleai_name(name),
                fn=self._create_callback(name),
                metadata=model_metadata,
            )

    def _create_callback(
        self, model: str
    ) -> Callable[[GenerateRequest], GenerateResponse]:
        def model_callback(request: GenerateRequest) -> GenerateResponse:
            reqest_msgs: list[genai.types.Content] = []
            for msg in request.messages:
                message_parts: list[genai.types.Part] = []
                for p in msg.content:
                    message_parts.append(
                        genai.types.Part.from_text(text=p.root.text)
                    )
                reqest_msgs.append(
                    genai.types.Content(parts=message_parts, role=msg.role)
                )
            response = self._client.models.generate_content(
                model=model, contents=reqest_msgs
            )

            return GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[TextPart(text=response.text)],
                )
            )

        return model_callback
