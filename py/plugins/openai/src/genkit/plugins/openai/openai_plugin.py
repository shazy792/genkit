# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0


"""
OpenAI Plugin for Genkit.
"""

from genkit.core.action import ActionKind
from genkit.core.plugin_abc import Plugin
from genkit.core.registry import Registry
from genkit.plugins.openai.models import (
    SUPPORTED_OPENAI_MODELS,
    OpenAIModelHandler,
)


class OpenAI(Plugin):
    """
    A plugin for integrating OpenAI models with the Genkit framework.

    This class registers OpenAI model handlers within a registry, allowing
    interaction with supported OpenAI models.
    """

    def __init__(self, **openai_params):
        """
        Initializes the OpenAI plugin with the specified parameters.

        :param openai_params: Additional parameters that will be passed to the OpenAI client constructor.
                              These parameters may include API keys, timeouts, organization IDs, and
                              other configuration settings required by OpenAI's API.
        """
        self._openai_params = openai_params

    def initialize(self, registry: Registry) -> None:
        """
        Registers supported OpenAI models in the given registry.

        :param registry: The registry where OpenAI models will be registered.
        """
        for model_name, model_info in SUPPORTED_OPENAI_MODELS.items():
            handler = OpenAIModelHandler.get_model_handler(
                model=model_name, **self._openai_params
            )

            registry.register_action(
                kind=ActionKind.MODEL,
                name=f'openai/{model_name}',
                fn=handler.generate,
                metadata={
                    'model': {
                        'label': model_info.label,
                        'supports': {
                            'multiturn': model_info.supports.multiturn
                        },
                    },
                },
            )


def openai_model(name: str) -> str:
    """
    Returns a string representing the OpenAI model name to use with Genkit.
    """
    return f'openai/{name}'


__all__ = ['OpenAI', 'openai_model']
