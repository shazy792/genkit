# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0


"""
OpenAI Models for Genkit.
"""

from typing import Any

from genkit.core.action import ActionRunContext
from genkit.core.typing import (
    GenerateRequest,
    GenerateResponse,
    GenerationCommonConfig,
)
from genkit.plugins.openai.models.model import OpenAIModel
from genkit.plugins.openai.models.model_info import SUPPORTED_OPENAI_MODELS


class OpenAIModelHandler:
    """
    Handles OpenAI API interactions for the Genkit plugin.
    """

    def __init__(self, model: Any):
        """
        Initializes the OpenAIModelHandler with a specified model.

        :param model: An instance of a Model subclass representing the OpenAI model.
        """
        self._model = model

    @classmethod
    def get_model_handler(
        cls, model: str, **openai_options: Any
    ) -> 'OpenAIModelHandler':
        """
        Factory method to initialize the model handler for the specified OpenAI model.

        :param model: The OpenAI model name.
        :param openai_options: Additional parameters for OpenAI client initialization.
        :return: An instance of OpenAIModelHandler with the corresponding model.
        :raises ValueError: If the specified model is not supported.
        """
        if model not in SUPPORTED_OPENAI_MODELS:
            raise ValueError(f"Model '{model}' is not supported.")

        openai_model = OpenAIModel(model, **openai_options)
        return cls(openai_model)

    def validate_version(self, version: str):
        """
        Validates whether the specified model version is supported.

        :param version: The version of the model to be validated.
        :raises ValueError: If the specified model version is not supported.
        """
        model_info = SUPPORTED_OPENAI_MODELS[self._model.name]
        if version not in model_info.versions:
            raise ValueError(f"Model version '{version}' is not supported.")

    def generate(
        self, request: GenerateRequest, ctx: ActionRunContext
    ) -> GenerateResponse:
        """
        Processes the request using OpenAI's chat completion API.

        :param request: The request containing messages and configurations.
        :return: A GenerateResponse containing the model's response.
        :raises ValueError: If the specified model version is not supported.
        """
        if request.config and isinstance(
            request.config, GenerationCommonConfig
        ):
            self.validate_version(request.config.version)

        return self._model.generate(request)
