# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from genkit.core.typing import ModelInfo, Supports

gemini10Pro = ModelInfo(
    label='Google AI - Gemini Pro',
    versions=['gemini-pro', 'gemini-1.0-pro-latest', 'gemini-1.0-pro-001'],
    supports=Supports(
        multiturn=True,
        media=False,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained='no-tools',
    ),
)

gemini15Pro = ModelInfo(
    label='Google AI - Gemini 1.5 Pro',
    versions=[
        'gemini-1.5-pro-latest',
        'gemini-1.5-pro-001',
        'gemini-1.5-pro-002',
    ],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained='no-tools',
    ),
)

gemini15Flash = ModelInfo(
    label='Google AI - Gemini 1.5 Flash',
    versions=[
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash-001',
        'gemini-1.5-flash-002',
    ],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained='no-tools',
    ),
)


gemini15Flash8b = ModelInfo(
    label='Google AI - Gemini 1.5 Flash',
    versions=['gemini-1.5-flash-8b-latest', 'gemini-1.5-flash-8b-001'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained='no-tools',
    ),
)


gemini20Flash = ModelInfo(
    label='Google AI - Gemini 2.0 Flash',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained='no-tools',
    ),
)


gemini20ProExp0205 = ModelInfo(
    label='Google AI - Gemini 2.0 Pro Exp 02-05',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained='no-tools',
    ),
)


class GoogleAiVersion(StrEnum):
    GEMINI_1_0_PRO = 'gemini-1.0-pro'
    GEMINI_1_5_PRO = 'gemini-1.5-pro'
    GEMINI_1_5_FLASH = 'gemini-1.5-flash'
    GEMINI_1_5_FLASH_8B = 'gemini-1.5-flash-8b'
    GEMINI_2_0_FLASH = 'gemini-2.0-flash'
    GEMINI_2_0_PRO_EXP_02_05 = 'gemini-2.0-pro-exp-02-05'


SUPPORTED_MODELS = {
    GoogleAiVersion.GEMINI_1_0_PRO: gemini10Pro,
    GoogleAiVersion.GEMINI_1_5_PRO: gemini15Pro,
    GoogleAiVersion.GEMINI_1_5_FLASH: gemini15Flash,
    GoogleAiVersion.GEMINI_1_5_FLASH_8B: gemini15Flash8b,
    GoogleAiVersion.GEMINI_2_0_FLASH: gemini20Flash,
    GoogleAiVersion.GEMINI_2_0_PRO_EXP_02_05: gemini20ProExp0205,
}
