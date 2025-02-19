# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from genkit.core.typing import Message, Role, TextPart
from genkit.plugins.google_ai import (
    GoogleAi,
    GoogleAiPluginOptions,
    googleai_name,
)
from genkit.plugins.google_ai.models import models
from genkit.veneer import Genkit

ai = Genkit(
    plugins=[GoogleAi(plugin_params=GoogleAiPluginOptions())],
    model=googleai_name(models.GoogleAiVersion.GEMINI_2_0_PRO_EXP_02_05),
)


@ai.flow()
async def say_hi(data: str):
    return await ai.generate(
        messages=[
            Message(role=Role.USER, content=[TextPart(text=f'hi {data}')])
        ]
    )


def main() -> None:
    print(asyncio.run(say_hi(', tell me a joke')).message.content)


if __name__ == '__main__':
    main()
