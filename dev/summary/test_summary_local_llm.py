"""
Test script for SummaryMemory entity and scene summary functionality
using a local LLM and unittest.db as the database.
"""

import sys
import os
import json
from pathlib import Path
from unittest.mock import patch

from src.brain.chat import SummaryMemory
from src.llmclient.llm_client import LLMClientDeepSeek, LLMClientLocal
from src.crud import crud

# crud_instance in crud is hardcoded to use memory.db, so we patch it to use unittest.db
new_instance = crud.CRUD(dbase="sqlite:///unittest.db")
with patch(
    "src.brain.chat.crud_instance",
    new=new_instance,
):

    llm_client_local = LLMClientLocal(base_url="http://127.0.0.1:5000")

    # api_key = os.getenv("API_KEY_DEEPSEEK")
    # llm_client_local = LLMClientDeepSeek(api_key=api_key, model="deepseek-reasoner")

    src_path = Path(__file__).parents[2] / "src"

    with open(
        src_path / "brain/prompt_templates/text_summary_prompt.txt",
        "r",
        encoding="utf-8",
    ) as f:
        summary_template = f.read()
    with open(
        src_path / "brain/prompt_templates/text_entity_prompt.txt",
        "r",
        encoding="utf-8",
    ) as f:
        entity_template = f.read()
    with open(
        src_path / "brain/prompt_templates/text_scene_prompt_examples.txt",
        "r",
        encoding="utf-8",
    ) as f:
        scene_template = f.read()

    # Instantiate SummaryMemory with dummy LLM and test db
    memory = SummaryMemory(
        llm_client=llm_client_local,
        summary_template=summary_template,
        entity_template=entity_template,
        scene_template=scene_template,
        game_name="expanse",
        last_k=5,
        mission_id=26,
        min_summary_tokens=2048,
    )

    interaction_candidates = memory._history[memory._n_summarized : -memory._last_k]
    text = "\n".join(
        [
            interaction.format_interaction_summary()
            for interaction in interaction_candidates
        ]
    )

    scenes = memory.scene_summary(text)
    breakme = 1
