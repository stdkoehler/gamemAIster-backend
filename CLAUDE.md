# gamemAIster Backend

Python/FastAPI backend for a multi-TTRPG-system AI gamemaster app.

**Sibling repo:** `../gamemAIster-frontend` (React/TypeScript) has its own `CLAUDE.md` — read it too when a task touches both repos. This is common: character-sheet/NPC field changes (the most frequent kind of cross-cutting change here) always span both, because the frontend owns the schema and the backend must conform to it (see below).

## Commands

- `poetry run python -m pytest tests/` — run the test suite. Note: the `dev/` directory holds manual scripts that expect a local LLM server (textgen-webui) running on `127.0.0.1:5000` — they are not part of the real suite and will fail to collect without that server; run `pytest tests/` (not bare `pytest .`) to skip them.
- `poetry run python -m pytest tests/test_npc_schema_contract.py` — verify the NPC generation pipeline's output still matches the frontend's character schemas (run this after touching `npc_models.py`, `system_registry.py`, or `character_formatters.py`).

## Supported game systems

Defined per-system in `src/brain/system_registry.py` — the single source of truth for story prompts, oracle classes, and NPC-generation config (`GAME_CONFIGS`, `NPC_CONFIGS`). Currently: Shadowrun (5E/6E), Vampire: The Masquerade (5E), Call of Cthulhu (7E), Seventh Sea (2E), The Expanse (AGE system), Dragonlance: Shadow of the Dragon Queen (D&D 5E), a free-form Custom RPG (no NPC-generation support), and **Slavic** — a homebrew "Baltic Slavic 800 A.D." setting built on **Forbidden Lands** (Free League Publishing, Year Zero Engine) rules: 4 attributes double as health pools, damage applies directly to the governing attribute (no separate HP pool), "Broken" at 0. Mirrors the frontend's `SlavicCharacter` doc comment in `CharacterProps.tsx`.

## Character schema contract — the frontend is the source of truth, not this repo

The frontend's `*Character` TS interfaces (`gamemAIster-frontend/src/models/CharacterProps.tsx`) and the hand-crafted `NpcCard.tsx`/`*Sheet.tsx` components that render them are the canonical shape for both player character sheets and NPCs. This repo's job is to *conform* to that shape, not define it.

- `tests/schemas/<game_type>.schema.json` are JSON Schemas **generated from those TS interfaces** by `gamemAIster-frontend/scripts/generate-npc-schemas.mjs` (`npm run gen:npc-schemas`, run from the frontend repo). **Never hand-edit these JSON files** — they're build output and will silently drift from the real contract the next time someone regenerates them.
- `tests/test_npc_schema_contract.py` validates `merge_npc()`'s output (in `src/brain/system_registry.py`, dispatching to the per-system merge functions in `src/brain/npc_models.py`) against those schemas, for every supported game type.
- **Practical workflow when a character/NPC field changes:** the change starts in the frontend (`CharacterProps.tsx`) → run `npm run gen:npc-schemas` there → switch to this repo and update the relevant `*NpcStats`/`*NpcEquipment` Pydantic models and `_merge_*` function in `npc_models.py`, the NPC-equipment/stats generation prompts under `src/brain/prompt_templates/<system>/`, and the party/NPC formatters in `character_formatters.py` → run `tests/test_npc_schema_contract.py` to confirm.
- See also `README.md`'s "NPC sheet schema contract" section (same contract, shorter).

## NPC generation pipeline

3-stage pipeline (Profiler → Stats → Equipment) in `Gamemaster.generate_npc()` (`src/brain/gamemaster.py`), using per-system Pydantic models and a `_merge_*` function defined in `src/brain/npc_models.py`, dispatched via `NPC_CONFIGS` in `system_registry.py`. NPC stats/equipment are intentionally less detailed than a full player character sheet — scoped to the combat-relevant fields `NpcCard.tsx`'s NPC views actually render — and the merge step backfills every other required field with the same safe defaults the frontend's `createBlankCharacter()` uses.

## Party/NPC prompt injection

`src/brain/character_formatters.py` turns character-sheet content into the JSON injected into the LLM system prompt, with a deliberate split: `format_party` is lean (the player already knows their own skills/gear, so only identity/attributes/resource-tracks are included), `format_npc` is rich (skills, equipment, special abilities — since the GM plays them). Fields that are structurally not applicable (e.g. a mortal vampire's `clan`/`generation`) are omitted rather than sent as null/zero. The per-system story-prompt templates in `src/brain/prompt_templates/<system>/<system>_story_prompt.txt` inject this via `{PARTY}`/`{NPCS}` placeholders (alongside the pre-existing `{MISSION}`/`{BACKGROUND}`), each with a field legend tailored to that system, formatted in `SummaryChat._build_messages()` (`src/brain/chat.py`).
