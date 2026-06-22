```shell
C:\Users\Stefan\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\activate
pip install pip-tools
```

```shell
pip-compile --extra dev --no-emit-index-url --resolver=backtracking pyproject.toml --verbose
pip-sync requirements.txt
```

## NPC sheet schema contract

`src/brain/npc_models.py`'s `merge_npc()` output (the NPC generation
pipeline's result) must match the frontend's hand-crafted `*Character` TS
interfaces (`gamemAIster-frontend/src/models/CharacterProps.tsx`) and the
`NpcCard.tsx` components that render them — those are the design source of
truth, not this repo.

`tests/schemas/<game_type>.schema.json` are JSON Schemas generated from
those TS interfaces (run `npm run gen:npc-schemas` in the frontend repo to
regenerate them; see that repo's README). `tests/test_npc_schema_contract.py`
validates `merge_npc()`'s output against them for every supported game type.

If `npc_models.py`'s Stats/Equipment models or merge functions change, run
the test suite (`pytest tests/test_npc_schema_contract.py`) to confirm the
output still satisfies the frontend's shape. If the frontend interfaces
themselves changed, regenerate the schemas in the frontend repo first and
commit the updated `tests/schemas/*.json` files here.