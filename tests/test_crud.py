import pytest

from src.brain.data_types import Entity, EntityResponse
from src.crud.crud import CRUD
import io
import sys

from src.routers.schema.mission import Mission


@pytest.fixture(scope="module")
def crud_instance():
    return CRUD(dbase="sqlite:///unittest.db")


def test_get_entities(crud_instance):
    mission_id = 13
    entities = crud_instance.get_entities(mission_id)
    assert isinstance(entities, list)
    assert all(isinstance(entity, Entity) for entity in entities)


def test_update_entities(crud_instance):
    crud_instance._cleanse_unpersisted()
    crud_instance.insert_mission(
        mission=Mission(
            mission_id=14,
            name="Test Mission",
            description="This is a test mission.",
        )
    )

    mission_id = 14
    entity_response = EntityResponse(
        entities=[
            Entity(name="Entity1", type="Type1", summary="Summary1"),
            Entity(name="Entity2", type="Type2", summary="Summary2"),
        ],
        updated_entities=[
            Entity(name="Entity3", type="Type1", summary="Summary3"),
        ],
    )
    # only entities shall be inserted, updated_entities are ignored because they
    # are not in the db yet
    crud_instance.update_entities(mission_id, entity_response)
    entities = crud_instance.get_entities(mission_id)
    assert len(entities) == 2

    entity_response = EntityResponse(
        entities=[
            Entity(name="Entity1", type="Type1", summary="Summary1Append"),
            Entity(name="Entity3", type="Type2", summary="Summary3"),
        ],
        updated_entities=[
            Entity(name="Entity2", type="Type1", summary="UpdatedSummary2"),
        ],
    )
    # Entity1 will be appended because it already exists in the db
    # Entity2 will be updated because it is in the updated_entities list
    # Entity3 will be inserted because it is in the entities list and not in the db yet
    crud_instance.update_entities(mission_id, entity_response)

    entities = crud_instance.get_entities(mission_id)
    assert len(entities) == 3
    entity1 = next((e for e in entities if e.name == "Entity1"), None)
    assert entity1.summary == "Summary1; Summary1Append"
    entity2 = next((e for e in entities if e.name == "Entity2"), None)
    assert entity2.summary == "UpdatedSummary2"
    entity3 = next((e for e in entities if e.name == "Entity3"), None)
    assert entity3.summary == "Summary3"

    crud_instance._cleanse_unpersisted()
