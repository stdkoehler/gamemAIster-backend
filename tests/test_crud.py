import pytest

from src.brain.data_types import Entity, EntityResponse, UpdatedEntity, Scene
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
    mission_id = 27
    crud_instance.insert_mission(
        mission=Mission(
            mission_id=mission_id,
            name="Test Mission",
            description="This is a test mission.",
            game_type="shadowrun",
            background="Test background",
        )
    )

    entity_response = EntityResponse(
        entities=[
            Entity(name="Entity1", type="Type1", summary="Summary1"),
            Entity(name="Entity2", type="Type2", summary="Summary2"),
        ],
        updated_entities=[
            UpdatedEntity(
                name="Entity3",
                updated_name="Entitiy3u",
                type="Type1",
                summary="Summary3",
            ),
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
            UpdatedEntity(
                name="Entity2",
                updated_name="Entity2u",
                type="Type1",
                summary="UpdatedSummary2",
            ),
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
    entity2 = next((e for e in entities if e.name == "Entity2u"), None)
    assert entity2.summary == "UpdatedSummary2"
    entity3 = next((e for e in entities if e.name == "Entity3"), None)
    assert entity3.summary == "Summary3"
    with pytest.raises(StopIteration):
        next(e for e in entities if e.name == "Entity2")

    crud_instance._cleanse_unpersisted()


def test_update_entities_with_deletion(crud_instance):
    crud_instance._cleanse_unpersisted()
    mission_id = 27
    crud_instance.insert_mission(
        mission=Mission(
            mission_id=mission_id,
            name="Deletion Test Mission",
            description="This mission tests entity deletion.",
            game_type="shadowrun",
            background="Test background",
        )
    )

    entity_response = EntityResponse(
        entities=[
            Entity(name="EntityA", type="TypeA", summary="SummaryA"),
            Entity(name="EntityB", type="TypeB", summary="SummaryB"),
        ],
        updated_entities=[],
    )
    # Insert initial entities
    crud_instance.update_entities(mission_id, entity_response)
    entities = crud_instance.get_entities(mission_id)
    assert len(entities) == 2

    # Update with deletion of EntityA
    entity_response = EntityResponse(
        entities=[],
        updated_entities=[
            UpdatedEntity(
                name="EntityA",
                updated_name="DELETE",
                type="TypeA",
                summary="",
            ),
        ],
    )
    crud_instance.update_entities(mission_id, entity_response)

    entities = crud_instance.get_entities(mission_id)
    assert len(entities) == 1
    assert entities[0].name == "EntityB"

    # Ensure EntityA is deleted
    with pytest.raises(StopIteration):
        next(e for e in entities if e.name == "EntityA")

    crud_instance._cleanse_unpersisted()


def test_update_scenes(crud_instance):
    crud_instance._cleanse_unpersisted()
    mission_id = 27
    crud_instance.insert_mission(
        mission=Mission(
            mission_id=mission_id,
            name="Scene Test Mission",
            description="This mission tests scene CRUD.",
            game_type="shadowrun",
            background="Test background",
        )
    )

    # Insert initial scenes
    scenes = [
        Scene(
            id=1,
            title="Scene One",
            location="test",
            characters=["A", "B"],
            summary="Summary One",
            completed=False,
        ),
        Scene(
            id=2,
            title="Scene Two",
            location="test",
            characters=["A", "B"],
            summary="Summary Two",
            completed=False,
        ),
    ]
    crud_instance.update_scenes(mission_id, scenes)
    db_scenes = crud_instance.get_scenes(mission_id)
    assert len(db_scenes) == 2
    assert any(s.title == "Scene One" for s in db_scenes)
    assert any(s.title == "Scene Two" for s in db_scenes)

    # Update Scene One and remove Scene Two
    updated_scenes = [
        Scene(
            id=1,
            title="Scene One Updated",
            location="test2",
            characters=["A", "B", "C"],
            summary="Updated Summary",
            completed=True,
        ),
    ]
    crud_instance.update_scenes(mission_id, updated_scenes)
    db_scenes = crud_instance.get_scenes(mission_id)
    assert len(db_scenes) == 2
    assert db_scenes[0].title == "Scene One Updated"
    assert db_scenes[0].completed is True
    assert db_scenes[0].location == "test2"
    assert db_scenes[0].characters == ["A", "B", "C"]

    crud_instance._cleanse_unpersisted()
