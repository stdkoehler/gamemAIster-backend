import pytest

from src.brain.data_types import Entity
from src.crud.crud import CRUD


@pytest.fixture(scope="module")
def crud_instance():
    return CRUD(dbase="sqlite:///unittest.db")


def test_get_entities(crud_instance):
    mission_id = 13
    entities = crud_instance.get_entities(mission_id)
    assert isinstance(entities, list)
    assert all(isinstance(entity, Entity) for entity in entities)
