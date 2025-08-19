from pydantic import BaseModel
from typing import Callable, Optional, List
import numpy as np
from uuid import UUID
from mythologizer_core.types import id_type, Embedding
from mythologizer_core.myth import Myth

# Embedding type: list of floats or numpy array 

# Mytheme 
class Mytheme(BaseModel):
    id: id_type
    name: str
    embedding: Embedding

# Type aliases for better readability
MythemeGetter = Callable[[id_type], Optional[Mytheme]]
MythemeBulkGetter = Callable[[List[id_type]], List[Mytheme]]
MythemeSaver = Callable[[Mytheme], id_type]
MythemeBulkSaver = Callable[[List[Mytheme]], List[id_type]]

MythGetter = Callable[[id_type], Optional[Myth]]
MythBulkGetter = Callable[[List[id_type]], List[Myth]]
MythSaver = Callable[[Myth], id_type]
MythBulkSaver = Callable[[List[Myth]], List[id_type]]
MythUpdater = Callable[[Myth], None]
MythBulkUpdater = Callable[[List[Myth]], None]
MythDeleter = Callable[[id_type], bool]
MythBulkDeleter = Callable[[List[id_type]], List[bool]]

class MythemeStoreConnector(BaseModel):
    # function to get mytheme by id single
    get_mytheme: Optional[MythemeGetter] = None
    # function to get mythemes by id bulk
    get_mythemes: Optional[MythemeBulkGetter] = None
    # function to save mytheme single
    save_mytheme: Optional[MythemeSaver] = None
    # function to save mythemes bulk
    save_mythemes: Optional[MythemeBulkSaver] = None

class MythStoreConnector(BaseModel):
    # function to get myth by id single
    get_myth: Optional[MythGetter] = None
    # function to get myths by id bulk
    get_myths: Optional[MythBulkGetter] = None
    # function to save myth single
    save_myth: Optional[MythSaver] = None
    # function to save myths bulk
    save_myths: Optional[MythBulkSaver] = None
    # function to update myth single
    update_myth: Optional[MythUpdater] = None
    # function to update myths bulk
    update_myths: Optional[MythBulkUpdater] = None
    # function to delete myth single
    delete_myth: Optional[MythDeleter] = None
    # function to delete myths bulk
    delete_myths: Optional[MythBulkDeleter] = None

# DB connectors 
class MythologizerStoreConnector(BaseModel):
    mytheme_store_connector: Optional[MythemeStoreConnector] = None
    myth_store_connector: Optional[MythStoreConnector] = None

class MythologizerCore(BaseModel):
    mythologizer_store_connector: Optional[MythologizerStoreConnector] = None



