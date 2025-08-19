from pydantic import BaseModel
from typing import Callable, Dict, Set, TYPE_CHECKING
from mythologizer_core.types import id_type

if TYPE_CHECKING:
    from mythologizer_core.myth.myth import Myth

class UpdateMythQueue(BaseModel):

    update_myth_function_bulk: Callable
    _queue: Dict[id_type, 'Myth'] = {}

    def enqueue(self, myth: 'Myth') -> None:
        if myth.id is None:
            raise ValueError("Cannot enqueue a myth without an ID")
        
        self._queue[myth.id] = myth

    def flush(self) -> None:
        if not self._queue:
            return
        myhts = list(self._queue.values())
        self.update_myth_function_bulk(myhts)
        self._queue.clear()

        