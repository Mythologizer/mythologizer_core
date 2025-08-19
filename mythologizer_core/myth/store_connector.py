# SAVING MYTH:
MythSaver = Callable[[Myth], id_type] # save myth and return id

# GETTING MYTH:
MythGetter = Callable[[id_type], Myth] # get myth by id
MythBulkGetter = Callable[[List[id_type]], List[Myth]] # get myths by ids

# UPDATING MYTH:
MythBulkUpdater = Callable[[List[Myth]], None] # update myths by ids



