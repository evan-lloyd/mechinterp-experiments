import jurigged

_has_run = False

if not _has_run:
    _jurigged_watcher = jurigged.watch(
        "tiny_stories_sae/*.py", autostart=False, logger=jurigged.live.conservative_logger
    )
    _jurigged_watcher.start()
    _has_run = True
