import jurigged

_has_run = False

if not _has_run:
    _jurigged_watcher = jurigged.watch(
        "transformers_sae/*.py",
        autostart=False,
        logger=jurigged.live.conservative_logger,
    )
    _jurigged_watcher.registry.auto_register(
        jurigged.live.to_filter("transformers_sae/training_step/*.py")
    )
    _jurigged_watcher.start()
    _has_run = True
