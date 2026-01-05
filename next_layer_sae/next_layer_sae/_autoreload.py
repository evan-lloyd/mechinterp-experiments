from functools import partial

import jurigged

_has_run = False

if not _has_run:

    class _SilentLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def critical(self, *args, **kwargs):
            pass

    _jurigged_watcher = jurigged.watch(
        "next_layer_sae/*.py", autostart=False, logger=_SilentLogger()
    )
    _jurigged_watcher.start()
    _has_run = True
