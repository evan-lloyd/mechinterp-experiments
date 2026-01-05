from functools import partial

import jurigged

_has_run = False


def _monkeypatched_on_prepare(self, module_name, filename):
    jurigged.live.JuriggedHandler(self, filename).schedule(self.observer)


if not _has_run:
    _jurigged_watcher = jurigged.watch("sae/*.py", autostart=False)
    _jurigged_watcher.registry.auto_register(
        jurigged.live.to_filter("backtracking/*.py")
    )
    _jurigged_watcher.registry.precache_activity[-1] = partial(
        _monkeypatched_on_prepare, _jurigged_watcher
    )
    _jurigged_watcher.start()
    _has_run = True
