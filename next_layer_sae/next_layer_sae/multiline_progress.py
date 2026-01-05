from typing import Mapping

from tqdm.auto import tqdm

# Based on https://github.com/tqdm/tqdm/issues/630#issuecomment-1304902022


class MultilineProgress(tqdm):
    def __init__(self, *args, desc=None, num_header_lines=1, **kwargs):
        if desc is None:
            desc = [""] * num_header_lines
        self.header_lines = [
            tqdm(
                bar_format="{desc}{postfix}",
                desc=desc[i],
                leave=kwargs.get("leave", True),
            )
            for i in range(num_header_lines)
        ]
        super().__init__(*args, **kwargs)

        # Tries to close the unused progress bar in the Header Bar
        for header_line in self.header_lines:
            if hasattr(header_line, "container"):
                header_line.container.children[1].close()

    def set_description(self, *args, **kwargs):
        for header_line in self.header_lines:
            header_line.set_description(*args, **kwargs)

    def update(self, n: float | None = 1) -> bool | None:
        for header_line in self.header_lines:
            header_line.update(n)
        return super().update(n)

    def set_postfix(
        self,
        ordered_dict: Mapping[str, object] | None = None,
        refresh: bool | None = True,
        **kwargs,
    ):
        if isinstance(ordered_dict, list):
            for postfix, header_line in zip(ordered_dict, self.header_lines):
                header_line.set_postfix(postfix, refresh=False, **kwargs)
        else:
            for header_line in self.header_lines:
                header_line.set_postfix(ordered_dict, refresh=False, **kwargs)

    def close(self, *args, **kwargs):
        for header_line in self.header_lines:
            header_line.close(*args, **kwargs)
        super().close(*args, **kwargs)
