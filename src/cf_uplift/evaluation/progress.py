"""Interactive-friendly experiment progress with a dependency-free fallback."""
import sys


class ExperimentProgress:
    def __init__(
        self, total: int, enabled: bool = True, description: str = "Training models"
    ):
        self.total, self.enabled, self.description = (
            int(total),
            bool(enabled),
            description,
        )
        self.completed = 0
        self._bar = None
        if self.enabled:
            try:
                from tqdm.auto import tqdm

                self._bar = tqdm(
                    total=self.total,
                    desc=self.description,
                    unit="model",
                    dynamic_ncols=True,
                )
            except ImportError:
                self._render("starting")

    def advance(self, model: str, phase: str = ""):
        if not self.enabled:
            return
        self.completed += 1
        label = f"{phase}: {model}" if phase else model
        if self._bar is not None:
            self._bar.set_postfix_str(label, refresh=False)
            self._bar.update(1)
        else:
            self._render(label)

    def _render(self, label: str):
        width = 28
        ratio = self.completed / self.total if self.total else 1.0
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{self.description} [{bar}] {self.completed}/{self.total} {label}",
            end="\n" if self.completed >= self.total else "",
            file=sys.stderr,
            flush=True,
        )

    def close(self):
        if self._bar is not None:
            self._bar.close()
