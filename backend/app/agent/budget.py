from dataclasses import dataclass


@dataclass
class SearchBudget:
    """
    Controls how many SerpApi searches a mission can consume.
    """

    maximum: int = 8
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(self.maximum - self.used, 0)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.maximum

    def consume(self) -> None:
        if self.exhausted:
            raise RuntimeError(
                f"Search budget exhausted. "
                f"Maximum allowed searches: {self.maximum}"
            )

        self.used += 1

    def status(self) -> dict:
        return {
            "maximum": self.maximum,
            "used": self.used,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
        }
        