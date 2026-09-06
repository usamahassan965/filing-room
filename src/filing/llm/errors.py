from __future__ import annotations


class LLMError(RuntimeError):
    pass


class ModelUnavailable(LLMError):
    """A registered model ID no longer resolves.

    This is expected to happen: NVIDIA is renaming its retriever models. The fix
    is one line in filing.config.MODEL_REGISTRY, which is the entire reason the
    registry exists -- so this error names the candidates for you.
    """

    def __init__(
        self, role: str, model_id: str, alternates: tuple[str, ...], detail: str = ""
    ) -> None:
        options = "\n  ".join(alternates) if alternates else "(none registered)"
        super().__init__(
            f"model {model_id!r} for role {role!r} did not resolve. {detail}\n"
            f"Run 'filing probe' to see which IDs are live, then edit MODEL_REGISTRY "
            f"in src/filing/config.py. Registered alternates:\n  {options}"
        )
        self.role = role
        self.model_id = model_id
        self.alternates = alternates


class MissingCredentials(LLMError):
    pass
