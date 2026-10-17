"""User-facing errors. Each carries a `hint` telling the user how to fix it."""


class MeetingsError(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class ConfigError(MeetingsError):
    pass


class AudioError(MeetingsError):
    pass


class TranscriptionError(MeetingsError):
    pass


class DiarizationError(MeetingsError):
    pass


class LLMError(MeetingsError):
    pass
