"""Public errors never include credentials, input text, or upstream bodies."""


class ProviderError(RuntimeError):
    def __init__(self, code: str, *, status: int | None = None, detail: dict | None = None):
        self.code, self.status = code, status
        # Numeric diagnostics only; never request text, credentials or upstream bodies.
        self.detail = detail or {}
        super().__init__(code if status is None else f"{code} (HTTP {status})")


class ContractError(ProviderError):
    pass


class BudgetExceeded(ProviderError):
    pass


class ConfigurationError(ValueError):
    pass
