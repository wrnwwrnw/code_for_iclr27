from dataclasses import asdict, dataclass, field, fields
import math


@dataclass
class RecallConfig:
    enabled: bool = True
    search_interval: int = 32
    topk_per_head: int = 4
    max_admit_per_head: int = 4
    archive_capacity_per_head: int = 32768
    archive_buffers: int = 16
    cpu_threads: int = 2
    timeout_seconds: float = 60.0
    wait_at_audit: bool = False

    def validate(self):
        for name in ("search_interval", "topk_per_head", "max_admit_per_head",
                     "archive_capacity_per_head", "archive_buffers", "cpu_threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"recall.{name} must be a positive integer")
        if self.max_admit_per_head > self.topk_per_head:
            raise ValueError("recall.max_admit_per_head must not exceed topk_per_head")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("recall.timeout_seconds must be finite and positive")
        if type(self.enabled) is not bool or type(self.wait_at_audit) is not bool:
            raise ValueError("Recall switches must be boolean")


@dataclass
class GenerationConfig:
    model: str = ""
    device: str = "cuda:0"
    batch_size: int = 1
    memory_safety_fraction: float = 0.85
    memory_reserve_mib: int = 1024
    budget: int = 2000
    max_new_tokens: int = 8000
    audit_period: int = 250
    temporal_window: int = 250
    evict_interval: int = 8
    protect_recent: int = 32
    pool_kernel: int = 7
    query_chunk_size: int = 250
    denominator_epsilon: float = 1e-6
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int = 6211027
    stop_on_eos: bool = True
    chat_template: bool = True
    log_audits: bool = True
    recall: RecallConfig = field(default_factory=RecallConfig)

    def validate(self):
        for name in ("batch_size", "memory_reserve_mib", "budget", "max_new_tokens", "audit_period", "temporal_window",
                     "evict_interval", "protect_recent", "pool_kernel", "query_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.budget <= self.protect_recent + self.evict_interval:
            raise ValueError("budget must exceed protect_recent + evict_interval")
        if self.pool_kernel % 2 == 0:
            raise ValueError("pool_kernel must be odd")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("stop_on_eos", "chat_template", "log_audits"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("temperature", "top_p", "repetition_penalty", "denominator_epsilon"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.repetition_penalty <= 0:
            raise ValueError("Invalid generation sampling parameters")
        if not 0 < self.denominator_epsilon < 1:
            raise ValueError("denominator_epsilon must lie in (0,1)")
        if not math.isfinite(self.memory_safety_fraction) or not 0 < self.memory_safety_fraction < 1:
            raise ValueError("memory_safety_fraction must lie in (0,1)")
        if not self.device.startswith("cuda"):
            raise ValueError("Generation requires a CUDA device")
        self.recall.validate()
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        recall = values.get("recall", {})
        if not isinstance(recall, dict):
            raise ValueError("recall must be a JSON object")
        unknown = set(recall) - {item.name for item in fields(RecallConfig)}
        if unknown:
            raise ValueError(f"Unknown recall keys: {sorted(unknown)}")
        values["recall"] = RecallConfig(**recall)
        return cls(**values).validate()
