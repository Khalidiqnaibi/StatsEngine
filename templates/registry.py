from typing import Dict, Type, List, Optional, Any
from core.base_template import BaseTemplate

from templates.distributions.binomial import BinomialTrialTemplate


class TemplateRegistry:
    """
    Master Registry for StatsEngine templates.
    Manages lookup, validation, template metadata, and dynamic registration.
    """

    _instance: "TemplateRegistry" = None

    def __new__(cls) -> "TemplateRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._registry: Dict[str, Type[BaseTemplate]] = {}
        self._register_defaults()
        self._initialized = True

    def _register_defaults(self) -> None:
        """Register default core templates on engine initialization."""
        default_templates: List[Type[BaseTemplate]] = [
            BinomialTrialTemplate
        ]
        for template_cls in default_templates:
            self.register(template_cls)

    def register(self, template_cls: Type[BaseTemplate]) -> Type[BaseTemplate]:
        """
        Registers a template class into the registry.
        Can be called directly or used as a class decorator.
        """
        if not issubclass(template_cls, BaseTemplate):
            raise TypeError(
                f"Cannot register {template_cls.__name__}: must inherit from BaseTemplate"
            )

        tid = getattr(template_cls, "template_id", None)
        if not tid or not isinstance(tid, str):
            raise ValueError(
                f"Class {template_cls.__name__} must define a non-empty string 'template_id'."
            )

        self._registry[tid] = template_cls
        return template_cls

    def get(self, template_id: str) -> Optional[Type[BaseTemplate]]:
        """Retrieves a template class by ID, returning None if not found."""
        return self._registry.get(template_id)

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        Returns metadata for all registered templates.
        Useful for feeding template definitions directly into the LLM/Parser prompt layer.
        """
        catalog = []
        for tid, cls in self._registry.items():
            catalog.append(
                {
                    "template_id": tid,
                    "description": getattr(cls, "description", ""),
                    "required_params": getattr(cls, "required_params", []),
                }
            )
        return catalog

    def __getitem__(self, template_id: str) -> Type[BaseTemplate]:
        if template_id not in self._registry:
            raise KeyError(
                f"Template '{template_id}' is not registered in StatsEngine."
            )
        return self._registry[template_id]

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._registry


# Primary singleton instance
template_registry = TemplateRegistry()

# Backwards-compatible dict interface for router.py
REGISTRY: Dict[str, Type[BaseTemplate]] = template_registry._registry


def register(template_cls: Type[BaseTemplate]) -> Type[BaseTemplate]:
    """
    Decorator helper for registering new templates in under 5 seconds.
    
    Example:
        @register
        class HypergeometricTemplate(BaseTemplate):
            template_id = "hypergeometric_event"
            ...
    """
    return template_registry.register(template_cls)