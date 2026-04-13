from pathlib import Path
from jinja2 import meta, Environment, FileSystemLoader, StrictUndefined, TemplateSyntaxError, UndefinedError
import logging
from typing import Dict, Any, Set

logger = logging.getLogger(__name__)


class PromptLoader:
    @classmethod
    def load(cls, filepath: str | Path, **default_kwargs):
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Prompt file not found: {filepath}")
            raise FileNotFoundError(f"Prompt file not found: {filepath}")
        
        # StrictUndefined makes Jinja2 raise errors on missing variables
        env = Environment(
            loader=FileSystemLoader(filepath.parent),
            undefined=StrictUndefined
        )
        
        try:
            template = env.get_template(filepath.name)
        except TemplateSyntaxError as e:
            logger.error(f"Syntax error in {filepath}: Line {e.lineno}: {e.message}")
            raise
        
        # extract required variables from template source
        try:
            # raw template source
            source, _, _ = env.loader.get_source(env, filepath.name)
            ast = env.parse(source)
            required_vars = cls._extract_variables(ast)
        except Exception as e:
            logger.warning(f"Could not extract variables from {filepath}: {e}")
            required_vars = set()
        
        return Prompt(
            name=filepath.stem,
            template=template,
            default_kwargs=default_kwargs,
            required_vars=required_vars,
            filepath=filepath
        )
    
    @classmethod
    def _extract_variables(cls, ast) -> Set[str]:
        return meta.find_undeclared_variables(ast)


class Prompt:
    def __init__(
        self, 
        name: str, 
        template, 
        default_kwargs: Dict[str, Any],
        required_vars: Set[str],
        filepath: Path
    ):
        self.name = name
        self.template = template
        self.default_kwargs = default_kwargs
        self.required_vars = required_vars
        self.filepath = filepath
    
    def validate_kwargs(self, **kwargs) -> None:
        all_kwargs = {**self.default_kwargs, **kwargs}
        provided = set(all_kwargs.keys())
        missing = self.required_vars - provided
        
        if missing:
            error_msg = (
                f"Missing required variables for prompt '{self.name}':\n"
                f"  Missing: {sorted(missing)}\n"
                f"  Required: {sorted(self.required_vars)}\n"
                f"  Provided: {sorted(provided)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        unused = provided - self.required_vars
        if unused:
            logger.debug(
                f"Unused variables in prompt '{self.name}': {sorted(unused)}"
            )
    
    def format(self, validate: bool = True, **kwargs) -> Dict[str, str]:
        if validate:
            self.validate_kwargs(**kwargs)
        
        all_kwargs = {**self.default_kwargs, **kwargs}
        
        try:
            rendered = self.template.render(**all_kwargs)
            return self._parse_output(rendered)
        except Exception as e:
            logger.error(f"Error rendering prompt '{self.name}': {e}")
            raise
    
    def get_messages(self, validate: bool = True, **kwargs) -> list[dict]:
        formatted = self.format(validate=validate, **kwargs)
        return [
            {"role": "system", "content": formatted["system"]},
            {"role": "user", "content": formatted["user"]}
        ]
    
    def _parse_output(self, rendered: str) -> Dict[str, str]:
        if '---SYSTEM---' not in rendered:
            logger.warning(f"No ---SYSTEM--- section found in prompt '{self.name}'")
            return {"system": "", "user": rendered.strip()}
        
        parts = rendered.split('---SYSTEM---', 1)
        if len(parts) < 2:
            return {"system": "", "user": rendered.strip()}
        
        rest = parts[1].split('---USER---', 1)
        system = rest[0].strip()
        user = rest[1].strip() if len(rest) > 1 else ""
        
        return {"system": system, "user": user}
    
    def get_required_vars(self) -> Set[str]:
        return self.required_vars - set(self.default_kwargs.keys())
    
    def __repr__(self) -> str:
        required = self.get_required_vars()
        return f"Prompt(name='{self.name}', required_vars={sorted(required)})"
