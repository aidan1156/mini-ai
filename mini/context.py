from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import Engine

from mini.config import Config

if TYPE_CHECKING:
    from mini.mini_agents.browser_agent import BrowserAgent
    from mini.mini_agents.general_agent import GeneralAgent
    from mini.mini_agents.organising_agent import OrganisingAgent


@dataclass(slots=True)
class AppContext:
    config: Config
    engine: Engine
    organising_agent: OrganisingAgent 
    general_agent: GeneralAgent
    browser_agent: BrowserAgent
