from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from sqlalchemy import create_engine
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from mini.database import create_db
from mini.config import load_config
from mini.context import AppContext
from mini.mini_agents.browser_agent import BrowserAgent
from mini.mini_agents.general_agent import GeneralAgent
from mini.mini_agents.organising_agent import OrganisingAgent

load_dotenv()  

repo_root = Path(__file__).resolve().parent.parent
config = load_config(str(repo_root / "config.json"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    database_path = repo_root / config.database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    create_db(engine)

    general_agent = GeneralAgent(workspace_path=config.workspace_path)
    browser_agent = BrowserAgent()

    context = AppContext(
        config=config,
        engine=engine,
        general_agent=general_agent,
        browser_agent=browser_agent,
        organising_agent=OrganisingAgent(engine=engine, general_agent=general_agent, browser_agent=browser_agent)
    )
    app.state.context = context

    try:
        yield
    finally:
        app.state.context.engine.dispose()


app = FastAPI(lifespan=lifespan)


def get_context(request: Request) -> AppContext:
    return request.app.state.context


class SendMessageRequest(BaseModel):
    message: str
    conversation_id: int | None = None


@app.post("/send-message")
def send_message(payload: SendMessageRequest, context: AppContext = Depends(get_context)) -> dict[str, Any]:
    response, conversation_id = context.organising_agent.create_agent(payload.message, conversation_id=payload.conversation_id)
    return {
        "response": response,
        "conversation_id": conversation_id
    }

if __name__ == "__main__":
    uvicorn.run("mini.main:app", host=config.server.host, port=config.server.port, reload=True)
