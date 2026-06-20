from datetime import datetime
import os

from google import genai
from google.genai import types
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from mini.database import Skill
from mini.mini_agents.browser_agent import BrowserAgent
from mini.mini_agents.general_agent import GeneralAgent

from mini.api_adapters.llm_adapter import LLMConversationAdapter


class OrganisingAgent:
    def __init__(self, engine: Engine, general_agent: GeneralAgent, browser_agent: BrowserAgent) -> None:
        self._engine = engine
        self._general_agent = general_agent
        self._browser_agent = browser_agent
        self._tools = Tools(engine, general_agent, browser_agent)

    def _get_system_prompt(self) -> str:
        with Session(self._engine) as session:
            skills = session.query(Skill).all()
        
        skills = [f"ID: {skill.id}\nName: {skill.name}\nDescription: {skill.brief_description}" for skill in skills]
        skills_formatted = f"""
            You have access to some skills, you can get the full description of how to do each skill by calling the tool 
            get_skill with the skill's ID. Here are the skills:\n{'\n'.join(skills)}
        """
        if not skills:
            skills_formatted = ""

        prompt = f"""You are an assistant called Mini. 
        The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. 
        {skills_formatted}
        """

        return prompt

    def create_agent(self, prompt: str, conversation_id: int | None = None) -> tuple[str, int]:
        llm_adapter = LLMConversationAdapter.create(self._engine, conversation_id=conversation_id)
        print("organising agent received prompt:", prompt)
        response = llm_adapter.send_message(
            message=prompt,
            system_prompt=self._get_system_prompt(),
            tools=self._tools.tools
        )
        return response, llm_adapter.conversation_id
        # client = genai.Client(api_key=os.environ["GEMINI_API"])
        # response = client.models.generate_content(
        #     model='gemini-3.5-flash',
        #     contents=prompt,
        #     config=types.GenerateContentConfig(
        #         system_instruction=self._get_system_prompt(),
        #         tools=self._tools.tools,
        #     )
        # )

        # return response.text


class Tools:
    def __init__(self, engine: Engine, general_agent: GeneralAgent, browser_agent: BrowserAgent) -> None:
        self._engine = engine
        self._general_agent = general_agent
        self._browser_agent = browser_agent

        self.tools = [
            self.get_skill,
            self.create_general_subagent,
            self.create_browser_subagent,
        ]


    def get_skill(self, skill_id: int) -> str:
        """
        Get the full description and execution path of a skill by its ID. 
        """
        with Session(self._engine) as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if skill is None:
                return f"Skill with ID {skill_id} not found. Please quote the skill ID exactly as it appears in the system prompt."
            return f"ID: {skill.id}\nName: {skill.name}\nBrief Description: {skill.brief_description}\nFull Description: {skill.full_description}"
    
    def create_general_subagent(self, prompt: str) -> str:
        """
        Create a general agent which can perform long running tasks with the given prompt and returns its response. It can:
            - Access a workspace including a website that the user will be able to see changes to.
            - Access the internet
        """

        response = self._general_agent.create_agent(prompt)
        return response

    def create_browser_subagent(self, prompt: str, start_url: str) -> str:
        """
        Create a browser agent to start navigating from start_url, which can view and navigate websites with the given prompt and returns its response.
        """

        response = self._browser_agent.create_agent(prompt, start_url)
        return response
        