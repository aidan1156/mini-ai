import asyncio
from pathlib import Path
import time

from browser_use import Browser, Agent
from browser_use_sdk import AsyncBrowserUse
from browser_use import ChatGoogle

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()  


class BrowserAgent:
    def create_agent(self, prompt: str, start_url: str) -> str:
        print(f"BrowserAgent received prompt:\n{prompt}\n")
        result = asyncio.run(self._run_agent(prompt, start_url))
        return result
    
    async def _run_agent(self, prompt: str, start_url: str) -> str:
        return """
        I only found this single listing:
        Title: Boosted mini S Electric Skateboard
        Price: £400
        Link: https://www.facebook.com/marketplace/item/835753052940274/
        """
        # client = AsyncBrowserUse()

        # session = await client.sessions.create(start_url=start_url)

        # print(f"Created browser view it at: {session.live_url}")

        # result = await client.run(prompt, session_id=str(session.id))

        # return result.output

        # Initialize the browser
        browser = Browser(
            headless=False,
            wait_between_actions=3.0,                  # ⏳ The core fix: forces a 3-second pause between every single click, scroll, or type.
            wait_for_network_idle_page_load_time=2.0,  # ⏳ Waits for background API requests (like image loading) to settle before parsing the page.
            minimum_wait_page_load_time=2.0            # ⏳ Forces a mandatory 2-second wait on every new page load.
        )

        await browser.navigate(start_url)  

        await asyncio.sleep(3)
            
        # Give the agent the context containing our page
        agent = Agent(
            task=prompt,
            browser=browser,
            llm=ChatGoogle('gemini-3.5-flash'), # Switched from 3.5 to 2.0 to test the 502 issue
        )

        # 1. Capture the history object when the agent finishes
        history = await agent.run()
        
        # 2. Extract the final answer text from the history
        final_result = history.final_result()
        
        print("\n--- AGENT COMPLETED TASK ---")
        print(f"Captured Result: {final_result}")
        
        # 3. Don't forget to close the browser context when done! 
        await browser.close()
        
        return final_result




# a = BrowserAgent()
# result = a.create_agent("""
# Please visit the marketplace search page for 'boosted board' and extract all 'boosted board' listings. Since the search results might include non-boosted board listings (like accessories, other electric skateboards, or unrelated items), please filter those out and only include actual Boosted Boards (e.g., Boosted Board V2, V3, Stealth, Mini, Mini S, Mini X, Plus, etc.). For each Boosted Board listing, please extract:
# 1. Title
# 2. Price
# 3. Link (the absolute URL of the listing)

# Format each listing strictly as:
# Title: <title>
# Price: <price>
# Link: <link to board>

# And return the list of these boards.
# """, "https://www.facebook.com/marketplace/105752519458380/search/?query=boosted%20board&exact=false&radius=500&locale=en_GB")

