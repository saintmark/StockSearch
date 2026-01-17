import requests
import json
import time
from typing import Dict, Any
from config import LLM_API_KEY, LLM_API_URL, LLM_MODEL, SYSTEM_PROMPT

class LLMClient:
    def __init__(self):
        self.api_key = LLM_API_KEY
        self.api_url = LLM_API_URL
        self.model = LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def analyze_news(self, text: str) -> Dict[str, Any]:
        """
        Calls the LLM API to analyze the sentiment of the given news text.
        Returns a dictionary with score, sentiment, sector, and reasoning.
        """
        # Quick fallback if no API key is configured
        if not self.api_key or "YOUR_API_KEY" in self.api_key:
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"新闻内容：{text}"}
            ],
            "temperature": 0.3, # Low temperature for consistent output
            "max_tokens": 256,
            "stream": False
        }

        retries = 2
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=30 # Increased timeout to 30s
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Robust JSON extraction using regex
                import re
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    data = json.loads(json_str)
                    return data
                else:
                    print(f"[LLMClient] No JSON found in response: {content[:100]}...")
                    return None
                
            except requests.exceptions.Timeout:
                print(f"[LLMClient] Timeout on attempt {attempt+1}")
                if attempt == retries:
                    print("[LLMClient] Max retries reached. Returning None.")
                    return None
            except json.JSONDecodeError:
                print(f"[LLMClient] Failed to parse JSON: {content[:100]}...")
                return None
            except Exception as e:
                print(f"[LLMClient] Error: {e}")
                return None
            
            time.sleep(1) # Backoff
            
        return None

if __name__ == "__main__":
    # Test stub
    client = LLMClient()
    if "YOUR_" not in client.api_key:
        test_text = "宁德时代第三季度净利润增长30%，超出市场预期。"
        print("Testing with:", test_text)
        print(client.analyze_news(test_text))
    else:
        print("Please configure API Key in config.py to test.")
