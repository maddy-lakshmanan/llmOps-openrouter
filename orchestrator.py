from openai import OpenAI
import random
from dotenv import load_dotenv
import os
import sys
load_dotenv()
# Initialize client for DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com" , # or "https://api.deepseek.com/v1",
    timeout=5
)



def call_model(prompt, model_name="deepseek-chat"):
    """Send prompt to specified DeepSeek model and return response text."""
    try:
        print(f"[DEBUG] Sending request to {model_name}...")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        print(f"[DEBUG] Got response from {model_name}")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Failed to call {model_name}: {e}")
        sys.exit(1)

def ab_test(prompt, model_a="deepseek-chat", model_b="deepseek-reasoner", num_runs=3):
    """Run A/B testing between two models."""
    results = {"A": [], "B": []}
    for i in range(num_runs):
        print(f"\n[INFO] A/B Test iteration {i+1}")
        results["A"].append(call_model(prompt, model_a))
        results["B"].append(call_model(prompt, model_b))
    return results

def route_by_skill(prompt, task_type):
    """Route prompt to model based on skill/task type."""
    skill_map = {
        "code": "deepseek-reasoner",
        "math": "deepseek-reasoner",
        "chat": "deepseek-chat",
        "summarize": "deepseek-chat"
    }
    model = skill_map.get(task_type, "deepseek-chat")
    return call_model(prompt, model)

# ====== Predefined Non-Interactive Execution ======
if __name__ == "__main__":
    ab_prompt = "Explain the difference between supervised and unsupervised learning in simple terms."
    ab_results = ab_test(ab_prompt)
   
    print("\n=== A/B Test Results ===")
    print("Model A (deepseek-chat):")
    for r in ab_results["A"]:
        print("-", r)
    print("\nModel B (deepseek-reasoner):")
    for r in ab_results["B"]:
        print("-", r)

    print("\n=== Skill Routing Examples ===")
    print("Code Task:", route_by_skill("Write a Python function to reverse a list.", "code"))
    print("\nMath Task:", route_by_skill("Solve: 1234 * 5678", "math"))
    print("\nSummary Task:", route_by_skill("Summarize: Large language models are AI systems...", "summarize"))