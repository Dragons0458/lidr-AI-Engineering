PRICING = {
    # OpenAI (USD per 1M tokens)
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    # Anthropic (USD per 1M tokens)
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6-20250514": {"input": 3.00, "output": 15.00},
    # Google (USD per 1M tokens)
    "gemini/gemini-2.5-flash": {"input": 0.1, "output": 0.4},
}


def calculate_cost(model, input_tokens, output_tokens):
    """Calculate the cost of an API call in USD."""
    prices = PRICING.get(model)
    if not prices:
        for key in PRICING:
            if key in model or model in key:
                prices = PRICING[key]
                break
    if not prices:
        return None

    cost_in = (input_tokens / 1_000_000) * prices["input"]
    cost_out = (output_tokens / 1_000_000) * prices["output"]
    return {"input": cost_in, "output": cost_out, "total": cost_in + cost_out}
