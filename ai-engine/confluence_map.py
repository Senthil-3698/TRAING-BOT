import redis
import os

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)

def check_confluence(symbol, current_tf, current_action):
    """
    The Scalper's Law: Never trade against the higher gear.
    """
    # Get the "State" of the higher timeframe from Redis
    macro_bias = r.get(f"{symbol}_1h_bias")
    
    if macro_bias and macro_bias.decode('utf-8') != current_action:
        print(f"Conflict Detected: {current_tf} {current_action} vs 1H Bias. Aborting.")
        return False
        
    print(f"Confluence Confirmed for {symbol}.")
    return True