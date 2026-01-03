from sai_rl import SAIClient
import sys

# List of likely names based on SAI naming conventions
candidates = [
    "lower-t1-penalty-with-obstacles",
    "lower-t1-obstacle-penalty-kick",
    "lower-t1-penalty-obstacle",
    "lower-t1-penalty-kick-w-obstacles",
    "lower-t1-obstacle",
    "lower-t1-penalty-with-obstacle"
]

print("Scanning for Obstacle Task ID...")

for task_id in candidates:
    print(f"Testing: {task_id} ...", end=" ")
    try:
        # Try to initialize the client
        c = SAIClient(comp_id=task_id)
        # Try to make the env (this confirms it exists)
        env = c.make_env()
        print("✅ SUCCESS! This is the one.")
        print(f"\n>>> USE THIS ID: {task_id} <<<\n")
        env.close()
        sys.exit(0)
    except Exception as e:
        print("❌ No")

print("\nCould not find ID. Please check the website URL for the exact slug.")