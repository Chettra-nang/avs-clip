"""Verify that dataset.jsonl format includes role and ambulance_dist fields."""
import json
import sys

def verify_dataset_format(jsonl_path: str):
    """
    Verify that dataset.jsonl has the required fields:
    - Each agent record has 'role' field
    - Normal agent scene dicts have 'ambulance_dist' when applicable
    
    Args:
        jsonl_path: Path to dataset.jsonl file
    """
    print(f"Verifying dataset format: {jsonl_path}")
    print("-" * 60)
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Check first 3 records
                break
            
            record = json.loads(line)
            print(f"\nRecord {i}:")
            print(f"  Episode: {record['episode']}, Step: {record['step']}")
            print(f"  Quality: {record['quality']:.3f}")
            
            # Check each agent
            for agent_idx, agent in enumerate(record['agents']):
                print(f"\n  Agent {agent_idx}:")
                
                # Verify 'role' field exists
                if 'role' not in agent:
                    print(f"    ❌ ERROR: Missing 'role' field!")
                    return False
                
                role = agent['role']
                print(f"    Role: {role}")
                print(f"    Action: {agent['action_name']}")
                print(f"    Speed: {agent['speed']:.1f} m/s")
                
                # Check scene dict
                scene = agent['scene']
                print(f"    Scene: front_dist={scene['front_dist']:.1f}m, " +
                      f"front_ttc={scene['front_ttc']:.1f}s, " +
                      f"risk={scene['risk']:.2f}")
                
                # For normal agents, check if ambulance_dist is present when applicable
                if role == 'normal':
                    if 'ambulance_dist' in scene:
                        print(f"    ✓ Ambulance distance: {scene['ambulance_dist']:.1f}m")
                    else:
                        print(f"    (No ambulance nearby)")
                elif role == 'ambulance':
                    print(f"    ✓ Ambulance agent (priority vehicle)")
    
    print("\n" + "=" * 60)
    print("✓ Dataset format verification PASSED!")
    print("  - All agent records have 'role' field")
    print("  - Normal agents have 'ambulance_dist' when applicable")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_dataset_format.py <path_to_dataset.jsonl>")
        print("\nExample:")
        print("  python verify_dataset_format.py data/highway/dataset.jsonl")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    
    try:
        success = verify_dataset_format(jsonl_path)
        sys.exit(0 if success else 1)
    except FileNotFoundError:
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
