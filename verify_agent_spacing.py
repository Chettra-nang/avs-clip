#!/usr/bin/env python3
"""
Verify agent spacing and ambulance detection in heterogeneous mode.

This script analyzes collected data to check:
1. Initial agent spacing
2. Ambulance detection rates
3. Yielding behavior when ambulance is nearby
"""

import json
import numpy as np
import sys

def analyze_dataset(dataset_path):
    """Analyze agent spacing and behavior from collected dataset."""
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        frames = [json.loads(line) for line in f]
    
    print("=" * 70)
    print("AGENT SPACING AND BEHAVIOR ANALYSIS")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Total frames: {len(frames)}\n")
    
    # 1. Initial Spacing Analysis
    print("1. INITIAL AGENT SPACING")
    print("-" * 70)
    first_frame = frames[0]
    positions = []
    for i, agent in enumerate(first_frame['agents']):
        pos = agent['position']
        positions.append(pos)
        print(f"   Agent {i} ({agent['role']:9s}): x={pos[0]:7.1f}m, y={pos[1]:5.1f}m")
    
    print(f"\n   Inter-agent distances:")
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            print(f"   Agent {i} <-> Agent {j}: {dist:6.1f}m")
    
    # 2. Ambulance Detection Analysis
    print(f"\n2. AMBULANCE DETECTION ANALYSIS")
    print("-" * 70)
    
    detections_by_agent = {1: [], 2: [], 3: []}
    close_encounters = {1: 0, 2: 0, 3: 0}  # Within 50m
    
    for frame in frames:
        amb_pos = np.array(frame['agents'][0]['position'])
        
        for i in range(1, 4):
            agent = frame['agents'][i]
            agent_pos = np.array(agent['position'])
            actual_dist = np.linalg.norm(amb_pos - agent_pos)
            
            if actual_dist < 50.0:
                close_encounters[i] += 1
                detected = 'ambulance_dist' in agent['scene']
                detections_by_agent[i].append({
                    'frame': frame['index'],
                    'actual_dist': actual_dist,
                    'detected': detected,
                    'action': agent['action_name']
                })
    
    for agent_id in [1, 2, 3]:
        detections = detections_by_agent[agent_id]
        if detections:
            detected_count = sum(1 for d in detections if d['detected'])
            print(f"\n   Agent {agent_id} (normal):")
            print(f"   - Close encounters (<50m): {len(detections)}")
            print(f"   - Ambulance detected: {detected_count}/{len(detections)} ({100*detected_count/len(detections):.1f}%)")
            
            # Show sample detections
            if detected_count > 0:
                print(f"   - Sample detections:")
                for d in detections[:3]:
                    if d['detected']:
                        print(f"     Frame {d['frame']}: dist={d['actual_dist']:.1f}m, action={d['action']}")
        else:
            print(f"\n   Agent {agent_id} (normal): No close encounters (<50m)")
    
    # 3. Yielding Behavior Analysis
    print(f"\n3. YIELDING BEHAVIOR ANALYSIS")
    print("-" * 70)
    
    yielding_actions = ['LANE_RIGHT', 'SLOWER', 'IDLE']
    aggressive_actions = ['LANE_LEFT', 'FASTER']
    
    for agent_id in [1, 2, 3]:
        detections = detections_by_agent[agent_id]
        if detections:
            detected_frames = [d for d in detections if d['detected']]
            if detected_frames:
                yielding_count = sum(1 for d in detected_frames if d['action'] in yielding_actions)
                aggressive_count = sum(1 for d in detected_frames if d['action'] in aggressive_actions)
                
                print(f"\n   Agent {agent_id} when ambulance detected:")
                print(f"   - Yielding actions: {yielding_count}/{len(detected_frames)} ({100*yielding_count/len(detected_frames):.1f}%)")
                print(f"   - Aggressive actions: {aggressive_count}/{len(detected_frames)} ({100*aggressive_count/len(detected_frames):.1f}%)")
    
    # 4. Data Quality Assessment
    print(f"\n4. DATA QUALITY ASSESSMENT")
    print("-" * 70)
    
    avg_quality = np.mean([f['quality'] for f in frames])
    print(f"   Average quality score: {avg_quality:.3f}")
    
    # Check if spacing is reasonable
    initial_distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            initial_distances.append(dist)
    
    avg_initial_dist = np.mean(initial_distances)
    print(f"   Average initial inter-agent distance: {avg_initial_dist:.1f}m")
    
    if avg_initial_dist > 400:
        print(f"\n   ⚠️  WARNING: Agents start very far apart (avg {avg_initial_dist:.0f}m)")
        print(f"   This is NORMAL for highway-env but may reduce ambulance interactions.")
        print(f"   Consider adjusting 'initial_spacing' in config if more interactions needed.")
    elif avg_initial_dist < 50:
        print(f"\n   ⚠️  WARNING: Agents start very close (avg {avg_initial_dist:.0f}m)")
        print(f"   This may cause immediate collisions. Consider increasing spacing.")
    else:
        print(f"\n   ✓ Agent spacing is reasonable for highway scenarios")
    
    # Check ambulance detection rate
    total_close_encounters = sum(close_encounters.values())
    if total_close_encounters > 0:
        total_detections = sum(len([d for d in detections_by_agent[i] if d['detected']]) 
                              for i in [1, 2, 3])
        detection_rate = total_detections / total_close_encounters
        
        if detection_rate > 0.8:
            print(f"   ✓ Ambulance detection working well ({detection_rate*100:.1f}%)")
        elif detection_rate > 0.5:
            print(f"   ⚠️  Ambulance detection moderate ({detection_rate*100:.1f}%)")
        else:
            print(f"   ❌ Ambulance detection low ({detection_rate*100:.1f}%)")
    else:
        print(f"   ⚠️  No close encounters in this dataset - agents stayed far apart")
        print(f"   Consider collecting more episodes or longer episodes")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = 'data_test/highway_heterogeneous/dataset.jsonl'
    
    try:
        analyze_dataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        print("Usage: python verify_agent_spacing.py [path/to/dataset.jsonl]")
        sys.exit(1)
