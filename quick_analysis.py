import json
from pathlib import Path
import statistics
from collections import Counter

print("Frame Analysis starting...")

results = {}
base_dir = Path("output/original_optimized")
if base_dir.exists():
    for session_dir in base_dir.iterdir():
        if session_dir.is_dir():
            reports_dir = session_dir / "reports"
            if reports_dir.exists():
                for report_file in reports_dir.glob("*_report.json"):
                    try:
                        with open(report_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            filename = report_file.stem.replace("_report", "")
                            results[filename] = data
                    except:
                        pass

print(f"Found {len(results)} files")

oversegmentation = []
undersegmentation = []
inconsistent_sizes = []
frame_counts = Counter()

for filename, data in results.items():
    if "frames" not in data or not data["frames"]:
        continue
        
    frames = data["frames"]
    frame_count = len(frames)
    frame_counts[frame_count] += 1
    
    areas = []
    for frame in frames:
        if "area" in frame:
            areas.append(frame["area"])
        elif "size" in frame:
            try:
                w, h = map(int, frame["size"].split("x"))
                areas.append(w * h)
            except:
                continue
    
    if not areas:
        continue
        
    area_mean = statistics.mean(areas)
    area_std = statistics.stdev(areas) if len(areas) > 1 else 0
    area_cv = area_std / area_mean if area_mean > 0 else 0
    
    if frame_count > 32:
        oversegmentation.append((filename, frame_count))
    elif frame_count < 2:
        undersegmentation.append(filename)
    elif area_cv > 1.0:
        inconsistent_sizes.append((filename, area_cv))

problematic = len(oversegmentation) + len(undersegmentation) + len(inconsistent_sizes)
total = len(results)

print(f"Problematic files: {problematic}/{total} ({problematic/total*100:.1f}%)")
print(f"Oversegmentation: {len(oversegmentation)}")
print(f"Undersegmentation: {len(undersegmentation)}")  
print(f"Inconsistent sizes: {len(inconsistent_sizes)}")

print("Top oversegmentation cases:")
for filename, count in sorted(oversegmentation, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {filename}: {count} frames")

print("Top inconsistent size cases:")
for filename, cv in sorted(inconsistent_sizes, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {filename}: CV = {cv:.2f}")

