from pathlib import Path

# Set these paths as needed
images_dir      = Path("/home/gautamarora/ultra/dark_face_nonsplit/images")
labels_raw_dir  = Path("/home/gautamarora/ultra/dark_face_nonsplit/label")
labels_yolo_dir = Path("/home/gautamarora/ultra/yololabels")
labels_yolo_dir.mkdir(parents=True, exist_ok=True)

# Image size (DarkFace uses full HD resolution)
img_width, img_height = 1280, 720

# Process each annotation file
for txt_file in sorted(labels_raw_dir.glob("*.txt")): #loops through all `.txt` files inside the directory
    with open(txt_file, "r") as f:
        lines = f.read().strip().split("\n")

    if not lines or not lines[0].isdigit():
        print(f"Skipping invalid or empty file: {txt_file.name}")
        continue

    num_boxes = int(lines[0]) # no of bbox in img
    yolo_lines = []           # stores all bbox converted to yolo format

    for line in lines[1:num_boxes+1]:
        try:
            x1, y1, x2, y2 = map(int, line.strip().split())
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        except ValueError:
            print(f"Invalid box in {txt_file.name}: {line}")

    # Save the YOLO formatted file
    output_file = labels_yolo_dir / txt_file.name
    with open(output_file, "w") as out:
        out.write("\n".join(yolo_lines))

print("Conversion complete!")
