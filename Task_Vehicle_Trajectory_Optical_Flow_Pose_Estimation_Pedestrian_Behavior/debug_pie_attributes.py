import os
import xml.etree.ElementTree as ET
from glob import glob

ATTR_DIR = "data/raw/pie/annotations_attributes"

xml_files = glob(os.path.join(ATTR_DIR, "**", "*.xml"), recursive=True)
if not xml_files:
    print("❌ Không tìm thấy file XML trong annotations_attributes")
    exit()

sample_file = xml_files[0]
print(f"📄 Kiểm tra file: {sample_file}")

try:
    tree = ET.parse(sample_file)
    root = tree.getroot()
    
    print(f"\n🔹 Root tag: {root.tag}")
    print(f"🔹 Số lượng con trực tiếp: {len(root)}")
    
    # Tìm kiếm attribute action trong các thẻ
    print("\n🔹 Tìm kiếm attribute action trong box:")
    found = False
    for track in root.findall(".//track"):
        for box in track.findall("box"):
            for attr in box.findall("attribute"):
                attr_name = attr.get("name")
                if attr_name == "action":
                    print(f"   ✅ Found action: {attr.text} (in track {track.get('id')})")
                    found = True
                else:
                    print(f"   - Other attribute: {attr_name} = {attr.text}")
            if found:
                break
        if found:
            break
    
    if not found:
        print("   ❌ Không tìm thấy attribute action")
        
        # In cấu trúc của 1 track đầu tiên
        print("\n🔹 Cấu trúc track đầu tiên:")
        for track in root.findall(".//track")[:1]:
            print(f"   Track ID: {track.get('id')}")
            print(f"   Track label: {track.get('label')}")
            # In box đầu tiên
            box = track.find("box")
            if box is not None:
                print(f"   Box: frame={box.get('frame')}, xtl={box.get('xtl')}")
                print(f"   Box attributes:")
                for attr in box.findall("attribute"):
                    print(f"      - {attr.get('name')}: {attr.text}")
        
        # Tìm kiếm attribute ở bất kỳ vị trí nào
        print("\n🔹 Tìm kiếm attribute action trong toàn bộ XML:")
        for attr in root.findall(".//attribute"):
            name = attr.get("name")
            if name == "action":
                print(f"   ✅ Found action at: {attr.text}")
                # In parent
                parent = attr.getparent() if hasattr(attr, 'getparent') else None
                if parent is not None:
                    print(f"      Parent tag: {parent.tag}, parent attrib: {parent.attrib}")
    
    # In 500 ký tự XML đầu tiên
    print("\n🔹 500 ký tự XML đầu tiên:")
    print(ET.tostring(root, encoding='unicode', method='xml')[:500])
    
except Exception as e:
    print(f"❌ Lỗi: {e}")