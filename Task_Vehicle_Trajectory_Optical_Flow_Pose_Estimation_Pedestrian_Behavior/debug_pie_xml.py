import os
import xml.etree.ElementTree as ET
from glob import glob

# Đường dẫn đến thư mục annotations
ANN_DIR = "data/raw/pie/annotations"

# Tìm một file XML bất kỳ
xml_files = glob(os.path.join(ANN_DIR, "**", "*.xml"), recursive=True)
if not xml_files:
    print("Không tìm thấy file XML")
    exit()

# Chọn file đầu tiên
sample_file = xml_files[0]
print(f"📄 Kiểm tra file: {sample_file}")

# Parse và in cấu trúc
try:
    tree = ET.parse(sample_file)
    root = tree.getroot()
    
    # In tên root và các thẻ con
    print(f"\n🔹 Root tag: {root.tag}")
    print(f"🔹 Root attributes: {root.attrib}")
    print(f"🔹 Số lượng con trực tiếp: {len(root)}")
    
    print("\n🔸 Các thẻ con của root:")
    for child in root[:5]:  # chỉ lấy 5 thẻ đầu để xem mẫu
        print(f"   - {child.tag} (attributes: {child.attrib})")
        # In thêm 1 cấp nếu có
        for sub in child[:3]:
            print(f"      - {sub.tag} (attrib: {sub.attrib})")
    
    # Tìm kiếm các thẻ liên quan đến object
    print("\n🔹 Tìm kiếm các thẻ object:")
    for obj in root.findall(".//object"):
        print(f"   Object: {obj.attrib}")
        break
    else:
        print("   ❌ Không tìm thấy thẻ <object>")
    
    # Tìm kiếm các thẻ tracklet
    print("\n🔹 Tìm kiếm các thẻ tracklet:")
    for tracklet in root.findall(".//tracklet"):
        print(f"   Tracklet: {tracklet.attrib}")
        break
    else:
        print("   ❌ Không tìm thấy thẻ <tracklet>")
    
    # Tìm kiếm các thẻ box
    print("\n🔹 Tìm kiếm các thẻ box:")
    boxes = root.findall(".//box")
    if boxes:
        print(f"   Tìm thấy {len(boxes)} thẻ <box>")
        print(f"   Box mẫu: {boxes[0].attrib}")
    else:
        print("   ❌ Không tìm thấy thẻ <box>")
        
    # In toàn bộ XML để xem
    print("\n🔹 Toàn bộ XML (1000 ký tự đầu):")
    print(ET.tostring(root, encoding='unicode', method='xml')[:1000])
    
except Exception as e:
    print(f"❌ Lỗi: {e}")