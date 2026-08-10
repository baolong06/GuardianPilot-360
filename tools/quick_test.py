import requests, base64, cv2, numpy as np
img = np.zeros((240, 320, 3), dtype=np.uint8)
ok, buf = cv2.imencode('.jpg', img)
url = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
r = requests.post('http://127.0.0.1:5000/api/analyze_lite', json={'image': url})
print(f'Status: {r.status_code}')
d = r.json()
print(f'face_found: {d["face_found"]}  alarm: {d["alarm_on"]}  state: {d["drowsiness_state"]}')
print(f'Keys: {list(d.keys())}')
