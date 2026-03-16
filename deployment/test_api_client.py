import requests
import json

def test_api():
    url = "http://localhost:8000/verify"
    
    # Giả lập một giao dịch (Sử dụng các giá trị V ngẫu nhiên)
    payload = {
        "amount": 2500.0,
        "time_offset": 45000.0,
        "v_features": [0.1] * 28  # Giả lập 28 đặc trưng V
    }
    
    print("--- Đang gửi yêu cầu xác thực giao dịch đến API ---")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("\n[RESULT FROM BANKING CORE]:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Lỗi API: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Không thể kết nối đến API. Đã bật backend.py chưa?\nLỗi: {e}")

if __name__ == "__main__":
    test_api()
