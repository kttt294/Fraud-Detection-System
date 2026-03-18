import requests
import json

def test_api():
    url = "http://localhost:8000/verify"
    
    # Giả lập một giao dịch bị BLOCK
    payload = {
        "amount": 999.0,
        "time_val": 1.0, 
        "v_features": [
            0.0, 0.0, 0.0, 10.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            -15.0,
            10.0,
            -15.0,
            0.0,
            -15.0,
            0.0, 0.0,
            -15.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    }

    # Giả lập một giao dịch được APPROVE
    '''
    payload = {
        "amount": 1.0,
        "time_val": 45000.0,
        "v_features": [-50] * 28
    }
    '''

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
