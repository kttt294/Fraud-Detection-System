import os
import psycopg2
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta

def seed_database():
    load_dotenv()
    
    # Kết nối DB
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode=os.getenv("DB_SSLMODE")
        )
        cur = conn.cursor()
        
        print("--- Đang khởi tạo 5 dòng dữ liệu mẫu (Seed Data) ---")
        
        # Dữ liệu mẫu
        sources = ["API (User App)", "API (Mobile App)", "API (Web Checkout)"]
        
        for i in range(5):
            amount = round(random.uniform(500, 5000), 2)
            prob = round(random.uniform(0.70, 0.99), 4)
            time_val = random.uniform(1000, 100000)
            source = random.choice(sources)
            # Tạo thời gian ngẫu nhiên trong 1 giờ qua
            created_at = datetime.now() - timedelta(minutes=random.randint(1, 60))
            
            cur.execute(
                "INSERT INTO api_fraud_logs (amount, time_val, fraud_probability, source, created_at) VALUES (%s, %s, %s, %s, %s)",
                (amount, time_val, prob, source, created_at)
            )
        
        conn.commit()
        print("✅ Đã chèn thành công 5 giao dịch gian lận vào bảng fraud_logs.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Lỗi kết nối hoặc chèn dữ liệu: {e}")

if __name__ == "__main__":
    seed_database()
