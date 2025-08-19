from jsonrpclib import Server
check_string = "I love india"

def main():
    conn = Server("http://localhost:1086")
    print(f"The sentiment is : {conn.check_sentiment(check_string)}")
    
if __name__ == "__main__":
    main()
