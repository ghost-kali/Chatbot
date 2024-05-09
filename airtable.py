import requests
AIRTABLE_BASE_ID = "appONLo1mGUZDlc3o"
AIRTABLE_API_KEY = "patRtv7tfjiUJuEX0.46b14546ad87b10c91df3b03cfa8a6b763b4732893e317a3bd19300a93fa0133"
AIRTABLE_TABLE_NAME = "chatbot"
endpoint = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

headers = {
"Authorization": f"Bearer {AIRTABLE_API_KEY}",
"Content-Type": "application/json"
}
def save_rec(user_query,response):
    data = {
  "records": [
          {
        "fields": {
            "Query": f"{user_query}",
            "Response": f"{response}"
        }
        },
    ]
    }

    r = requests.post(endpoint, json=data , headers=headers)
    r.json()


if __name__ == '__main__':
    save_rec("hello","hi there")