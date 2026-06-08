import requests
r = requests.post("http://127.0.0.1:8000/predict",
                  json={"BlockHeight":5800000,"TimeStamp":1518200000,"From":"0xabc123","To":"0xdef456","Value":1.5},
                  timeout=8)
print("status:", r.status_code)
print("text:", r.text)
