import requests

url = "http://127.0.0.1:5000/process"

data = {
  "model": "xgboost",
  "columns": ["step","customer","age","gender","zipcodeori","merchant","zipmerchant","category","amount","fraud"],
  "input": [
	  	[65,'C226946948',"'3'","'M'",'28007','M348934600','28007','es_transportation',27.51,0],
	    [156,'C1732568163',"'4'","'F'",'28007','M3697346','28007','es_leisure',257.93,1]
    ]
}

print("Getting available models...")

response = requests.get("http://localhost:5000/tasks")
response = response.json()

i = 0
for task in response:
	i+=1
	print(f"  {i} - {task}")

type = input(f"\nSelect model (1-{i}): ")

data["model"] = response[int(type)-1]

print(f"sending... {data['model']} | {len(data['input'])} rows")

response = requests.post(url, json=data).json()

print(f"response: {response}")
print(f"elapsed:  {response['time_taken_seconds']:.5f} seconds\n")

print(f"{'probability':<25} | prediction")
print("-" * 40)

for prob, pred in response["output"]:
	print(f"{prob:<25.22f} | {pred}")

# python test2.py