import requests

url = "http://127.0.0.1:5000/process"

data = {
  "model": "decision_tree",
  "columns": ["customer_amount","customer_time_since_last_transaction","customer_transaction_count","customer_std_amount","customer_merchant_count","customer_category_count","customer_time_since_last_merchant_transaction","merchant_transaction_count","merchant_std_amount","merchant_time_since_last_transaction","global_std_amount","fraud","customer_log_amount_ratio","customer_zscore","merchant_log_amount_ratio","merchant_amount_zscore","merchant_fraud_rate","global_z_score","global_log_median_amount_ratio","age_'0'","age_'1'","age_'2'","age_'3'","age_'4'","age_'5'","age_'6'","age_'U'","gender_'E'","gender_'F'","gender_'M'","gender_'U'"],
  "input": [[0.057296839444050424,0.36305222919974256,1.6412233395802918,0.18870663595144999,1.6820335381641276,1.5942581297056568,-0.0796579464525722,1.961392964950962,-0.14339979811337802,4.025059631808929,-0.9705788236888468,0,0.7149879529054807,0.02579846072765269,0.9784300021582067,1.0158453894045987,0.0,0.05920293725175104,0.9803644609977813,0,0,0,1,0,0,0,0,0,0,1,0]]
}

print("sending...\n", data)

response = requests.post(url, json=data)

print("response:\n",response.json())