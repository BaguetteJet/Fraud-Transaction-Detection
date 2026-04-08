import requests

url = "http://127.0.0.1:5000/process"

data = {
  "model": "logistic_regression",
  "columns": ["customer_amount","customer_time_since_last_transaction","customer_transaction_count","customer_std_amount","customer_merchant_count","customer_category_count","customer_time_since_last_merchant_transaction","merchant_transaction_count","merchant_std_amount","merchant_time_since_last_transaction","global_std_amount","fraud","customer_log_amount_ratio","customer_zscore","merchant_log_amount_ratio","merchant_amount_zscore","merchant_fraud_rate","global_z_score","global_log_median_amount_ratio","age_'0'","age_'1'","age_'2'","age_'3'","age_'4'","age_'5'","age_'6'","age_'U'","gender_'E'","gender_'F'","gender_'M'","gender_'U'"],
  "input": [[0.057296839444050424,0.36305222919974256,1.6412233395802918,0.18870663595144999,1.6820335381641276,1.5942581297056568,-0.0796579464525722,1.961392964950962,-0.14339979811337802,4.025059631808929,-0.9705788236888468,0,0.7149879529054807,0.02579846072765269,0.9784300021582067,1.0158453894045987,0.0,0.05920293725175104,0.9803644609977813,0,0,0,1,0,0,0,0,0,0,1,0],
            [1.908949710646438,-0.5653801784432849,0.23260809102770483,0.5017035372289033,-1.116318302660477,-1.2796465064417373,-0.4341265049083296,-1.337698219126412,1.0113620900723748,8.112758302764068,-0.9712574076528085,1,1.6925744932337305,1.7983165369307186,0.6395747956977311,-0.30255208019991825,0.9338235294117647,1.9382750715216737,2.3597662244611186,0,0,0,0,1,0,0,0,0,1,0,0]
          ]
}

print("sending...\n", f"{data['model']} | {len(data['input'])} rows")

response = requests.post(url, json=data)

print("response:\n",response.json())