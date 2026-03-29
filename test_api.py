import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "cgpa": 7.5,
    "dsa": 3,
    "projects": 3,
    "internship": 1,
    "communication": 3,
    "aptitude": 70,
    "certifications": 2,
    "consistency": 7,
    "score": 70
}

response = requests.post(url, json=data)

print(response.json())