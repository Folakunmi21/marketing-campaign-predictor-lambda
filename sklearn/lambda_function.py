import pickle

#load the model

with open ('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba([customer])[0, 1]
    return float(result)

def lambda_handler(event, context):    
    print("Parameters:", event)
    customer = event['customer']
    prob = predict_single(customer)
    return {
        "response_probability": prob,
        "response": bool(prob >= 0.5)
    }