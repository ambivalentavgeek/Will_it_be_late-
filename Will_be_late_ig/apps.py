"""
Will It Be Late? — Flask Backend API v3
- No internet/geocoding needed
- User types city name → matched locally
- Returns deliverable restaurants with ML predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, json, numpy as np, os
from math import radians, cos, sin, asin, sqrt

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load(fname):
    with open(os.path.join(BASE_DIR, fname), 'rb' if fname.endswith('.pkl') else 'r') as f:
        return pickle.load(f) if fname.endswith('.pkl') else json.load(f)

bundle   = load('best_model.pkl')
encoders = load('encoders.pkl')
scaler   = load('scaler.pkl')
model_metrics     = load('model_metrics.json')
feature_importance= load('feature_importance.json')
LATE_THRESHOLD    = load('threshold.json')['late_threshold_min']
CITY_DATA         = load('city_restaurants.json')

model      = bundle['model']
MODEL_NAME = bundle['name']
USE_SCALER = bundle['scaled']

# ── City keyword map ──────────────────────────────────────
# Each city has a list of keywords that match user input
CITY_KEYWORDS = {
    'Bangalore':   ['bangalore','bengaluru','koramangala','indiranagar','whitefield','btm','jayanagar','hsr','malleswaram'],
    'Hyderabad':   ['hyderabad','banjara','kukatpally','hitech','gachibowli','secunderabad','jubilee','madhapur','kondapur'],
    'Mumbai':      ['mumbai','andheri','bandra','dadar','borivali','thane','kurla','malad','powai','juhu','goregaon'],
    'Chennai':     ['chennai','madras','anna nagar','velachery','adyar','t nagar','nungambakkam','porur','chrompet'],
    'Pune':        ['pune','kothrud','aundh','baner','wakad','hinjewadi','viman nagar','kharadi','shivaji nagar'],
    'Kolkata':     ['kolkata','calcutta','salt lake','park street','ballygunge','howrah','dum dum','jadavpur'],
    'Indore':      ['indore','vijay nagar','palasia','rajwada','bhawarkua','mahalaxmi'],
    'Jaipur':      ['jaipur','civil lines','malviya nagar','vaishali nagar','mansarovar','c scheme'],
    'Surat':       ['surat','adajan','vesu','athwa','katargam','udhna'],
    'Mysuru':      ['mysuru','mysore','vijayanagar','kuvempunagar','nazarbad'],
    'Coimbatore':  ['coimbatore','rs puram','gandhipuram','peelamedu','saibaba colony'],
    'Ranchi':      ['ranchi','doranda','harmu','kanke','lalpur'],
    'Vadodara':    ['vadodara','baroda','alkapuri','gotri','fatehgunj'],
    'Agra':        ['agra','taj ganj','civil lines','shahganj','sikandra'],
    'Ludhiana':    ['ludhiana','sarabha nagar','model town','brs nagar','pakhowal'],
    'Kanpur':      ['kanpur','swaroop nagar','kakadeo','kidwai nagar','gwaltoli'],
    'Dehradun':    ['dehradun','rajpur','prem nagar','dalanwala','karanpur'],
    'Goa':         ['goa','panjim','panaji','margao','mapusa','calangute','baga','vasco'],
    'Aurangabad':  ['aurangabad','cidco','garkheda','waluj','osmanpura'],
    'Kochi':       ['kochi','cochin','ernakulam','fort kochi','edapally','kakkanad','tripunithura'],
    'Bhopal':      ['bhopal','mp nagar','arera colony','kolar','misrod','habibganj'],
    'Allahabad':   ['allahabad','prayagraj','civil lines','george town','naini','katra'],
}

def match_city(address_lower):
    """Find best city match from user's typed address."""
    for city_name, keywords in CITY_KEYWORDS.items():
        for kw in keywords:
            if kw in address_lower:
                return city_name
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2*R*asin(sqrt(a))

def safe_encode(key, val, fallback=0):
    le = encoders.get(key)
    if not le: return fallback
    try: return int(le.transform([val])[0])
    except: return fallback

def predict_restaurant(dist_km, weather, traffic, food_type, order_hour, day_of_week):
    food_c = {'Meal':3,'Buffet':4,'Snack':2,'Drinks':1}.get(food_type, 2)
    is_peak = 1 if (12<=order_hour<=14 or 19<=order_hour<=22) else 0
    is_weekend = 1 if day_of_week>=5 else 0

    X = np.array([[28, 4.0, dist_km, 10, 1, 0, food_c, order_hour, is_peak,
                   day_of_week, is_weekend,
                   safe_encode('Weather_conditions', weather),
                   safe_encode('Road_traffic_density', traffic),
                   safe_encode('Type_of_order', food_type),
                   safe_encode('Type_of_vehicle', 'motorcycle'),
                   safe_encode('Festival', 'No'),
                   safe_encode('City', 'Metropolitian')]])

    Xi = scaler.transform(X) if USE_SCALER else X
    prob = float(model.predict_proba(Xi)[0][1])

    base  = 15 + dist_km*1.2
    base += {'Low':0,'Medium':4,'High':8,'Jam':14}.get(traffic, 4)
    base += {'Sunny':0,'Cloudy':1,'Windy':2,'Fog':4,'Sandstorms':6,'Stormy':8}.get(weather, 0)

    return {'is_late': prob>=0.5, 'probability': round(prob*100,1), 'estimated_minutes': round(base)}

print(f"✅ Loaded {MODEL_NAME} | {len(CITY_DATA)} cities | threshold={LATE_THRESHOLD}min")

# ── Routes ────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({'status':'ok','model':MODEL_NAME,'cities':len(CITY_DATA),'threshold':LATE_THRESHOLD})

@app.route('/api/cities')
def cities():
    return jsonify([{'city':c['city'],'code':c['code']} for c in CITY_DATA])

@app.route('/api/geocode', methods=['POST'])
def geocode():
    data = request.get_json()
    address = data.get('address','').strip()
    order_hour   = int(data.get('order_hour', 13))
    day_of_week  = int(data.get('order_day_of_week', 0))

    if not address:
        return jsonify({'error': 'Please enter an address.'}), 400

    # Match city from typed address
    city_name = match_city(address.lower())
    if not city_name:
        # fallback: check if any city name itself appears in the address
        for c in CITY_DATA:
            if c['city'].lower() in address.lower():
                city_name = c['city']
                break

    if not city_name:
        available = ', '.join([c['city'] for c in CITY_DATA])
        return jsonify({
            'error': f'City not recognized. Try including the city name. Available: {available}'
        }), 404

    # Find city data
    city = next((c for c in CITY_DATA if c['city']==city_name), None)
    if not city:
        return jsonify({'error': f'No data for {city_name}'}), 404

    # Use city center as user location (no real geocoding needed)
    user_lat = city['lat']
    user_lon = city['lon']

    # All restaurants in this city are "deliverable" — sort by dist from center
    restaurants = []
    for r in city['restaurants']:
        dist = haversine(user_lat, user_lon, r['lat'], r['lon'])
        pred = predict_restaurant(dist, r.get('weather','Sunny'),
                                  r.get('traffic','Medium'), r.get('food_type','Meal'),
                                  order_hour, day_of_week)
        restaurants.append({**r, 'delivery_distance_km': round(dist,2), **pred})

    restaurants.sort(key=lambda x: x['delivery_distance_km'])

    return jsonify({
        'user_lat': user_lat, 'user_lon': user_lon,
        'display_name': address,
        'city': city_name,
        'city_lat': city['lat'], 'city_lon': city['lon'],
        'restaurants': restaurants,
        'total_deliverable': len(restaurants),
    })

@app.route('/api/metrics')
def metrics():
    return jsonify({'models':model_metrics,'best_model':MODEL_NAME,
                    'feature_importance':feature_importance,'late_threshold':LATE_THRESHOLD})

if __name__ == '__main__':
    print("🚀 Will It Be Late? v3 — http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
