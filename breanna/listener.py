from flask   import Flask, request, jsonify
from skimage import io
from breanna import ctr_model_management as cmm

app = Flask(__name__)

@app.route("/predict-ctr")
def predict_ctr():
    aggregators = request.args.getlist('aggregator')
    path2banner = request.args.get('path2banner')
    return jsonify( predict_ctr_dataframe(aggregators, path2banner) )

def predict_ctr_dataframe(aggregators, path2banner):
   
    aggregator2ctrmodel = {
        frozenset(['publisher']): 
            'publisher-randomforestregressor.p',
        frozenset(['operatingsystem']): 
            'os-randomforestregressor.p',
        frozenset(['device']): 
            'device-randomforestregressor.p',
        frozenset(['time']): 
            'event_hour-randomforestregressor.p',
        frozenset(['publisher', 'time']):
            'event_hour-publisher-randomforestregressor.p',
        frozenset(['operatingsystem', 'time']):
            'event_hour-os-randomforestregressor.p',
        frozenset(['device', 'time']):
            'event_hour-device-randomforestregressor.p',
        frozenset(['publisher', 'operatingsystem', 'time']):
            'event_hour-publisher-os-randomforestregressor.p',
        frozenset(['publisher', 'device', 'time']):
            'event_hour-publisher-device-randomforestregressor.p'
    }
    banner   = io.imread(path2banner)
    ctrmodel = cmm.load_CTRModel(aggregator2ctrmodel[frozenset(aggregators)])
    predict_ctr_df = ctrmodel.predict(banner)
    
    return {'predictions': predict_ctr_df.to_dict(orient='records')}