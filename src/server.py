import json
from datetime import datetime

import torch
from flask import Flask, request, make_response
from flask_cors import CORS

from src.model import TemperatureModel
from src.util import extract_date, date_range

app = Flask(__name__)
CORS(app)

epoch = 2997
BATCH_SIZE = 128

model = TemperatureModel()
model.load_state_dict(torch.load(f'best_models/epoch{epoch}_batchSize{BATCH_SIZE}_8layers_noOutputReLU_3channels.pt'))
model.eval()


@app.route("/forecast", methods=['POST'])
def forecast():
    year, month, day = extract_date(request.form['date'])
    inp = torch.tensor([year - 1961, month, day]).reshape(1, 3)
    out = model(inp).reshape(3).tolist()
    dictionary = {
        "dates": f"[{int(month)}/{int(day)}/{int(year)}]",
        "med": [out[0]],
        "max": [out[1]],
        "min": [out[2]],
    }
    return make_response(json.dumps(dictionary, indent=4), 200)


@app.route("/multiForecast", methods=['POST'])
def multi_forecast():
    year1, month1, day1 = extract_date(request.form['beginDate'], data_type=int)
    year2, month2, day2 = extract_date(request.form['endDate'], data_type=int)

    start_date = datetime(year1, month1, day1)
    end_date = datetime(year2, month2, day2)
    dates = date_range(start_date, end_date)
    inp = torch.cat([torch.tensor([float(date.year - 1961), float(date.month), float(date.day)]).reshape(1, 3) for date in dates])

    out = model(inp).reshape(-1, 3)
    dictionary = {
        "dates": [f"{date.month}/{date.day}/{date.year}" for date in dates],
        "med": out[:, 0].tolist(),
        "max": out[:, 1].tolist(),
        "min": out[:, 2].tolist()
    }
    return make_response(str(json.dumps(dictionary, indent=4)), 200)
