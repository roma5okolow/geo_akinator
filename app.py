import streamlit as st
import torch
from transformers import BertTokenizer, BertConfig
import pydeck as pdk

from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn


# Bounded regressor
class BoundedLatLonRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
	        nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, x):
        latlon = self.mlp(x)
        lat = latlon[:, 0] * 90
        lon = latlon[:, 1] * 180
        return torch.stack([lat, lon], dim=1)

class BertForLatLonCosineLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = BoundedLatLonRegressor(config.hidden_size)
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        latlon_pred = self.regressor(pooled_output)
        return latlon_pred


@st.cache_resource
def load_model():
    model_path = "./model"
    config = BertConfig.from_pretrained(model_path)
    model = BertForLatLonCosineLoss.from_pretrained(model_path, config=config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model.eval(), tokenizer


model, tokenizer = load_model()

st.title("🌍 City Location Predictor")

prompt = st.text_area("Enter a description of a city or place:", height=150)

if st.button("Predict Location") and prompt:
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        lat, lon = output[0].tolist()
        st.success(f"🌐 Predicted Coordinates: Latitude = {lat:.4f}°, Longitude = {lon:.4f}°")

        # Plot on map
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=3,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=[{"lat": lat, "lon": lon}],
                    get_position='[lon, lat]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=50000,
                )
            ],
        ))
